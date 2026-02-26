import os
import torch
import torch.nn as nn
from transformers import Trainer
from transformers.trainer import is_sagemaker_mp_enabled, get_parameter_names, TRAINER_STATE_NAME, PREFIX_CHECKPOINT_DIR, logger, ExportableState, SaveStrategy
try:
    from transformers.trainer import ALL_LAYERNORM_LAYERS
except ImportError:
    ALL_LAYERNORM_LAYERS = [torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.modules.normalization.LayerNorm]
from training.train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3

class QwenSFTTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        opt_model = self.model
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [n for n in decay_parameters if "bias" not in n]
            visual_parameters = [n for n, _ in opt_model.named_parameters() if "visual" in n and "merger" not in n] if self.args.vision_lr else []
            merger_parameters = [n for n, _ in opt_model.named_parameters() if "merger" in n] if self.args.merger_lr else []
            special = merger_parameters + visual_parameters
            if special:
                grps = [{"params": [p for n, p in opt_model.named_parameters() if n in decay_parameters and n not in special and p.requires_grad], "weight_decay": self.args.weight_decay}, {"params": [p for n, p in opt_model.named_parameters() if n not in decay_parameters and n not in special and p.requires_grad], "weight_decay": 0.0}]
                for lr, names in [(self.args.vision_lr, visual_parameters), (self.args.merger_lr, merger_parameters)]:
                    if lr and names:
                        grps.extend([{"params": [p for n, p in opt_model.named_parameters() if n in decay_parameters and n in names and p.requires_grad], "weight_decay": self.args.weight_decay, "lr": lr}, {"params": [p for n, p in opt_model.named_parameters() if n not in decay_parameters and n in names and p.requires_grad], "weight_decay": 0.0, "lr": lr}])
            else:
                grps = [{"params": [p for n, p in opt_model.named_parameters() if n in decay_parameters and p.requires_grad], "weight_decay": self.args.weight_decay}, {"params": [p for n, p in opt_model.named_parameters() if n not in decay_parameters and p.requires_grad], "weight_decay": 0.0}]
            opt_cls, opt_kw = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = opt_cls(grps, **opt_kw)
            if opt_cls.__name__ == "Adam8bit":
                import bitsandbytes
                m = bitsandbytes.optim.GlobalOptimManager.get_instance()
                for mod in opt_model.modules():
                    if isinstance(mod, nn.Embedding):
                        m.register_module_override(mod, "weight", {"optim_bits": 32})
        return self.optimizer
    def _save_checkpoint(self, model, trial):
        if self.args.lora_enable:
            ckpt = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            if self.hp_search_backend is None and trial is None:
                self.store_flos()
            out = os.path.join(self._get_output_dir(trial=trial), ckpt)
            self.save_model(out, _internal_call=True)
            torch.save(get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=False), os.path.join(out, "non_lora_state_dict.bin"))
            if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
                best = os.path.join(self._get_output_dir(trial=trial), f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}")
                if os.path.exists(best):
                    self.state.best_model_checkpoint = best
            if not self.args.save_only_model:
                self._save_optimizer_and_scheduler(out)
                self._save_scaler(out)
                self._save_rng_state(out)
            if self.args.should_save:
                for cb in [c for c in self.callback_handler.callbacks + [self.control] if isinstance(c, ExportableState)]:
                    s = self.state.stateful_callbacks
                    n = cb.__class__.__name__
                    s[n] = s.get(n, []) + [cb.state()] if isinstance(s.get(n), list) else cb.state()
                self.state.save_to_json(os.path.join(out, TRAINER_STATE_NAME))
            if self.args.push_to_hub:
                self._push_from_checkpoint(out)
        else:
            super()._save_checkpoint(model, trial)

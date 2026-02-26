import transformers
import torch
import logging


def print_component_stats(name, trainable, total):
    if total > 0:
        ratio = 100 * trainable / total
    else:
        ratio = 0.0
    print(
        f"{name:20} | "
        f"Trainable: {trainable:>12,} | "
        f"Total: {total:>12,} | "
        f"Ratio: {ratio:>6.2f}%"
    )


def numel(p):
    return p.ds_numel if hasattr(p, "ds_numel") else p.numel()


def print_trainable_parameters(model, training_args):
    """Print detailed trainable parameters for each model component."""
    total_params = 0
    trainable_params = 0
    total_params_non_lora = 0
    trainable_params_non_lora = 0

    vision_encoder_total = 0
    vision_encoder_trainable = 0
    merger_total = 0
    merger_trainable = 0
    llm_total = 0
    llm_trainable = 0
    lora_params = 0

    if training_args.lora_enable:
        model = model.base_model.model

    for name, param in model.named_parameters():
        param_count = numel(param)
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count

        if 'lora_' in name:
            lora_params += param_count
            assert param.requires_grad, f"LoRA parameter {name} should be trainable"
            continue

        total_params_non_lora += param_count
        if param.requires_grad:
            trainable_params_non_lora += param_count

        if name.startswith('visual.merger'):
            merger_total += param_count
            if param.requires_grad:
                merger_trainable += param_count
        elif name.startswith('visual.'):
            vision_encoder_total += param_count
            if param.requires_grad:
                vision_encoder_trainable += param_count
        elif name.startswith('model.') or name.startswith('lm_head'):
            llm_total += param_count
            if param.requires_grad:
                llm_trainable += param_count
        else:
            print(f"Unrecognized parameter name: {name}.")

    print("=" * 80)
    print("MODEL PARAMETER ANALYSIS")
    print("=" * 80)
    print_component_stats("Vision Encoder", vision_encoder_trainable, vision_encoder_total)
    print_component_stats("Merger", merger_trainable, merger_total)
    print_component_stats("LLM", llm_trainable, llm_total)
    print_component_stats("Total (non-LoRA)", trainable_params_non_lora, total_params_non_lora)
    print_component_stats("LoRA", lora_params, lora_params)
    print_component_stats("Total (include LoRA)", trainable_params, total_params)
    print("=" * 80)


def maybe_zero_3(param, ignore_status=False, name=None, device=torch.device('cpu')):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if type(device) is str:
        device = torch.device(device)
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach()
    else:
        param = param.detach()
    if device == param.device:
        return param.clone()
    else:
        return param.to(device)


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)
        trainer.model.config.save_pretrained(output_dir)

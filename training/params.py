from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as HFTrainingArguments
from trl import GRPOConfig as GRPOConfigTRL


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="qwen3-vl-8b")
    model_name_or_path: Optional[str] = field(default=None)
    conv_type: Optional[str] = field(default="chatml")


@dataclass
class TrainingArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    max_seq_length: int = field(
        default=32768,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated).",
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type. One of `fp4` or `nf4`."},
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})

    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: Optional[str] = field(
        default=None,
        metadata={"help": "List of namespan to exclude for LoRA"},
    )
    num_lora_modules: int = -1
    use_liger: bool = True


@dataclass
class GRPOArguments(GRPOConfigTRL):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type. One of `fp4` or `nf4`."},
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})

    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: Optional[str] = field(
        default=None,
        metadata={"help": "List of namespan to exclude for LoRA"},
    )
    num_lora_modules: int = -1

    reward_funcs: str = field(
        default="tiou",
        metadata={"help": "Comma-separated list of reward functions to use."},
    )
    reward_weights: Optional[list[float]] = field(default=None)
    scale_rewards: bool = field(default=True)
    loss_type: str = field(default="bnpo")
    beta: float = field(default=0.0)
    num_iterations: int = field(default=1)

    use_liger: bool = field(default=False)
    use_liger_loss: bool = field(default=False)
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_completion_length: int = 512
    num_generations: int = 8


@dataclass
class DataArguments:
    data_path: Optional[str] = field(default=None)
    lazy_preprocess: bool = False

    datasets: Optional[str] = field(default=None)
    min_video_len: int = -1
    max_video_len: int = -1
    min_num_words: int = -1
    max_num_words: int = -1

    min_tokens: int = 64
    total_tokens: int = 14336
    fps: float = 2.0
    fps_max_frames: Optional[int] = None

    raw_anno_path: Optional[str] = field(default=None)
    fixed_gaussian_sampling: bool = field(default=False)
    gaussian_filter_mean: Optional[float] = None
    gaussian_filter_std: Optional[float] = None
    target_size: int = 2500

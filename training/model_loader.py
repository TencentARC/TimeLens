"""Model/config/processor loader for TimeLens-8B (Qwen3) and TimeLens-7B (Qwen2.5-VL)."""

from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor


def _validate_model_path(model_path: str) -> None:
    model_path_lower = model_path.lower()
    supported = (
        "qwen3" in model_path_lower
        or "timelens-8b" in model_path_lower
        or "qwen2.5-vl" in model_path_lower
        or "qwen2.5_vl" in model_path_lower
        or "timelens-7b" in model_path_lower
    )
    if not supported:
        raise ValueError(
            f"Unsupported model_path={model_path!r}. "
            "Expected Qwen3-VL/TimeLens-8B or Qwen2.5-VL/TimeLens-7B."
        )


def get_model_class(model_path: str):
    _validate_model_path(model_path)
    return AutoModelForImageTextToText


def get_config_class(model_path: str):
    _validate_model_path(model_path)
    return AutoConfig


def get_processor_class(model_path: str):
    _validate_model_path(model_path)
    return AutoProcessor

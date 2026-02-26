"""Model/config/processor loader for TimeLens-8B (Qwen3 only)."""

from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor


def _validate_model_path(model_path: str) -> None:
    model_path_lower = model_path.lower()
    if "qwen3" not in model_path_lower and "timelens-8b" not in model_path_lower:
        raise ValueError(
            f"Only Qwen3-VL/TimeLens-8B is supported, got model_path={model_path!r}."
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

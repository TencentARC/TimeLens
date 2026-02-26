from dataclasses import dataclass
from typing import List
import warnings

import torch


IGNORE_INDEX = -100


@dataclass
class Conversation:
    style: str
    system: str
    roles: List[str]
    seps: List[str]
    messages: List[str]


def get_conv(conv_type: str) -> Conversation:
    if conv_type != "chatml":
        raise ValueError(f"Unsupported conversation type: {conv_type}")
    return Conversation(
        style="chatml",
        system="<|im_start|>system\nYou are a helpful assistant.",
        roles=("\n<|im_start|>user\n", "\n<|im_start|>assistant\n"),
        seps=("<|im_end|>", "<|im_end|>"),
        messages=[],
    )


def preprocess_chatml(input_ids: torch.Tensor, text: str, tokenizer) -> torch.Tensor:
    conv = get_conv("chatml")
    rounds = [m + conv.seps[0] for m in text.split(conv.seps[0])]

    last_round = rounds[-1]
    if (
        last_round == conv.seps[0]
        or last_round.strip() == conv.seps[0]
        or last_round == "\n" + conv.seps[0]
    ):
        rounds = rounds[:-1]
    else:
        raise ValueError(f"Unexpected chatml trailing round: {last_round!r}")

    has_system_in_text = "<|im_start|>system" in text
    if (len(rounds) % 2 == 1) != has_system_in_text:
        raise ValueError(
            "Rounds count mismatch for chatml formatting: "
            f"len(rounds)={len(rounds)}, has_system={has_system_in_text}."
        )

    if has_system_in_text:
        rounds = ["".join(rounds[:3])] + [
            "".join(rounds[i : i + 2]) for i in range(3, len(rounds), 2)
        ]
    else:
        rounds = ["".join(rounds[i : i + 2]) for i in range(0, len(rounds), 2)]

    if text.endswith("\n") and rounds and not rounds[-1].endswith("\n"):
        rounds[-1] += "\n"

    labels = input_ids.clone()
    sep = conv.seps[0] + conv.roles[1]
    cur_len = 0

    has_vision_tokens = any(
        token in text
        for token in (
            "<|video_pad|>",
            "<|image_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
        )
    )

    for rou in rounds:
        if not rou:
            break
        ins = sep.join(rou.split(sep)[:-1]) + sep
        if has_vision_tokens:
            rou_len = tokenizer(
                rou, return_length=True, add_special_tokens=False
            ).length[0]
            ins_len = tokenizer(
                ins, return_length=True, add_special_tokens=False
            ).length[0]
        else:
            rou_len = tokenizer(rou, return_length=True).length[0]
            ins_len = tokenizer(ins, return_length=True).length[0]

        labels[cur_len : cur_len + ins_len] = IGNORE_INDEX
        cur_len += rou_len

    if has_vision_tokens and labels.size(0) != cur_len:
        im_start_token = tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_token_ids = tokenizer.encode("assistant", add_special_tokens=False)
        input_ids_list = input_ids.tolist()
        assistant_start_pos = None

        for idx in range(len(input_ids_list) - len(assistant_token_ids)):
            if input_ids_list[idx] != im_start_token:
                continue
            match = True
            for j, token_id in enumerate(assistant_token_ids):
                if (
                    idx + 1 + j >= len(input_ids_list)
                    or input_ids_list[idx + 1 + j] != token_id
                ):
                    match = False
                    break
            if match:
                assistant_start_pos = idx + 1 + len(assistant_token_ids) + 1
                break

        if assistant_start_pos is not None:
            labels[:assistant_start_pos] = IGNORE_INDEX
        else:
            warnings.warn(
                "Could not find assistant position in input_ids. "
                "Using token-length masking fallback."
            )
            warnings.warn(
                f"Tokenization mismatch for vision sample: {labels.size(0)} vs {cur_len}."
            )
    elif labels.size(0) != cur_len:
        warnings.warn(f"Tokenization mismatch: {labels.size(0)} vs {cur_len}.")

    return labels


def preprocess(input_ids: torch.Tensor, text: str, tokenizer, conv_type: str):
    if conv_type != "chatml":
        raise ValueError(f"Unsupported conversation type: {conv_type}")
    return preprocess_chatml(input_ids, text, tokenizer)

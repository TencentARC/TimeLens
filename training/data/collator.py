import warnings

import torch
from torch.nn.utils.rnn import pad_sequence


IGNORE_INDEX = -100


class HybridDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [d["input_ids"] for d in batch]
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        labels = [d["labels"] for d in batch]
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        if input_ids.size() != labels.size():
            raise ValueError(
                f"input_ids and labels must have same shape, got {input_ids.size()} vs {labels.size()}."
            )

        seq_len = input_ids.size(1)
        max_len = self.tokenizer.model_max_length
        if seq_len > max_len:
            warnings.warn(
                f"Input sequence length exceeds tokenizer max length: {seq_len} > {max_len}"
            )
            input_ids = input_ids[:, :max_len]
            labels = labels[:, :max_len]

        data = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
        }

        for key in (
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
        ):
            if key in batch[0]:
                data[key] = torch.cat([d[key] for d in batch])

        if "second_per_grid_ts" in batch[0]:
            data["second_per_grid_ts"] = [t for d in batch for t in d["second_per_grid_ts"]]

        return data

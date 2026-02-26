from itertools import accumulate

from torch.utils.data import Dataset

from training.data.grounding import GroundingDataset


def _build_default_filter_args(target_size: int):
    ranges = [(i, i + 30) for i in range(0, 240, 30)] + [(240, float("inf"))]
    per_range = int(target_size / len(ranges))
    return {
        "filter_range": ranges,
        "filter_target_size": [per_range] * len(ranges),
    }


class HybridDataset(Dataset):
    """Minimal hybrid dataset wrapper for TimeLens-8B training."""

    def __init__(
        self,
        processor,
        model_config,
        model_args,
        data_args,
        training_args,
        training_mode="sft",
    ):
        super().__init__()
        if not data_args.datasets:
            raise ValueError("data_args.datasets is required.")

        dataset_names = [name.strip() for name in data_args.datasets.split(",") if name.strip()]
        datasets = []
        for name in dataset_names:
            if name in ("gemini_refined_data", "timelens-100k"):
                filter_args = _build_default_filter_args(data_args.target_size)
                dataset_name = name
            elif name == "filtered_hybrid":
                filter_args = _build_default_filter_args(data_args.target_size)
                dataset_name = "filtered_hybrid"
            else:
                raise ValueError(
                    f"Unsupported dataset name: {name}. "
                    "Supported: gemini_refined_data, timelens-100k, filtered_hybrid."
                )

            datasets.append(
                GroundingDataset(
                    processor=processor,
                    model_args=model_args,
                    data_args=data_args,
                    training_args=training_args,
                    dataset_name=dataset_name,
                    filter_args=filter_args,
                    training_mode=training_mode,
                )
            )

        cum_length = [0] + list(accumulate([len(d) for d in datasets]))
        self.idx_ranges = [[cum_length[i], cum_length[i + 1]] for i in range(len(cum_length) - 1)]
        self.datasets = datasets

    def __len__(self):
        return self.idx_ranges[-1][-1]

    def __getitem__(self, idx):
        for (start, end), dataset in zip(self.idx_ranges, self.datasets):
            if start <= idx < end:
                return dataset[idx - start]
        raise IndexError(f"Index out of range: {idx}")

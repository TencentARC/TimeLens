import copy
import random
from pathlib import Path

import nncore
import numpy as np
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

from timelens.dataset.timelens_data import TimeLens100KDataset, parse_query
from training.data.preprocess import preprocess

GROUNDING_PROMPT = (
    "Please find the visual event described by the sentence '{}', determining its starting and ending times. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'."
)

AUDIO_QUERY_KEYWORDS = {
    "hear",
    "heard",
    "hears",
    "hearing",
    "sound",
    "sounded",
    "sounds",
    "sounding",
    "audio",
}


def _is_audio_related_query(query: str) -> bool:
    words = query.strip("?").lower().split()
    return any(keyword in words for keyword in AUDIO_QUERY_KEYWORDS)


def _normalize_spans(span):
    if isinstance(span, tuple):
        return [list(span)]
    if isinstance(span, list) and len(span) > 0 and isinstance(span[0], (list, tuple)):
        return [list(s) for s in span]
    if isinstance(span, list) and len(span) == 2 and isinstance(span[0], (int, float)):
        return [span]
    raise ValueError(f"Unsupported span format: {span}")


def _format_response(spans):
    return (
        "The event happens in "
        + ", ".join([f"{s:.1f} - {e:.1f} seconds" for s, e in spans])
        + "."
    )


def _build_video_content(anno, data_args, include_video_range=False):
    content = {
        "type": "video",
        "video": anno["video_path"],
        "min_pixels": int(data_args.min_tokens * 32 * 32),
        "total_pixels": int(data_args.total_tokens * 32 * 32),
        "fps": float(data_args.fps),
    }
    if include_video_range:
        content["video_start"] = anno.get("video_start")
        content["video_end"] = anno.get("video_end")
    if getattr(data_args, "fps_max_frames", None) is not None:
        content["max_frames"] = int(data_args.fps_max_frames)
    return content


def _load_filtered_annos(path: str):
    loaded = nncore.load(path)
    if isinstance(loaded, dict):
        loaded = [loaded]
    if loaded is None:
        return []
    annos = []
    for raw in loaded:
        if "source" not in raw or "query" not in raw:
            continue
        annos.append(
            {
                "source": raw["source"],
                "data_type": raw.get("data_type", "grounding"),
                "video_path": raw["video_path"],
                "duration": raw["duration"],
                "query": parse_query(raw["query"]),
                "span": raw["span"],
                "iou": raw.get("iou"),
                "pred": raw.get("pred"),
                "answer": raw.get("answer"),
            }
        )
    return annos


class GroundingDataset(Dataset):
    def __init__(
        self,
        processor,
        model_args,
        data_args,
        training_args,
        dataset_name: str,
        filter_args=None,
        training_mode: str = "sft",
    ):
        super().__init__()
        self.processor = processor
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.training_mode = training_mode

        if dataset_name in ("gemini_refined_data", "timelens-100k"):
            base_annos = TimeLens100KDataset.load_annos(split="train")
            if dataset_name == "gemini_refined_data":
                raw_annos = [
                    anno
                    for anno in base_annos
                    if not _is_audio_related_query(anno["query"])
                ]
            else:
                raw_annos = base_annos
        elif dataset_name == "filtered_hybrid":
            if not data_args.raw_anno_path:
                raise ValueError(
                    "raw_anno_path is required for filtered_hybrid dataset."
                )
            if not Path(data_args.raw_anno_path).exists():
                raise FileNotFoundError(
                    f"raw_anno_path does not exist: {data_args.raw_anno_path}"
                )
            raw_annos = _load_filtered_annos(data_args.raw_anno_path)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        annos = []
        for anno in raw_annos:
            num_words = len(anno["query"].split(" "))
            if data_args.min_num_words >= 0 and num_words < data_args.min_num_words:
                continue
            if data_args.max_num_words >= 0 and num_words > data_args.max_num_words:
                continue
            if (
                data_args.min_video_len >= 0
                and anno.get("duration", float("inf")) < data_args.min_video_len
            ):
                continue
            if (
                data_args.max_video_len >= 0
                and anno.get("duration", 0) > data_args.max_video_len
            ):
                continue
            duration = anno.get("duration")
            spans = _normalize_spans(anno["span"])
            if duration and not any(0 <= s <= e <= duration for s, e in spans):
                continue
            anno = dict(anno)
            anno["span"] = spans
            annos.append(anno)

        if filter_args is not None:
            annos = self._filter_annos(annos, filter_args)

        self.annos = annos
        self.raw_length = len(raw_annos)

    def _filter_annos(self, annos, filter_args):
        unique_videos = filter_args.get("unique_videos", False)
        if unique_videos:
            seen = set()
            uniq = []
            for anno in annos:
                vpath = anno["video_path"]
                if vpath in seen:
                    continue
                seen.add(vpath)
                uniq.append(anno)
            annos = uniq

        filter_ratio = filter_args.get("filter_ratio")
        filter_target_size = filter_args.get("filter_target_size")
        if filter_ratio is None and filter_target_size is None:
            return annos

        gaussian_filter_mean = getattr(self.data_args, "gaussian_filter_mean", None)
        gaussian_filter_std = getattr(self.data_args, "gaussian_filter_std", None)
        if (gaussian_filter_mean is None) != (gaussian_filter_std is None):
            raise ValueError(
                "gaussian_filter_mean and gaussian_filter_std should be provided together."
            )
        if gaussian_filter_mean is not None and not annos:
            return annos
        if gaussian_filter_mean is not None and "iou" not in annos[0]:
            raise ValueError("Gaussian filtering requires 'iou' in annotations.")

        seed = getattr(self.training_args, "seed", 42)
        rng = np.random.default_rng(seed)
        py_rng = random.Random(seed)

        buckets = {duration_range: [] for duration_range in filter_args["filter_range"]}
        kept_indices = []
        for idx, anno in enumerate(annos):
            matched = False
            for duration_range in buckets:
                min_duration, max_duration = duration_range
                if min_duration <= anno["duration"] <= max_duration:
                    buckets[duration_range].append(idx)
                    matched = True
                    break
            if not matched:
                kept_indices.append(idx)

        for i, (duration_range, indices) in enumerate(buckets.items()):
            if len(indices) == 0:
                continue
            num_to_select = (
                int(len(indices) * filter_ratio[i])
                if filter_ratio is not None
                else int(filter_target_size[i])
            )
            num_to_select = min(num_to_select, len(indices))

            if gaussian_filter_mean is not None:
                iou_list = np.array(
                    [annos[idx]["iou"] for idx in indices], dtype=np.float64
                )
                weights = np.exp(
                    -0.5
                    * ((iou_list - gaussian_filter_mean) / gaussian_filter_std) ** 2
                )
                if getattr(self.data_args, "fixed_gaussian_sampling", False):
                    num_bins = 20
                    counts, bin_edges = np.histogram(
                        iou_list, bins=num_bins, range=(0, 1)
                    )
                    bin_indices = np.digitize(iou_list, bins=bin_edges)
                    bin_indices = np.clip(bin_indices, 1, num_bins) - 1
                    inverse_density = 1.0 / (counts + 1e-6)
                    weights *= inverse_density[bin_indices]
                weights = weights / weights.sum()
                selected_indices = rng.choice(
                    indices, size=num_to_select, replace=False, p=weights
                ).tolist()
            else:
                selected_indices = py_rng.sample(indices, num_to_select)
            kept_indices.extend(selected_indices)

        return [annos[i] for i in range(len(annos)) if i in kept_indices]

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        if self.training_mode == "sft":
            return self._getitem_sft(idx)
        if self.training_mode == "grpo":
            return self._getitem_grpo(idx)
        raise ValueError(f"Unsupported training_mode: {self.training_mode}")

    def _getitem_sft(self, idx):
        anno = copy.deepcopy(self.annos[idx])
        spans = _normalize_spans(anno["span"])

        messages = [
            {
                "role": "user",
                "content": [
                    _build_video_content(anno, self.data_args),
                    {"type": "text", "text": GROUNDING_PROMPT.format(anno["query"])},
                ],
            }
        ]

        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None

        response = _format_response(spans)
        messages.append({"role": "assistant", "content": response})

        text = self.processor.apply_chat_template(messages, tokenize=False)
        text = [text.strip()]
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        )
        inputs["input_ids"] = inputs["input_ids"][0]
        inputs["labels"] = preprocess(
            inputs["input_ids"],
            text[0],
            self.processor.tokenizer,
            self.model_args.conv_type,
        )
        return inputs

    def _getitem_grpo(self, idx):
        anno = copy.deepcopy(self.annos[idx])

        messages = [
            {
                "role": "user",
                "content": [
                    _build_video_content(
                        anno, self.data_args, include_video_range=True
                    ),
                    {"type": "text", "text": GROUNDING_PROMPT.format(anno["query"])},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        text = [text]

        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        )
        inputs["input_ids"] = inputs["input_ids"][0]
        inputs["prompt"] = messages
        inputs["prompt_text"] = text[0]
        inputs["anno"] = anno
        return inputs

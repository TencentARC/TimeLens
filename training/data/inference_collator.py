import copy

from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

from timelens.dataset.timelens_data import parse_query

GROUNDING_PROMPT = (
    "Please find the visual event described by the sentence '{}', determining its starting and ending times. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'."
)

# prompt for TimeLens-7B (Qwen2.5-VL) with interleaved textual timestamps
GROUNDING_PROMPT_TEXT_TIMESTAMP = (
    "You are given a video with multiple frames. "
    "The numbers before each video frame indicate its sampling timestamp (in seconds). "
) + GROUNDING_PROMPT


def _is_timelens_7b_model(model_name: str) -> bool:
    return bool(model_name) and "timelens-7b" in model_name.lower()


def _is_qwen2_model(model_name: str) -> bool:
    if not model_name:
        return False
    m = model_name.lower()
    return "qwen2" in m or "qwen2.5-vl" in m or "qwen2.5_vl" in m


def collate_fn(batch, processor, model_name="qwen3-vl"):
    messages = [item["messages"] for item in batch]
    annos = [item["anno"] for item in batch]
    texts = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if _is_timelens_7b_model(model_name):
        # TimeLens-7B: interleave timestamp with textual frame timestamps
        images, videos = process_vision_info(messages, return_video_metadata=True)
        if videos is None or len(videos) == 0:
            raise ValueError(
                "Empty videos for TimeLens-7B strict path. "
                "Please ensure TimeLens-7B processor/config and qwen_vl_utils are aligned."
            )
        inputs = processor(
            text=texts,
            images=images,
            videos=videos,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            do_resize=False,
        )
    elif _is_qwen2_model(model_name):
        # Qwen2.5-VL base model path should follow Qwen2 processing path.
        images, videos, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
        )
        inputs = processor(
            text=texts,
            images=images,
            videos=videos,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        )
    else:
        # Qwen3-VL / TimeLens-8B
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
        inputs = processor(
            text=texts,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        )
    return {"inputs": inputs, "annos": annos}


class GroundingDatasetInference(Dataset):
    def __init__(self, annos, args):
        super().__init__()
        self.annos = annos
        self.args = args
        model_path = (
            getattr(args, "format_model_path", None)
            or getattr(args, "processor_path", None)
            or getattr(args, "model_path", "")
            or ""
        )
        self._is_timelens_7b = _is_timelens_7b_model(model_path)
        self._is_qwen2 = _is_qwen2_model(model_path)
        self._prompt = (
            GROUNDING_PROMPT_TEXT_TIMESTAMP
            if self._is_timelens_7b
            else GROUNDING_PROMPT
        )
        self._pixel_scale = (
            28 * 28 if (self._is_timelens_7b or self._is_qwen2) else 32 * 32
        )

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        anno = copy.deepcopy(self.annos[index])
        video_cfg = {
            "type": "video",
            "video": anno["video_path"],
            "min_pixels": int(self.args.min_tokens * self._pixel_scale),
            "total_pixels": int(self.args.total_tokens * self._pixel_scale),
            "fps": float(self.args.fps),
        }
        if getattr(self.args, "fps_max_frames", None) is not None:
            video_cfg["max_frames"] = int(self.args.fps_max_frames)
        message = {
            "role": "user",
            "content": [
                video_cfg,
                {
                    "type": "text",
                    "text": self._prompt.format(parse_query(anno["query"])),
                },
            ],
        }
        return {"messages": [message], "anno": anno}

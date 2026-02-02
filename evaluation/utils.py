# Copyright (c) 2025 Jun Zhang. Licensed under the BSD-3-Clause License.

import copy

from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

GROUNDER_PROMPT = (
    "Please find the visual event described by the sentence '{}', determining its starting and ending times. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'."
)

# prompt for TimeLens-7B (based on Qwen2.5-VL) with interleaved textual timestamps
GROUNDER_PROMPT_TEXT_TIMESTAMP = (
    "You are given a video with multiple frames. "
    "The numbers before each video frame indicate its sampling timestamp (in seconds). "
) + GROUNDER_PROMPT


class GroundingDataset(Dataset):
    def __init__(self, annos, processor, args):
        super().__init__()
        self.annos = annos
        self.processor = processor
        self.args = args
        if "timelens-7b" in args.model_path.lower():
            # prompt for TimeLens-7B (based on Qwen2.5-VL) with interleaved textual timestamps
            self.prompt = GROUNDER_PROMPT_TEXT_TIMESTAMP
        else:
            self.prompt = GROUNDER_PROMPT

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        anno = copy.deepcopy(self.annos[index])

        video_path = anno["video_path"]
        query = anno["query"]

        if "qwen3" in self.args.model_path.lower() or "timelens-8b" in self.args.model_path.lower():
            # for TimeLens-8B(based on Qwen3-VL) and Qwen3-VL models
            downsample_rate = 32
        elif "qwen2" in self.args.model_path.lower() or "timelens-7b" in self.args.model_path.lower():
            # for TimeLens-7B (based on Qwen2.5-VL) and Qwen2.5-VL models
            downsample_rate = 28
        else:
            raise NotImplementedError(
                f"Model {self.args.model_path} not supported yet."
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "min_pixels": self.args.min_tokens * downsample_rate * downsample_rate,
                        "total_pixels": self.args.total_tokens * downsample_rate * downsample_rate,
                        "fps": self.args.fps,
                    },
                    {"type": "text", "text": self.prompt.format(query)},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if "timelens-7b" in self.args.model_path.lower():
            # for TimeLens-7B (based on Qwen2.5-VL) with interleaved textual timestamps
            images, videos = process_vision_info(messages, return_video_metadata=True)
            inputs = self.processor(
                text=[text],
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
            )
        elif (
            "qwen3" in self.args.model_path.lower()
            or "timelens-8b" in self.args.model_path.lower()
        ):
            # for TimeLens-8B(based on Qwen3-VL) and Qwen3-VL models
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
            inputs = self.processor(
                text=[text],
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
        elif "qwen2" in self.args.model_path.lower():
            # for Qwen2.5-VL model
            images, videos, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )
            inputs = self.processor(
                text=[text],
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
        else:
            raise NotImplementedError(
                f"Model {self.args.model_path} not supported yet."
            )

        return {"inputs": inputs, "anno": anno}

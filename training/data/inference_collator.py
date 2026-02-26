import copy

from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

from timelens.dataset.timelens_data import parse_query


GROUNDING_PROMPT = (
    "Please find the visual event described by the sentence '{}', determining its starting and ending times. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'."
)


def collate_fn(batch, processor, model_name="qwen3-vl"):
    messages = [item["messages"] for item in batch]
    annos = [item["anno"] for item in batch]
    texts = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

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

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        anno = copy.deepcopy(self.annos[index])
        video_cfg = {
            "type": "video",
            "video": anno["video_path"],
            "min_pixels": int(self.args.min_tokens * 32 * 32),
            "total_pixels": int(self.args.total_tokens * 32 * 32),
            "fps": float(self.args.fps),
        }
        if getattr(self.args, "fps_max_frames", None) is not None:
            video_cfg["max_frames"] = int(self.args.fps_max_frames)
        message = {
            "role": "user",
            "content": [
                video_cfg,
                {"type": "text", "text": GROUNDING_PROMPT.format(parse_query(anno["query"]))},
            ],
        }
        return {"messages": [message], "anno": anno}

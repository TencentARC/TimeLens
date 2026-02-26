import argparse
import json
from functools import partial
from pathlib import Path
import sys

import nncore
import torch
from nncore.engine import set_random_seed
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from timelens.dataset.timelens_data import TimeLens100KDataset
from training.data import GroundingDatasetInference, collate_fn
from training.utils.parser import extract_answer, extract_time, iou

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gemini_refined_data")
    parser.add_argument("--pred_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_tokens", type=int, default=64)
    parser.add_argument("--total_tokens", type=int, default=14336)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--fps_max_frames", type=int, default=None)
    return parser.parse_args()


def load_annos(dataset_names: str, split: str):
    if split != "train":
        raise ValueError("Only train split is supported in filtering stage.")
    annos = []
    for dataset in dataset_names.split(","):
        dataset = dataset.strip()
        if dataset == "gemini_refined_data":
            annos.extend(
                [
                    anno
                    for anno in TimeLens100KDataset.load_annos(split="train")
                    if not _is_audio_related_query(anno["query"])
                ]
            )
        elif dataset == "timelens-100k":
            annos.extend(TimeLens100KDataset.load_annos(split="train"))
        else:
            raise ValueError(f"Unsupported dataset for filtering: {dataset}")
    return annos


def dump_jsonl(path: str, rows):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    args = parse_args()
    args.seed = set_random_seed(args.seed)
    print(f"Setting random seed to {args.seed}")

    pred_path = f"{args.pred_path}_{args.index}.jsonl"
    print(
        f"Dataset: {args.dataset}({args.split}) | Chunk: {args.chunk} | "
        f"Index: {args.index} | Output Path: {pred_path}"
    )
    if args.device != "auto":
        raise ValueError('Only device="auto" is supported.')

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=args.device,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        padding_side="left",
        do_resize=False,
        trust_remote_code=True,
    )

    annos = load_annos(args.dataset, args.split)
    annos.sort(key=lambda x: x["duration"], reverse=True)
    annos = annos[args.index :: args.chunk]

    dataset = GroundingDatasetInference(annos, args)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=10,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=partial(collate_fn, processor=processor, model_name=args.model_path),
    )

    dumps = []
    for data in nncore.ProgressBar(data_loader):
        inputs = data["inputs"].to("cuda", non_blocking=True)
        annos = data["annos"]

        output_ids = model.generate(
            **inputs,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            max_new_tokens=512,
            use_cache=True,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        answers = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for anno, answer in zip(annos, answers):
            duration = anno["duration"]
            full_answer = answer
            parsed_answer = extract_answer(answer)
            timestamps = extract_time(parsed_answer)
            if len(timestamps) == 0:
                timestamps = [[duration + 10, duration + 20]]
            pred = timestamps

            gt_span = anno["span"]
            if isinstance(gt_span[0], list):
                gt_span = gt_span[0]
            cur_iou = iou(pred[0], gt_span)
            anno.update({"pred": pred, "answer": full_answer, "iou": cur_iou})
            dumps.append(anno)

    dump_jsonl(pred_path, dumps)

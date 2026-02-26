# Copyright (c) 2025 Jun Zhang. Licensed under the BSD-3-Clause License.

import json
import os
import re


def parse_query(query):
    "Clean and normalize a text query by removing unnecessary whitespace and trailing periods"
    return re.sub(r"\s+", " ", query).strip().strip(".").strip()


class ActivitynetTimeLensDataset:
    ANNO_PATH_TEST = "data/TimeLens-Bench/activitynet-timelens.json"
    VIDEO_ROOT = "data/TimeLens-Bench/videos/activitynet"
    DATASET_SOURCE = "ActivityNet-TimeLens"

    @classmethod
    def load_annos(cls, split="test"):
        assert split == "test", f"Invalid split: {split}"

        with open(cls.ANNO_PATH_TEST, "r") as f:
            raw_annos = json.load(f)

        annos = []
        for vid, raw_anno in raw_annos.items():
            video_path = os.path.join(cls.VIDEO_ROOT, vid + ".mp4")
            # if not os.path.exists(video_path):
            # raise FileNotFoundError(f"Video path does not exist: {video_path}")
            for span, query in zip(raw_anno["spans"], raw_anno["queries"]):
                anno = dict(
                    source=cls.DATASET_SOURCE,
                    data_type="grounding",
                    video_path=video_path,
                    duration=raw_anno["duration"],
                    query=parse_query(query),
                    span=[span],
                )

                annos.append(anno)

        return annos


class QVHighlightsTimeLensDataset(ActivitynetTimeLensDataset):
    ANNO_PATH_TEST = "data/TimeLens-Bench/qvhighlights-timelens.json"
    VIDEO_ROOT = "data/TimeLens-Bench/videos/qvhighlights"
    DATASET_SOURCE = "QVHighlights-TimeLens"


class CharadesTimeLensDataset(ActivitynetTimeLensDataset):
    ANNO_PATH_TEST = "data/TimeLens-Bench/charades-timelens.json"
    VIDEO_ROOT = "data/TimeLens-Bench/videos/charades"
    DATASET_SOURCE = "Charades-TimeLens"


class TimeLens100KDataset:
    ANNO_PATH_TRAIN = "data/TimeLens-100K/timelens-100k.jsonl"
    VIDEO_ROOT = "data/TimeLens-100K/videos"

    @classmethod
    def load_annos(self, split="train"):
        assert split == "train", f"Invalid split: {split}"
        raw_anno = []
        with open(self.ANNO_PATH_TRAIN, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                raw_anno.append(data)

        annos = []
        for raw_anno in raw_anno:
            video_path = os.path.join(self.VIDEO_ROOT, raw_anno["video_path"])
            # if not os.path.exists(video_path):
            # raise FileNotFoundError(f"Video path does not exist: {video_path}")
            for event in raw_anno["events"]:
                query = parse_query(event["query"])
                span = event["span"]
                anno = dict(
                    source=raw_anno["source"],
                    data_type="grounding",
                    video_path=video_path,
                    duration=raw_anno["duration"],
                    query=query,
                    span=span,
                )
                annos.append(anno)

        return annos


DATASET_DICT = {
    "activitynet-timelens": ActivitynetTimeLensDataset,
    "qvhighlights-timelens": QVHighlightsTimeLensDataset,
    "charades-timelens": CharadesTimeLensDataset,
    "timelens-100k": TimeLens100KDataset,
}

if __name__ == "__main__":
    # Example usage
    DATASET_NAME = "timelens-100k"
    annos = DATASET_DICT[DATASET_NAME].load_annos()
    print(f"Loaded {len(annos)} annotations from {DATASET_NAME}")

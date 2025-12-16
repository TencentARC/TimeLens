# Copyright (c) 2025 Jun Zhang. Licensed under the BSD-3-Clause License.

import json
import re


def read_json(path):
    with open(path, "r") as fin:
        datas = json.load(fin)
    return datas


def write_json(path, data):
    with open(path, "w") as fout:
        json.dump(data, fout)
    print("The format file has been saved at:{}".format(path))
    return


def iou(a, b):
    max0 = max((a[0]), (b[0]))
    min0 = min((a[0]), (b[0]))
    max1 = max((a[1]), (b[1]))
    min1 = min((a[1]), (b[1]))
    return max(min1 - max0, 0) / (max1 - min0)


def extract_time(paragraph):
    paragraph = paragraph.lower()

    timestamps = []

    # Check for HH:MM:SS and MM:SS formats FIRST (before checking individual numbers)
    # Also handle formats with optional milliseconds like MM:SS.xxx and HH:MM:SS.xx
    time_regex = re.compile(
        r"\b(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?)\b"
    )
    time_matches = re.findall(time_regex, paragraph)
    time_matches = time_matches[: len(time_matches) // 2 * 2]

    if time_matches:
        # convert to seconds
        time_matches_converted = []
        for t in time_matches:
            parts = t.split(":")
            if len(parts) == 3:  # HH:MM:SS.xx format
                h, m = map(int, parts[:2])
                s = float(parts[2])
                time_in_sec = h * 3600 + m * 60 + s
            elif len(parts) == 2:  # MM:SS.xxx format
                m = int(parts[0])
                s = float(parts[1])
                time_in_sec = m * 60 + s
            time_matches_converted.append(float(time_in_sec))
        timestamps = [
            (time_matches_converted[i], time_matches_converted[i + 1])
            for i in range(0, len(time_matches_converted), 2)
        ]

    # Check for The given query happens in m - n (seconds)
    if len(timestamps) == 0:
        patterns = [
            r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",  # 18.5 - 23.0
            r"(\d+\.?\d*)\s+to\s+(\d+\.?\d*)",  # 18.5 to 23.0
        ]
        for time_pattern in patterns:
            time_matches = re.findall(time_pattern, paragraph)
            if time_matches:
                timestamps = [(float(start), float(end)) for start, end in time_matches]
                break

    # Check for other formats e.g.:
    # 1. Starting time: 0.8 seconds. Ending time: 1.1 seconds
    # 2. The start time for this event is 0 seconds, and the end time is 12 seconds.
    if len(timestamps) == 0:
        time_regex = re.compile(r"\b(\d+\.\d+|\d+)\b")  # time formats (e.g., 18, 18.5)
        time_matches = re.findall(time_regex, paragraph)
        time_matches = time_matches[: len(time_matches) // 2 * 2]
        timestamps = [
            (float(time_matches[i]), float(time_matches[i + 1]))
            for i in range(0, len(time_matches), 2)
        ]

    timestamps = [(start, end) for start, end in timestamps]
    return timestamps

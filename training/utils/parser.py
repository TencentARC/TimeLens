# Parser utilities for temporal grounding (extract_time, extract_answer, iou)
# Adapted from VideoMind videomind.utils.parser

import re


def extract_answer(content):
    """
    Extract the answer within <answer> </answer> tags.
    If the format is not correct, return content as is.
    """
    format_pattern = r"<think>.*?</think>\s*<answer>(.*?)</answer>"
    match = re.match(format_pattern, content, re.DOTALL)
    if match:
        return match.group(1)
    elif any(tag in content for tag in ('<think>', '</think>', '<answer>', '</answer>')):
        return content
    return content


def iou(A, B):
    """Temporal intersection over union between two spans [start, end]."""
    max0 = max(A[0], B[0])
    min0 = min(A[0], B[0])
    max1 = max(A[1], B[1])
    min1 = min(A[1], B[1])
    denom = max1 - min0
    if denom <= 0:
        return 0.0
    return max(min1 - max0, 0) / denom


def extract_time(paragraph):
    """Extract timestamps from text. Returns list of (start, end) tuples in seconds."""
    paragraph = paragraph.lower().replace("to", "-")
    timestamps = []

    # HH:MM:SS and MM:SS formats
    time_regex = re.compile(r"\b(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?)\b")
    time_matches = re.findall(time_regex, paragraph)
    time_matches = time_matches[: len(time_matches) // 2 * 2]

    if time_matches:
        time_matches_converted = []
        for t in time_matches:
            parts = t.split(":")
            if len(parts) == 3:
                h, m = map(int, parts[:2])
                s = float(parts[2])
                time_in_sec = h * 3600 + m * 60 + s
            elif len(parts) == 2:
                m = int(parts[0])
                s = float(parts[1])
                time_in_sec = m * 60 + s
            time_matches_converted.append(float(time_in_sec))
        timestamps = [
            (time_matches_converted[i], time_matches_converted[i + 1])
            for i in range(0, len(time_matches_converted), 2)
        ]

    if len(timestamps) == 0:
        patterns = [
            r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",
            r"(\d+\.?\d*)\s+to\s+(\d+\.?\d*)"
        ]
        for time_pattern in patterns:
            time_matches = re.findall(time_pattern, paragraph)
            if time_matches:
                timestamps = [(float(start), float(end)) for start, end in time_matches]
                break

    if len(timestamps) == 0:
        time_regex = re.compile(r"\b(\d+\.\d+|\d+)\b")
        time_matches = re.findall(time_regex, paragraph)
        time_matches = time_matches[: len(time_matches) // 2 * 2]
        timestamps = [
            (float(time_matches[i]), float(time_matches[i + 1]))
            for i in range(0, len(time_matches), 2)
        ]

    return [(start, end) for start, end in timestamps]

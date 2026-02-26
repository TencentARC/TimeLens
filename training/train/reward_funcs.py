import re

from training.utils.parser import extract_time, extract_answer, iou


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has <think>...</think> <answer>...</answer> format."""
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content) for content in completion_contents]

    for i, match in enumerate(matches):
        if not match:
            print(f"Completion {i} does not match the required format: {completion_contents[i]}")

    return [1.0 if match else 0.0 for match in matches]


def tiou_reward(prompts, completions, completion_ids, anno, prompt_text, **kwargs):
    """Reward function that returns temporal IoU between predicted and ground truth spans."""
    pattern = r'<\|(video_pad|image_pad|vision_start|vision_end)\|>'
    prompt_text = [re.sub(pattern, '', text) for text in prompt_text]

    completions = [completion[0]["content"] for completion in completions]
    answers = [extract_answer(completion) for completion in completions]
    timestamps_list = [extract_time(answer) for answer in answers]

    rewards = []
    for i, timestamps in enumerate(timestamps_list):
        gt = anno[i]["span"]
        if isinstance(gt[0], list):
            gt = gt[0]

        pred = answers[i]

        if len(timestamps) == 0:
            print(f"Timestamp extraction failed: pred={pred}, IoU will be 0")
            rewards.append(0)
        elif timestamps[0][0] >= timestamps[0][1]:
            print(f"Warning: Invalid timestamp in prediction '{pred}', IoU will be 0")
            rewards.append(0)
        else:
            if len(timestamps) > 1:
                print(f"Warning: Multiple timestamps for '{pred}', using first: {timestamps[0]}")
            rewards.append(iou(gt, timestamps[0]))
            print(f"prompt: {prompt_text[i]}, completion: {completions[i]}, answer: {pred}, gt: {gt}, tIoU: {rewards[i]}")

    return rewards


REWARD_FUNCS_DICT = {
    "tiou": tiou_reward,
    "format": format_reward,
}


def load_reward_funcs(reward_func_names):
    return [
        REWARD_FUNCS_DICT[func_name.strip()]
        for func_name in reward_func_names.split(",")
    ]

<p align="center">
  <h1 align="center">TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs <br/>🏆 CVPR 2026</h1>
</p>

<p align="center">
  <a href="https://home.j-zh.top/">Jun Zhang</a>, <a href="http://ttengwang.com/">Teng Wang</a>, <a href="https://geyuying.github.io/">Yuying Ge</a>, <a href="https://geyixiao.com/">Yixiao Ge</a>, <a href="https://scholar.google.com/citations?user=evR3uR0AAAAJ">Xinhao Li</a>, <a href="https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en">Ying Shan</a>, <a href="https://scholar.google.com/citations?user=HEuN8PcAAAAJ&hl=en">Limin Wang</a>
</p>

<p align="center">
    &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2512.14698"><b>Paper</b></a>&nbsp&nbsp | &nbsp&nbsp🏠 <a href="https://timelens-arc-lab.github.io/"><b>Project Page</b></a>&nbsp&nbsp | 🤗 <a href="https://huggingface.co/collections/TencentARC/timelens"><b>Model & Data</b></a>&nbsp&nbsp | 🏆 <a href="https://timelens-arc-lab.github.io/#leaderboard"><b>TimeLens-Bench Leaderboard</b></a>&nbsp&nbsp
</p>

## 📰 News
- **2026.02.26**: 🚀 We now support training **TimeLens-8B** based on **Qwen3-8B-VL**.
- **2026.02.22**: 🎉 TimeLens has been accepted to **CVPR 2026**.

## 🔎 Overview
TimeLens rethinks video temporal grounding (VTG) with MLLMs along two axes:
- **Data Quality**. We expose critical quality issues in existing VTG benchmarks and propose quality-assured datasets for both training and evaluation.
- **Algorithmic Design**. Building upon reliable data, we explore effective timestamp encoding strategies and training recipes, achieving state-of-the-art performance among open-source models.

## 📚 Quick Navigation
In this repository, we release:
- 🤖 **TimeLens Models**: State-of-the-art open-source models for video temporal grounding.
  - [Model Usage](#-using-timelens-models)
- 📊 **TimeLens-Bench**: a comprehensive, high-quality evaluation benchmark for video temporal grounding.
  - 🏆 [Leaderboard](https://timelens-arc-lab.github.io/#leaderboard)
  - [Evaluation Guide](#-evaluation-on-timelens-bench)
- 🏋️ **TimeLens-100K**: a large-scale, diverse, high-quality training dataset for video temporal grounding, annotated with Gemini-2.5-Pro.
  - [Training Guide](#️-training-on-timelens-100k)

## 📦 Installation

Clone this repository and navigate to the folder
```bash
git clone https://github.com/TencentARC/TimeLens.git
cd TimeLens
```

Create a Conda environment and install the required packages
```bash
conda create -n timelens python=3.11 -y
conda activate timelens

# install dependencies for inference
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu124 # We use CUDA Version 12.4

# Optional: install extra dependencies for training
pip install -r requirements_train.txt

# Install flash-attn (required for BOTH training and inference!)
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
```

## 🤖 Using TimeLens Models
TimeLens models are a family of MLLMs with SotA video temporal grounding performance. They are built upon the Qwen2.5-VL and Qwen3-VL baselines through training on our high-quality [TimeLens-100K](#️-training-on-timelens-100k) dataset, leveraging our carefully crafted RLVR (reinforcement learning with verifiable rewards) recipe and improved timestamp encoding strategy.

### 🚀 Quick Start
All models are available on Hugging Face and support out-of-the-box inference using the 🤗Transformers library. For detailed usage instructions and code examples, please refer to the specific model's Hugging Face page linked below.

### 🏆 Model Zoo & Performance
The following table lists our models with their Hugging Face links and grounding performance:
<table>
  <thead>
    <tr>
      <th rowspan="2" align="center">Model <br>(with 🤗HuggingFace Link)</th>
      <th colspan="4" align="center">Charades-TimeLens</th>
      <th colspan="4" align="center">ActivityNet-TimeLens</th>
      <th colspan="4" align="center">QVHighlights-TimeLens</th>
    </tr>
    <tr>
      <th align="center">R1<br>@0.3</th>
      <th align="center">R1<br>@0.5</th>
      <th align="center">R1<br>@0.7</th>
      <th align="center">mIoU</th>
      <th align="center">R1<br>@0.3</th>
      <th align="center">R1<br>@0.5</th>
      <th align="center">R1<br>@0.7</th>
      <th align="center">mIoU</th>
      <th align="center">R1<br>@0.3</th>
      <th align="center">R1<br>@0.5</th>
      <th align="center">R1<br>@0.7</th>
      <th align="center">mIoU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct">Qwen2.5-VL-7B-Instruct</a></td>
      <td align="center">59.7</td>
      <td align="center">37.8</td>
      <td align="center">16.6</td>
      <td align="center">39.3</td>
      <td align="center">44.1</td>
      <td align="center">31.0</td>
      <td align="center">16.1</td>
      <td align="center">31.4</td>
      <td align="center">41.5</td>
      <td align="center">27.8</td>
      <td align="center">15.2</td>
      <td align="center">31.6</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/TencentARC/TimeLens-7B"><b>TimeLens-7B</b>🚀</a></td>
      <td align="center"><b>70.5</b></td>
      <td align="center"><b>55.6</b></td>
      <td align="center"><b>28.4</b></td>
      <td align="center"><b>48.8</b></td>
      <td align="center"><b>62.8</b></td>
      <td align="center"><b>51.0</b></td>
      <td align="center"><b>32.6</b></td>
      <td align="center"><b>46.2</b></td>
      <td align="center"><b>74.1</b></td>
      <td align="center"><b>62.7</b></td>
      <td align="center"><b>43.1</b></td>
      <td align="center"><b>56.0</b></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct">Qwen3-VL-8B-Instruct</a></td>
      <td align="center">69.2</td>
      <td align="center">53.4</td>
      <td align="center">27.5</td>
      <td align="center">48.3</td>
      <td align="center">62.1</td>
      <td align="center">51.2</td>
      <td align="center">34.4</td>
      <td align="center">46.8</td>
      <td align="center">74.2</td>
      <td align="center">64.6</td>
      <td align="center">49.3</td>
      <td align="center">59.4</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/TencentARC/TimeLens-8B"><b>TimeLens-8B</b>🚀</a></td>
      <td align="center"><b>76.6</b></td>
      <td align="center"><b>63.0</b></td>
      <td align="center"><b>35.2</b></td>
      <td align="center"><b>55.2</b></td>
      <td align="center"><b>68.9</b></td>
      <td align="center"><b>58.4</b></td>
      <td align="center"><b>40.6</b></td>
      <td align="center"><b>53.2</b></td>
      <td align="center"><b>80.2</b></td>
      <td align="center"><b>71.6</b></td>
      <td align="center"><b>55.5</b></td>
      <td align="center"><b>65.5</b></td>
    </tr>
  </tbody>
</table>

> TimeLens-7B is fine-tuned from Qwen2.5-VL-7B-Instruct, and TimeLens-8B is fine-tuned from Qwen3-VL-8B-Instruct.

> [!NOTE]
> For detailed comparison with other models, please refer to the 🏆 [Leaderboard](https://timelens-arc-lab.github.io/#leaderboard).


## 📊 Evaluation on TimeLens-Bench

### Download TimeLens-Bench

Download the [TimeLens-Bench dataset](https://huggingface.co/datasets/TencentARC/TimeLens-Bench) from Hugging Face and place it in the `data/TimeLens-Bench` directory:
```bash
hf download TencentARC/TimeLens-Bench \
  --repo-type=dataset \
  --local-dir data/TimeLens-Bench
```

Extract the compressed videos:
```bash
mkdir -p data/TimeLens-Bench/videos
find data/TimeLens-Bench/video_shards -name "*.tar.gz" | \
  xargs -P 4 -I {} tar -xzf {} -C data/TimeLens-Bench/videos # Parallel extraction with 4 processes
```

The folder structure should look like this:
```
TimeLens/
└── data/
    └── TimeLens-Bench/
        ├── activitynet-timelens.json
        ├── charades-timelens.json
        ├── qvhighlights-timelens.json
        ├── videos/              # extracted videos
        │   ├── activitynet/
        │   ├── charades/
        │   └── qvhighlights/
        └── video_shards/        # compressed videos (can be deleted after extraction)
```

### Evaluate with Our Codebase (TimeLens / Qwen-VL Models)

Our codebase supports evaluation of the following models:

| Model | Supported |
|:----------:|:---------:|
| [TimeLens-7B](https://huggingface.co/TencentARC/TimeLens-7B) | ✅ |
| [TimeLens-8B](https://huggingface.co/TencentARC/TimeLens-8B) | ✅ |
| [Qwen2.5-VL](https://huggingface.co/collections/Qwen/qwen25-vl) | ✅ |
| [Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl) | ✅ |

The evaluation script is [`scripts/eval_timelens_bench.sh`](./scripts/eval_timelens_bench.sh). You can set the following environment variables:
- **`model_path`**: Path or HuggingFace ID of the model to evaluate. Default: `TencentARC/TimeLens-8B`
- **`datasets`**: Comma-separated list of datasets to evaluate. Default: `charades-timelens,activitynet-timelens,qvhighlights-timelens`
- **`CUDA_VISIBLE_DEVICES`**: GPU indices to use (e.g., `0,1,2,3`). Default: Auto-detect all available GPUs
- **`pred_path`**: Directory to save results. Default: `./logs`
- **`min_tokens`**: Minimum tokens for video encoding. Default: `64`
- **`total_tokens`**: Total tokens for video encoding. Default: `14336`
- **`FPS`**: Frames per second for video sampling. Default: `2`

**Example 1**: Evaluate TimeLens-8B (default settings)
```bash
model_path="TencentARC/TimeLens-8B" bash scripts/eval_timelens_bench.sh
```

**Example 2**: Evaluate TimeLens-7B on specific datasets with specific GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1 \
datasets="activitynet-timelens,qvhighlights-timelens" \
model_path="TencentARC/TimeLens-7B" \
bash scripts/eval_timelens_bench.sh
```

**Example 3**: Evaluate Qwen3-VL with a local model path and a custom path to save results:
```bash
pred_path="./path/to/results" \
model_path="path/to/Qwen3-VL-8B-Instruct" \
bash scripts/eval_timelens_bench.sh
```

> [!TIP]
> **Faster Evaluation with DataLoader** 🚀
>
> Our evaluation script [evaluation/eval_dataloader.py](./evaluation/eval_dataloader.py) supports multi-GPU inference. More importantly, we use [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html) with multiple workers to prefetch and preprocess video data in parallel, while the GPU handles model inference. This significantly accelerates evaluation for long-video tasks like video temporal grounding. Additionally, this approach is more **research-friendly** compared to inference engines like vLLM, as it allows easy customization of the model inference code.
>
> Evaluating TimeLens-7B on ActivityNet-TimeLens with 8× H20 GPUs:
>
> | Method | Time |
> |:------:|:----:|
> | Without DataLoader | 1h23min |
> | With DataLoader | **~34min (~2.4x faster)** |


### Evaluate Your Own Model

To evaluate your own model on TimeLens-Bench, follow these steps:

1. **Load annotations**: Use our provided [timelens_data.py](./timelens/dataset/timelens_data.py) for loading annotations.

2. **Run inference and save results**: Run inference with your model and save results in a JSON or JSONL file with the following format:

   ```python
   {
       f'{video_name}>>>{query}>>>{ground_truth_span}': {
           "timestamps": timestamps,  # the predicted time span from the model
           "answers": answer,  # the full answer text from the model
       }
   }
   ```

   An example of a correctly saved JSON file:

   ```json
   {
       "v_BrgYIg6UXhU.mp4>>>A man wearing a blue jacket approaches a blue car>>>[0.0, 4.0]":
       {
           "timestamps": [[0.0, 5.0]],
           "answers": "The event happens in 0.0 - 5.0 seconds."
       },
       ...
   }
   ```

    In your inference results, you can provide **either** `timestamps` or `answers`. In the next step (Step 3, compute metrics), `evaluation/compute_metrics.py` applies the following logic:
      - If `timestamps` is provided, IoU metrics are computed directly from it.
   - If only `answers` is provided, the script will automatically extract the timestamp pair from the answer text.

3. **Compute metrics**: Use our provided [evaluation/compute_metrics.py](./evaluation/compute_metrics.py) to compute metrics.

  ```bash
  python evaluation/compute_metrics.py -f /path/to/your_result.json
  ```
> For more details on implementing the above steps, you can refer to the [evaluation scripts](#evaluate-with-our-codebase-timelens--qwen-vl-models) of our supported models.


## 🏋️ Training on TimeLens-100K

### Download TimeLens-100K

Download the [TimeLens-100K dataset](https://huggingface.co/datasets/TencentARC/TimeLens-100K) from Hugging Face and place it in the `data/TimeLens-100K` directory:
```bash
hf download TencentARC/TimeLens-100K \
  --repo-type=dataset \
  --local-dir data/TimeLens-100K
```

Extract the compressed videos:
```bash
mkdir -p data/TimeLens-100K/videos
find data/TimeLens-100K/video_shards -name "*.tar.gz" | \
  xargs -P 4 -I {} tar -xzf {} -C data/TimeLens-100K/videos # Parallel extraction with 4 processes
```

The folder structure should look like this:
```
TimeLens/
└── data/
    └── TimeLens-100K/
        ├── README.md
        ├── timelens-100k.jsonl
        ├── videos/              # extracted videos
        │   ├── cosmo_cap/
        │   ├── didemo/
        │   ├── hirest/
        │   ├── internvid_vtime/
        │   └── queryd/
        └── video_shards/        # compressed videos (can be deleted after extraction)
```

### Train with Your Own Codebase

We provide an example script [timelens_data.py](./timelens/dataset/timelens_data.py) for loading TimeLens-100K annotations. You can refer to this code to integrate TimeLens-100K into your own training codebase.

### Use Our Training Code

TimeLens-8B training is released as a 3-stage pipeline (SFT -> filter data -> GRPO):

1. **SFT on TimeLens-100K (30K sampled)**
   We provide a prebuilt SFT checkpoint:
   `https://huggingface.co/JungleGym/TimeLens-Qwen3-VL-8B-SFT`
   You can download it directly:

```bash
hf download JungleGym/TimeLens-Qwen3-VL-8B-SFT \
  --repo-type model \
  --local-dir output/TimeLens-8B/sft/prebuilt
```

   You can also reproduce SFT by yourself:

```bash
bash train_scripts/run_sft_qwen3_8b.sh \
  --model_path "/path/to/Qwen3-VL-8B-Instruct"
```

2. **Run filtering inference on full TimeLens-100K and compute IoU**
   We provide precomputed filtering inference output:
   `https://huggingface.co/datasets/JungleGym/TimeLens-Qwen3-VL-8B-filter-data/blob/main/FPS-2-maxframes-448_TOTALtokens-14336_MINtokens-64---20251209_223300/gemini_refined_data.jsonl`
   You can download it directly:

```bash
hf download JungleGym/TimeLens-Qwen3-VL-8B-filter-data \
  FPS-2-maxframes-448_TOTALtokens-14336_MINtokens-64---20251209_223300/gemini_refined_data.jsonl \
  --repo-type dataset \
  --local-dir output/TimeLens-8B/filter-data/prebuilt
```

   You can also generate it by yourself:

```bash
bash scripts/filter_data/filter_data_qwen3_vl.sh \
  --model_path "output/TimeLens-8B/sft/<your_sft_run_dir>" \
  --dataset gemini_refined_data
```

This stage writes inference output to:
`output/TimeLens-8B/filter-data/.../gemini_refined_data.jsonl`

3. **GRPO training from SFT checkpoint (filtering jsonl as input)**
   If you use prebuilt files downloaded above, use:
   `--model_path output/TimeLens-8B/sft/prebuilt` and
   `--raw_anno_path output/TimeLens-8B/filter-data/prebuilt/FPS-2-maxframes-448_TOTALtokens-14336_MINtokens-64---20251209_223300/gemini_refined_data.jsonl`

   Training + evaluation:

```bash
bash train_scripts/run_grpo_and_eval_qwen3_8b.sh \
  --model_path "output/TimeLens-8B/sft/<your_sft_run_dir>" \
  --raw_anno_path "output/TimeLens-8B/filter-data/<your_filter_run_dir>/gemini_refined_data.jsonl"
```

Training only:

```bash
bash train_scripts/run_grpo_qwen3_8b.sh \
  --model_path "output/TimeLens-8B/sft/<your_sft_run_dir>" \
  --raw_anno_path "output/TimeLens-8B/filter-data/<your_filter_run_dir>/gemini_refined_data.jsonl"
```

Final GRPO checkpoints are saved under:
`output/TimeLens-8B/grpo/...`

#### Evaluate Trained Checkpoints

Use the existing TimeLens-Bench evaluation code directly:

```bash
model_path="output/TimeLens-8B/grpo/<your_grpo_run_dir>" \
bash scripts/eval_timelens_bench.sh
```

## 📝 Citation
If you find our paper, code, model, and data helpful for your research and applications, please consider giving a star ⭐ and citation 📝 :)

```bibtex
@article{zhang2025timelens,
  title={TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs},
  author={Zhang, Jun and Wang, Teng and Ge, Yuying and Ge, Yixiao and Li, Xinhao and Shan, Ying and Wang, Limin},
  journal={arXiv preprint arXiv:2512.14698},
  year={2025}
}
```

## 🙏 Acknowledgement

Our project is built upon the following awesome works:

- [VideoMind](https://github.com/yeliudev/VideoMind)
- [Qwen3-VL and Qwen2.5-VL](https://github.com/QwenLM/Qwen3-VL)
- [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune)
- [TRL - Transformer Reinforcement Learning](https://github.com/huggingface/trl)
- [transformers](https://github.com/huggingface/transformers)

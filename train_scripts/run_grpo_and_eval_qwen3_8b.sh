#!/usr/bin/env bash

set -euo pipefail

cleanup() {
  pkill -P $$ 2>/dev/null || true
  exit 130
}
trap cleanup SIGINT SIGTERM

export PYTHONPATH="./:${PYTHONPATH:-}"

model_path=""
raw_anno_path=""
model_id="qwen3-vl-8b"

datasets="filtered_hybrid"
min_tokens=64
total_tokens=14336
fps=2
fps_max_frames=""
seed=42

global_batch_size=64
batch_per_device=1
num_devices=8
epochs=1
target_size=2500
deepspeed_config="scripts/zero1.json"
report_to="none"

job_name=""
train_output_root="output/TimeLens-8B/rlvr"
eval_pred_root="output/TimeLens-8B/eval-rlvr"
eval_datasets="charades-timelens,activitynet-timelens,qvhighlights-timelens"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) model_path="$2"; shift 2 ;;
    --raw_anno_path) raw_anno_path="$2"; shift 2 ;;
    --datasets) datasets="$2"; shift 2 ;;
    --min_tokens) min_tokens="$2"; shift 2 ;;
    --total_tokens) total_tokens="$2"; shift 2 ;;
    --fps) fps="$2"; shift 2 ;;
    --fps_max_frames) fps_max_frames="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --global_batch_size) global_batch_size="$2"; shift 2 ;;
    --batch_per_device) batch_per_device="$2"; shift 2 ;;
    --num_devices) num_devices="$2"; shift 2 ;;
    --epochs) epochs="$2"; shift 2 ;;
    --target_size) target_size="$2"; shift 2 ;;
    --deepspeed_config) deepspeed_config="$2"; shift 2 ;;
    --report_to) report_to="$2"; shift 2 ;;
    --job_name) job_name="$2"; shift 2 ;;
    --train_output_root) train_output_root="$2"; shift 2 ;;
    --eval_pred_root) eval_pred_root="$2"; shift 2 ;;
    --eval_datasets) eval_datasets="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z "${model_path}" ]]; then
  echo "--model_path is required (qwen3_8b SFT checkpoint path)."
  exit 1
fi

if [[ -z "${raw_anno_path}" ]]; then
  echo "--raw_anno_path is required (filtered jsonl path)."
  exit 1
fi

if [[ ! -d "${model_path}" ]]; then
  echo "model_path does not exist: ${model_path}"
  exit 1
fi

if [[ ! -f "${raw_anno_path}" ]]; then
  echo "raw_anno_path does not exist: ${raw_anno_path}"
  exit 1
fi

if [[ -z "${fps_max_frames}" ]]; then
  fps_max_frames=$((total_tokens / min_tokens * 2))
fi

if [[ -z "${job_name}" ]]; then
  job_name="$(date +%Y%m%d_%H%M%S)"
fi

run_root="${train_output_root}/${job_name}"
mkdir -p "${run_root}"

# Keep hash seed deterministic with training seed.
export PYTHONHASHSEED="${seed}"

echo "========== GRPO Training (${model_id}) =========="
echo "Checkpoint: ${model_path}"
echo "Raw annos  : ${raw_anno_path}"
echo "Run root   : ${run_root}"
echo "Seed       : ${seed}"

bash train_scripts/run_grpo_qwen3_8b.sh \
  --model_path "${model_path}" \
  --raw_anno_path "${raw_anno_path}" \
  --datasets "${datasets}" \
  --min_tokens "${min_tokens}" \
  --total_tokens "${total_tokens}" \
  --fps "${fps}" \
  --fps_max_frames "${fps_max_frames}" \
  --seed "${seed}" \
  --global_batch_size "${global_batch_size}" \
  --batch_per_device "${batch_per_device}" \
  --num_devices "${num_devices}" \
  --epochs "${epochs}" \
  --target_size "${target_size}" \
  --deepspeed_config "${deepspeed_config}" \
  --output_root "${run_root}" \
  --report_to "${report_to}"

trained_model_path="$(ls -dt "${run_root}"/grpo-* 2>/dev/null | awk 'NR==1 {print; exit}' || true)"
if [[ -z "${trained_model_path}" ]]; then
  echo "Cannot locate trained model under ${run_root}"
  exit 1
fi

eval_out="${eval_pred_root}/${job_name}"
mkdir -p "${eval_out}"

echo "========== TimeLens-Bench Eval (${model_id}) =========="
echo "Model path : ${trained_model_path}"
echo "Eval out   : ${eval_out}"
echo "Datasets   : ${eval_datasets}"

model_path="${trained_model_path}" \
datasets="${eval_datasets}" \
min_tokens="${min_tokens}" \
total_tokens="${total_tokens}" \
FPS="${fps}" \
pred_path="${eval_out}" \
bash scripts/eval_timelens_bench.sh

echo ""
echo "Done."
echo "Trained model: ${trained_model_path}"
echo "Eval outputs : ${eval_out}"

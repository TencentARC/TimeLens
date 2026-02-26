#!/usr/bin/env bash

set -euo pipefail

cleanup() {
  pkill -P $$ 2>/dev/null || true
  exit 130
}
trap cleanup SIGINT SIGTERM

export PYTHONPATH="./:${PYTHONPATH:-}"

dataset="gemini_refined_data"
model_path="/path/to/Qwen3-VL-8B-Instruct"
model_id="qwen3-vl-8b"
min_tokens=64
total_tokens=14336
fps=2
fps_max_frames=""
pred_root="output/TimeLens-8B/filter-data"
seed=42

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) dataset="$2"; shift 2 ;;
    --model_path) model_path="$2"; shift 2 ;;
    --model_id) model_id="$2"; shift 2 ;;
    --min_tokens) min_tokens="$2"; shift 2 ;;
    --total_tokens) total_tokens="$2"; shift 2 ;;
    --fps) fps="$2"; shift 2 ;;
    --fps_max_frames) fps_max_frames="$2"; shift 2 ;;
    --pred_root) pred_root="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z "${fps_max_frames}" ]]; then
  fps_max_frames=$((total_tokens / min_tokens * 2))
fi
run_name="FPS-${fps}-maxframes-${fps_max_frames}_TOTALtokens-${total_tokens}_MINtokens-${min_tokens}---$(date +%Y%m%d_%H%M%S)"
pred_path="${pred_root}/${run_name}/${dataset}"
mkdir -p "$(dirname "${pred_path}")"

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(($(nvidia-smi -L | wc -l)-1)))}"
CHUNKS=${#GPULIST[@]}
echo "Using GPUs: ${GPULIST[*]}"
echo "Output path: ${pred_path}"

for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python training/filter/infer_qwen3_vl_tvg_dataloader_filter_data.py \
    --dataset "${dataset}" \
    --split train \
    --pred_path "${pred_path}" \
    --model_path "${model_path}" \
    --chunk "${CHUNKS}" \
    --index "${IDX}" \
    --seed "${seed}" \
    --min_tokens "${min_tokens}" \
    --total_tokens "${total_tokens}" \
    --fps "${fps}" \
    --fps_max_frames "${fps_max_frames}" &
done

wait

cat "${pred_path}"_*.jsonl > "${pred_path}.jsonl"
rm -f "${pred_path}"_*.jsonl
echo "Filtered results saved to: ${pred_path}.jsonl"

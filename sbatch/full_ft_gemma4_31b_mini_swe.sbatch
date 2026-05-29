#!/usr/bin/env bash
#SBATCH --job-name=gemma4-full-mini-swe-modelonly
#SBATCH --partition=batch
#SBATCH --gres=gpu:H200:6
#SBATCH --cpus-per-task=64
#SBATCH --mem=1500G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}
UV=${UV:-uv}
CONFIG=${CONFIG:-${PROJECT_DIR}/configs/train/full_ft_gemma4_31b_mini_swe.yml}
RAW_DIR=${RAW_DIR:?Set RAW_DIR to the raw Vuljector trajectories dataset directory}
DATA_FILE=${DATA_FILE:-${PROJECT_DIR}/data/processed/gemma4_agent_trajectories.jsonl}
WANDB_KEY_FILE=${WANDB_KEY_FILE:?Set WANDB_KEY_FILE to a local file containing your W&B API key}
HF_KEY_FILE=${HF_KEY_FILE:?Set HF_KEY_FILE to a local file containing your Hugging Face token}
INITIAL_MODEL_ID=${INITIAL_MODEL_ID:-google/gemma-4-31B-it}
NPROC=${SLURM_GPUS_ON_NODE:-6}

mkdir -p "${PROJECT_DIR}/slurm_logs" "${PROJECT_DIR}/data/processed" "${PROJECT_DIR}/runs"
cd "${PROJECT_DIR}"

module purge || true
module load cuda/12.8 || true

export HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TRANSFORMERS_NO_TORCHVISION=1
export WANDB_API_KEY="$(tr -d '\n\r ' < "${WANDB_KEY_FILE}")"
export HF_TOKEN="$(tr -d '\n\r ' < "${HF_KEY_FILE}")"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export PROJECT_DIR RAW_DIR DATA_FILE WANDB_KEY_FILE HF_KEY_FILE INITIAL_MODEL_ID

echo "job=${SLURM_JOB_ID} node=$(hostname) date=$(date --iso-8601=seconds)"
nvidia-smi

rm -rf .venv-full
"${UV}" venv --python python3 .venv-full
source .venv-full/bin/activate

"${UV}" pip install --upgrade pip
"${UV}" pip install --upgrade \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  "torch==2.11.0+cu128" \
  "transformers>=5.8.1" \
  "accelerate>=1.13.0" \
  "datasets>=4.8.0" \
  "sentencepiece>=0.2.0" \
  "jinja2>=3.1.0" \
  "huggingface_hub>=1.0.0" \
  "safetensors>=0.6.0" \
  "pyyaml>=6.0" \
  "wandb>=0.23.0"

python scripts/prepare_agent_data.py \
  --raw-dir "${RAW_DIR}" \
  --output "${DATA_FILE}" \
  --model-id google/gemma-4-31B-it \
  --skip-token-stats

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  scripts/train_gemma4_full_sft.py \
  --config "${CONFIG}"

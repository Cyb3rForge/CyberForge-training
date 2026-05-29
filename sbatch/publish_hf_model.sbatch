#!/usr/bin/env bash
#SBATCH --job-name=publish-gemma4-agent
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${PROJECT_DIR}/runs/gemma4-31b-full-mini-swe-secbench-modelonly-r3/checkpoint-1108}
REPO_ID=${REPO_ID:-MananSuri27/gemma4-31b-full-mini-swe-secbench}
HF_KEY_FILE=${HF_KEY_FILE:?Set HF_KEY_FILE to a local file containing your Hugging Face token}
TOKENIZER_ID=${TOKENIZER_ID:-google/gemma-4-31B-it}
BASE_MODEL=${BASE_MODEL:-google/gemma-4-31B-it}

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/slurm_logs"

export HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}
export HF_XET_HIGH_PERFORMANCE=1
export HF_TOKEN="$(tr -d '\n\r ' < "${HF_KEY_FILE}")"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export TOKENIZERS_PARALLELISM=false

echo "job=${SLURM_JOB_ID} node=$(hostname) date=$(date --iso-8601=seconds)"
echo "checkpoint=${CHECKPOINT_DIR}"
echo "repo=${REPO_ID}"

source "${PROJECT_DIR}/.venv/bin/activate"

python scripts/publish_hf_model.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --repo-id "${REPO_ID}" \
  --token-file "${HF_KEY_FILE}" \
  --tokenizer-id "${TOKENIZER_ID}" \
  --base-model "${BASE_MODEL}"

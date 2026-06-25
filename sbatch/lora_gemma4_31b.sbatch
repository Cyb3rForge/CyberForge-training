#!/usr/bin/env bash
# Full fine-tune of google/gemma-4-31B-it on the AmL-hug Vuljector SEC-bench
# trajectories, on the 8i-147gb partition (rw02-06: 8x full H200 147GB / node).
#
# Usage:
#   HF_KEY_FILE=/path/to/hf_key.txt \
#     sbatch sbatch/full_ft_gemma4_31b_8xh200.sbatch
#
# Optional:
#   WANDB_KEY_FILE=/path/to/wandb_key.txt   # enable W&B logging (else disabled)
#   CONFIG=configs/train/full_ft_gemma4_31b_8xh200.yml
#   DATASET_REPO=AmL-hug/vuljector-secbench-trajectories
#
#SBATCH --job-name=g4-lora-v3
#SBATCH --partition=8i-147gb
#SBATCH --qos=gpulimit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
# Nodes have 72 cores but only 62 are allocatable per node in this partition
# (10 are spec-reserved); requesting >62 fails with "node configuration not
# available". 48 is ample -- the cache is prebuilt and training is GPU-bound.
#SBATCH --cpus-per-task=48
# Training RSS peaks ~414G, but with save_only_model:false the FULL_STATE_DICT
# save gathers model (~124G) + AdamW optimizer state (~248G) to rank-0 CPU, so
# peak host RAM during a checkpoint approaches ~800G. Reserve 1400G for headroom
# (page cache during the write too). NB: the prior run's mid-write SIGKILL was
# the home disk quota, not memory -- now fixed by writing to Weka scratch.
#SBATCH --mem=1400G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}
UV=${UV:-uv}
CONFIG=${CONFIG:-${PROJECT_DIR}/configs/train/full_ft_gemma4_31b_8xh200.yml}
DATASET_REPO=${DATASET_REPO:-AmL-hug/vuljector-secbench-trajectories}
DATA_FILE=${DATA_FILE:-${PROJECT_DIR}/data/processed/gemma4_agent_trajectories.jsonl}
HF_KEY_FILE=${HF_KEY_FILE:?Set HF_KEY_FILE to a local file containing your Hugging Face token (needs access to the gated google/gemma-4-31B-it)}
INITIAL_MODEL_ID=${INITIAL_MODEL_ID:-google/gemma-4-31B-it}
NPROC=${SLURM_GPUS_ON_NODE:-8}

mkdir -p "${PROJECT_DIR}/slurm_logs" "${PROJECT_DIR}/data/processed" "${PROJECT_DIR}/runs"
cd "${PROJECT_DIR}"

# This cluster exposes a real CUDA toolkit via modules (proven with cuda/12.9 on
# the vLLM jobs). Pure-pytorch SDPA training does not JIT-compile kernels, so
# this is mostly belt-and-suspenders, but load it if available.
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.9 2>/dev/null || true

# Base model cache lives on the home quota (~78GB, home has room). Keeping it OFF
# the scratch quota frees ~59GB there as headroom for the 473GB resumable
# checkpoints, which must fit scratch's 1.1TB per-user quota.
export HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export ACCELERATE_USE_FSDP=false
export TRANSFORMERS_NO_TORCHVISION=1
# Pin FSDP1. accelerate's FSDP2 path breaks on this recipe (frozen vision tower
# is not sharded -> fsdp2_load_full_state_dict hits a plain Tensor with no
# device_mesh). FSDP1 is the mature path the torchtune-style recipe targets.
export HF_TOKEN="$(tr -d '\n\r ' < "${HF_KEY_FILE}")"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# W&B is optional. Supply WANDB_KEY_FILE to log metrics; otherwise disable it so
# the run never blocks on a missing API key.
if [[ -n "${WANDB_KEY_FILE:-}" && -f "${WANDB_KEY_FILE}" ]]; then
  export WANDB_API_KEY="$(tr -d '\n\r ' < "${WANDB_KEY_FILE}")"
else
  export WANDB_MODE=disabled
fi

export PROJECT_DIR DATA_FILE HF_KEY_FILE INITIAL_MODEL_ID

echo "job=${SLURM_JOB_ID:-none} node=$(hostname) gpus=${NPROC} date=$(date --iso-8601=seconds)"
nvidia-smi

# Isolated training venv (torch 2.11 cu128 + transformers>=5.8.1 for Gemma 4).
# Reused if already present; set REBUILD_VENV=1 to force a clean rebuild.
if [[ "${REBUILD_VENV:-0}" == "1" || ! -x .venv-full/bin/python ]]; then
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
else
  echo "reusing existing .venv-full"
  source .venv-full/bin/activate
fi

# The dataset already ships the processed training JSONL (output of
# prepare_agent_data.py), so just pull it to the path the config expects --
# no RAW_DIR / prepare step needed.
if [[ ! -s "${DATA_FILE}" ]]; then
  echo "Downloading processed trajectories from ${DATASET_REPO}"
  hf download "${DATASET_REPO}" --repo-type dataset \
    --include "gemma4_agent_trajectories.jsonl" \
    --local-dir "${PROJECT_DIR}/data/_hf_dataset"
  cp "${PROJECT_DIR}/data/_hf_dataset/gemma4_agent_trajectories.jsonl" "${DATA_FILE}"
fi
echo "data rows: $(wc -l < "${DATA_FILE}")"

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  scripts/train_gemma4_full_sft.py \
  --config "${CONFIG}"

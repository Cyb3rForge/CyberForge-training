#!/usr/bin/env bash
# Single-GPU Unsloth SFT of gemma-4-31B. Shared .venv-unsloth built once under flock so a
# parallel sweep (many sbatch of this file) doesn't race the install. Config via env vars.
#SBATCH --job-name=g4-uns
#SBATCH --partition=8i-147gb
#SBATCH --qos=gpulimit
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}
cd "${PROJECT_DIR}"
mkdir -p slurm_logs runs_unsloth

source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.9 2>/dev/null || true
export HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}
export HF_TOKEN="$(tr -d '\n\r ' < "${HF_KEY_FILE:-${HOME}/.cache/huggingface/token}")"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export TOKENIZERS_PARALLELISM=false
export UNSLOTH_RETURN_LOGITS=0

VENV=${VENV:-${PROJECT_DIR}/.venv-unsloth}
LOCK=${PROJECT_DIR}/.venv-unsloth.lock
# Build venv once (flock: first job builds, others block until it exists, then reuse).
(
  flock 200
  if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "[venv] building ${VENV} ..."
    uv venv --python 3.12 "${VENV}"
    source "${VENV}/bin/activate"
    uv pip install --upgrade pip
    # Unsloth pulls a compatible torch; then ensure transformers new enough for Gemma-4.
    uv pip install "unsloth" "unsloth_zoo" "trl" "datasets" "bitsandbytes" "wandb"
    uv pip install --upgrade "transformers>=5.8.1"
    python -c "import unsloth, transformers, trl; print('unsloth', unsloth.__version__, 'transformers', transformers.__version__)"
    echo "[venv] built."
  else
    echo "[venv] reusing ${VENV}"
  fi
) 200>"${LOCK}"
source "${VENV}/bin/activate"

nvidia-smi
echo "EXP=${EXP:-default} node=$(hostname) job=${SLURM_JOB_ID:-none}"

python scripts/train_gemma4_unsloth.py \
  --model "${MODEL:-google/gemma-4-31B-it}" \
  --data "${DATA:-${PROJECT_DIR}/data/processed/normalized_trajectories_v4.jsonl}" \
  --output_dir "${OUTPUT_DIR:-/lab/hpc/redwing/wrk/anl31/unsloth-runs/${EXP:-default}}" \
  --max_seq_len "${MAX_SEQ_LEN:-16384}" \
  --load_in_4bit "${LOAD_IN_4BIT:-0}" \
  --r "${R:-32}" --alpha "${ALPHA:-64}" --dropout "${DROPOUT:-0.0}" \
  --lr "${LR:-1e-4}" --epochs "${EPOCHS:-2.0}" \
  --bs "${BS:-1}" --grad_accum "${GRAD_ACCUM:-8}" \
  --scheduler "${SCHEDULER:-cosine}" --optim "${OPTIM:-adamw_8bit}" \
  --save_steps "${SAVE_STEPS:-60}" --rslora "${RSLORA:-0}" \
  --task "${TASK:-both}" --success_only "${SUCCESS_ONLY:-1}"

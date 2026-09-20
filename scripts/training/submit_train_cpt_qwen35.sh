#!/bin/bash
# Submit from the project root:
#   sbatch scripts/training/submit_train_cpt_qwen35.sh
# Optional Python arguments are forwarded after the defaults.
#
# Example smoke test:
#   sbatch scripts/training/submit_train_cpt_qwen35.sh --repositories redhat_v1 --max-steps 20

#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --job-name=qwen35-cpt
#SBATCH --output=logs/qwen35-cpt-%j.out

set -euo pipefail


PROJECT_ROOT="${HOME}/scratch/its_project/issue_link_retrieval_rl_post_training"
TRAINING_SCRIPT="${PROJECT_ROOT}/scripts/training/train_cpt_qwen35.py"
MODEL_ROOT="${HOME}/scratch/llms_model/ilr_llms"
MODEL_PATH="${MODEL_ROOT}/base/Qwen3.5-9B-Base"
ADAPTER_DIR="${MODEL_ROOT}/adapters/qwen3.5-9b-cpt-all-v1"
CHECKPOINT_DIR="${MODEL_ROOT}/checkpoints/qwen3.5-9b-cpt-all-v1"
CONDA_ENV_NAME="ilr_rl_post_training_env"

CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

JOB_TEMP_DIR="${SLURM_TMPDIR:-/tmp}/qwen35-cpt-${SLURM_JOB_ID:-manual}"
export HF_HOME="${JOB_TEMP_DIR}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${PROJECT_ROOT}"
mkdir -p "${HF_HOME}" "${ADAPTER_DIR}" "${CHECKPOINT_DIR}" "${PROJECT_ROOT}/logs"

echo "Slurm job ID: ${SLURM_JOB_ID:-not-submitted}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Working directory: $(pwd)"
echo "Conda environment: ${CONDA_DEFAULT_ENV:-unset}"
echo "Model path: ${MODEL_PATH}"
echo "Adapter directory: ${ADAPTER_DIR}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo "Arguments: $*"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
    echo "Expected Conda environment ${CONDA_ENV_NAME}, got ${CONDA_DEFAULT_ENV:-unset}" >&2
    exit 1
fi
if [[ ! -f "${TRAINING_SCRIPT}" ]]; then
    echo "Training script is missing: ${TRAINING_SCRIPT}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Qwen3.5 model config is missing: ${MODEL_PATH}/config.json" >&2
    echo "Download Qwen/Qwen3.5-9B-Base into ${MODEL_PATH} before submitting." >&2
    exit 1
fi

python --version
python -c "import torch, transformers, datasets, peft, accelerate, bitsandbytes; print('PyTorch:', torch.__version__); print('Transformers:', transformers.__version__); print('Datasets:', datasets.__version__); print('PEFT:', peft.__version__); print('Accelerate:', accelerate.__version__); print('BitsAndBytes:', bitsandbytes.__version__); assert torch.cuda.is_available(), 'Allocated GPU is not visible'; print('GPU:', torch.cuda.get_device_name(0)); print('CUDA:', torch.version.cuda); print('BF16:', torch.cuda.is_bf16_supported())"
nvidia-smi

# Defaults are intentionally explicit. Arguments supplied after the script can
# override them; duplicate argparse options use the last occurrence.
srun --unbuffered python "${TRAINING_SCRIPT}" \
    --project-root "${PROJECT_ROOT}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${ADAPTER_DIR}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --method qlora \
    --max-seq-length 2048 \
    --per-device-train-batch-size 1 \
    --per-device-eval-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --num-train-epochs 1 \
    "$@"

echo "Finished: $(date --iso-8601=seconds)"
echo "Adapter directory: ${ADAPTER_DIR}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"


#For your A100 80GB, LoRA is the better choice for CPT and SFT.
#
#   Factor                   LoRA                            QLoRA
#  ━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#   GPU memory               Higher                          Lower
#  ───────────────────────  ──────────────────────────────  ──────────────────────────────────────
#   Training speed           Usually faster                  Usually slower
#  ───────────────────────  ──────────────────────────────  ──────────────────────────────────────
#   Quantization overhead    None                            4-bit dequantization overhead
#  ───────────────────────  ──────────────────────────────  ──────────────────────────────────────
#   Training stability       Simpler                         More dependencies and compatibility
#                                                            issues
#  ───────────────────────  ──────────────────────────────  ──────────────────────────────────────
#   Expected quality         Preferred when memory allows    Usually close, sometimes slightly
#                                                            worse
#  ───────────────────────  ──────────────────────────────  ──────────────────────────────────────
#   Your A100 80GB           Suitable                        Also suitable
#
#  Your QLoRA job uses only about 25.6GB of 80GB, so you have enough memory to use LoRA and
#  increase the batch size.
#
#  Recommended run:
#
#  sbatch scripts/training/submit_train_cpt_qwen35.sh \
#    --method lora \
#    --per-device-train-batch-size 2 \
#    --gradient-accumulation-steps 8 \
#    --dataloader-num-workers 8
#
#  This keeps the same effective batch size as the current run:
#
#  1 × 16 = 2 × 8
#
#  but should improve throughput substantially.
#
#  Use QLoRA when:
#
#  - LoRA exceeds available memory
#  - You need larger sequence lengths
#  - You need multiple models or reference models on one GPU
#  - You want to reduce memory pressure
#
#  For this project, I would use:
#
#  CPT: LoRA
#  SFT: LoRA
#  GRPO/PPO: QLoRA if memory becomes tight
#
#  Since the current QLoRA run is projected to exceed 60 hours, LoRA with batch size 2 is the
#  better next run.



#!/bin/bash
# Submit one independent SFT job from the project root.
#
# Examples:
#   sbatch --partition=a100 --gres=gpu:1 \
#     --job-name=sft-set-base-apache-smoke \
#     scripts/training/submit_train_sft_qwen35.sh \
#     --task set_retrieval --initialization base --dataset Apache
#
#   sbatch --partition=a100 --gres=gpu:1 \
#     --job-name=sft-pw-cpt-all-smoke \
#     scripts/training/submit_train_sft_qwen35.sh \
#     --task pointwise --initialization cpt --dataset all --max-steps 20

#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --job-name=qwen35-sft
# `%x` is the Slurm job name and `%j` is the unique job ID. Supply a detailed
# name with `sbatch --job-name=...` before this script path.
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT="${HOME}/scratch/its_project/issue_link_retrieval_rl_post_training"
MODEL_ROOT="${HOME}/scratch/llms_model/ilr_llms"
CONDA_ENV_NAME="ilr_rl_post_training_env"

TASK=""
INITIALIZATION=""
DATASET=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            TASK="${2:?--task requires a value}"
            shift 2
            ;;
        --initialization)
            INITIALIZATION="${2:?--initialization requires a value}"
            shift 2
            ;;
        --dataset|--datasets)
            DATASET="${2:?--dataset requires a value}"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

TASK="${TASK,,}"
INITIALIZATION="${INITIALIZATION,,}"
DATASET="${DATASET,,}"

case "${TASK}" in
    set_retrieval)
        TRAINER="${PROJECT_ROOT}/scripts/training/train_sft_set_retrieval.py"
        DEFAULT_ARGS=(
            --max-seq-length 4096
            --max-steps 1000
            --gradient-accumulation-steps 8
        )
        ;;
    pointwise)
        TRAINER="${PROJECT_ROOT}/scripts/training/train_sft_pointwise.py"
        DEFAULT_ARGS=(
            --max-seq-length 2048
            --max-steps 2000
            --gradient-accumulation-steps 16
            --negative-keep-probability 0.10
        )
        ;;
    *)
        echo "--task must be set_retrieval or pointwise" >&2
        exit 2
        ;;
esac

case "${INITIALIZATION}" in
    base|cpt) ;;
    *)
        echo "--initialization must be base or cpt" >&2
        exit 2
        ;;
esac

case "${DATASET}" in
    apache|jira|redhat|mongodb|qt|mojang|all) ;;
    *)
        echo "--dataset must be Apache, Jira, RedHat, MongoDB, Qt, Mojang, or all" >&2
        exit 2
        ;;
esac

CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

JOB_TEMP_DIR="${SLURM_TMPDIR:-/tmp}/qwen35-sft-${SLURM_JOB_ID:-manual}"
export ILR_MODEL_ROOT="${MODEL_ROOT}"
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
mkdir -p "${HF_HOME}" "${PROJECT_ROOT}/logs"

echo "Slurm job ID: ${SLURM_JOB_ID:-not-submitted}"
echo "Slurm partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Allocated GPUs: ${SLURM_GPUS_ON_NODE:-unknown}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Task: ${TASK}"
echo "Initialization: ${INITIALIZATION}"
echo "Dataset: ${DATASET}"
echo "Model root: ${MODEL_ROOT}"
echo "Trainer: ${TRAINER}"
echo "Extra arguments: ${EXTRA_ARGS[*]:-none}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
    echo "Expected Conda environment ${CONDA_ENV_NAME}, got ${CONDA_DEFAULT_ENV:-unset}" >&2
    exit 1
fi
if [[ ! -f "${TRAINER}" ]]; then
    echo "Training script is missing: ${TRAINER}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_ROOT}/base/Qwen3.5-9B-Base/config.json" ]]; then
    echo "Base model is missing under ${MODEL_ROOT}/base/Qwen3.5-9B-Base" >&2
    exit 1
fi
if [[ "${INITIALIZATION}" == "cpt" && ! -f "${MODEL_ROOT}/adapters/qwen3.5-9b-cpt-all-v1/adapter_config.json" ]]; then
    echo "CPT adapter is missing under ${MODEL_ROOT}/adapters/qwen3.5-9b-cpt-all-v1" >&2
    exit 1
fi

python --version
python -c "import torch, transformers, datasets, peft, accelerate; print('PyTorch:', torch.__version__); print('Transformers:', transformers.__version__); print('Datasets:', datasets.__version__); print('PEFT:', peft.__version__); print('Accelerate:', accelerate.__version__); assert torch.cuda.is_available(), 'Allocated GPU is not visible'; print('GPU:', torch.cuda.get_device_name(0)); print('BF16:', torch.cuda.is_bf16_supported())"
nvidia-smi

# EXTRA_ARGS are last so an explicit submission argument overrides a default.
srun --unbuffered python "${TRAINER}" \
    --project-root "${PROJECT_ROOT}" \
    --model-root "${MODEL_ROOT}" \
    --initialization "${INITIALIZATION}" \
    --dataset "${DATASET}" \
    --method lora \
    --per-device-train-batch-size 1 \
    --per-device-eval-batch-size 1 \
    --learning-rate 1e-4 \
    --dataloader-num-workers 1 \
    "${DEFAULT_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

echo "Finished: $(date --iso-8601=seconds)"

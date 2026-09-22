#!/bin/bash
# Submit one SFT evaluation. Put Slurm options before this script path:
# sbatch --partition=a100 --gres=gpu:1 --job-name=eval-set-cpt-apache-test \
#   scripts/evaluation/submit_evaluate_sft.sh \
#   --task set_retrieval --initialization cpt --model-dataset apache --split test

#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=i7_h200
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --job-name=qwen35-eval
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT="${HOME}/scratch/its_project/issue_link_retrieval_rl_post_training"
MODEL_ROOT="${HOME}/scratch/llms_model/ilr_llms"
CONDA_ENV_NAME="ilr_rl_post_training_env"

TASK=""
INITIALIZATION=""
MODEL_DATASET=""
EVAL_DATASET=""
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
        --model-dataset|--dataset)
            MODEL_DATASET="${2:?--model-dataset requires a value}"
            shift 2
            ;;
        --eval-dataset)
            EVAL_DATASET="${2:?--eval-dataset requires a value}"
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
MODEL_DATASET="${MODEL_DATASET,,}"
EVAL_DATASET="${EVAL_DATASET,,}"
[[ -n "${EVAL_DATASET}" ]] || EVAL_DATASET="${MODEL_DATASET}"

case "${TASK}" in
    set_retrieval)
        EVALUATOR="${PROJECT_ROOT}/scripts/evaluation/evaluate_sft_set_retrieval.py"
        DEFAULT_ARGS=(--max-seq-length 4096 --max-new-tokens 128)
        ;;
    pointwise)
        EVALUATOR="${PROJECT_ROOT}/scripts/evaluation/evaluate_sft_pointwise.py"
        DEFAULT_ARGS=(--max-seq-length 2048 --threshold 0.5 --score-normalization sum --batch-size 2)
        ;;
    *)
        echo "--task must be set_retrieval or pointwise" >&2
        exit 2
        ;;
esac
case "${INITIALIZATION}" in base|cpt) ;; *) echo "--initialization must be base or cpt" >&2; exit 2 ;; esac
case "${MODEL_DATASET}" in apache|jira|redhat|mongodb|qt|mojang|all) ;; *) echo "Invalid --model-dataset" >&2; exit 2 ;; esac
case "${EVAL_DATASET}" in apache|jira|redhat|mongodb|qt|mojang|all) ;; *) echo "Invalid --eval-dataset" >&2; exit 2 ;; esac

CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

JOB_TEMP_DIR="${SLURM_TMPDIR:-/tmp}/qwen35-eval-${SLURM_JOB_ID:-manual}"
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
mkdir -p "${HF_HOME}" "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/experiment_results/evaluation"

echo "Slurm job ID: ${SLURM_JOB_ID:-not-submitted}"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Task: ${TASK}"
echo "Initialization: ${INITIALIZATION}"
echo "Model dataset: ${MODEL_DATASET}"
echo "Evaluation dataset: ${EVAL_DATASET}"
echo "Evaluator: ${EVALUATOR}"
echo "Extra arguments: ${EXTRA_ARGS[*]:-none}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
    echo "Expected ${CONDA_ENV_NAME}, got ${CONDA_DEFAULT_ENV:-unset}" >&2
    exit 1
fi
if [[ ! -f "${EVALUATOR}" ]]; then
    echo "Evaluator is missing: ${EVALUATOR}" >&2
    exit 1
fi

python --version
python -c "import torch, transformers, peft; print('PyTorch:', torch.__version__); print('Transformers:', transformers.__version__); print('PEFT:', peft.__version__); assert torch.cuda.is_available(), 'Allocated GPU is not visible'; print('GPU:', torch.cuda.get_device_name(0)); print('BF16:', torch.cuda.is_bf16_supported())"
nvidia-smi

# Extra evaluator arguments go last and can override defaults. Slurm arguments
# such as --partition and --gres belong before the shell-script path in sbatch.
srun --unbuffered python "${EVALUATOR}" \
    --project-root "${PROJECT_ROOT}" \
    --model-root "${MODEL_ROOT}" \
    --initialization "${INITIALIZATION}" \
    --model-dataset "${MODEL_DATASET}" \
    --eval-dataset "${EVAL_DATASET}" \
    "${DEFAULT_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

echo "Finished: $(date --iso-8601=seconds)"

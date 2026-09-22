#!/bin/bash
# Submit the 28 core evaluations: 7 datasets x 2 initializations x 2 tasks.
# This helper runs on a login node; it is not itself a Slurm job.

set -euo pipefail

PROJECT_ROOT="${HOME}/scratch/its_project/issue_link_retrieval_rl_post_training"
PARTITION="a100"
GRES="gpu:1"
SPLIT="test"
DATA_VERSION="v1_full"
ADAPTER_SUFFIX=""
MAX_SAMPLES=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project-root) PROJECT_ROOT="${2:?missing value}"; shift 2 ;;
        --partition) PARTITION="${2:?missing value}"; shift 2 ;;
        --gres) GRES="${2:?missing value}"; shift 2 ;;
        --split) SPLIT="${2:?missing value}"; shift 2 ;;
        --data-version) DATA_VERSION="${2:?missing value}"; shift 2 ;;
        --adapter-suffix) ADAPTER_SUFFIX="${2:?missing value}"; shift 2 ;;
        --max-samples) MAX_SAMPLES="${2:?missing value}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown matrix argument: $1" >&2; exit 2 ;;
    esac
done

case "${SPLIT}" in validation|test) ;; *) echo "--split must be validation or test" >&2; exit 2 ;; esac
if ! [[ "${MAX_SAMPLES}" =~ ^[0-9]+$ ]]; then
    echo "--max-samples must be a nonnegative integer" >&2
    exit 2
fi

cd "${PROJECT_ROOT}"
SUBMITTER="scripts/evaluation/submit_evaluate_sft.sh"
if [[ ! -f "${SUBMITTER}" ]]; then
    echo "Missing ${PROJECT_ROOT}/${SUBMITTER}" >&2
    exit 1
fi

datasets=(apache redhat jira mongodb qt mojang all)
initializations=(base cpt)
tasks=(set_retrieval pointwise)

submitted=0
for dataset in "${datasets[@]}"; do
    for initialization in "${initializations[@]}"; do
        for task in "${tasks[@]}"; do
            if [[ "${task}" == "set_retrieval" ]]; then task_name="set"; else task_name="pw"; fi
            job_name="eval-${task_name}-${initialization}-${dataset}-${SPLIT}"
            command=(
                sbatch --partition="${PARTITION}" --gres="${GRES}" --job-name="${job_name}"
                "${SUBMITTER}"
                --task "${task}"
                --initialization "${initialization}"
                --model-dataset "${dataset}"
                --eval-dataset "${dataset}"
                --split "${SPLIT}"
                --data-version "${DATA_VERSION}"
            )
            if [[ -n "${ADAPTER_SUFFIX}" ]]; then
                command+=(--adapter-suffix "${ADAPTER_SUFFIX}")
            fi
            if (( MAX_SAMPLES > 0 )); then
                command+=(--max-samples "${MAX_SAMPLES}")
            fi
            if (( DRY_RUN )); then
                printf '%q ' "${command[@]}"
                printf '\n'
            else
                "${command[@]}"
            fi
            submitted=$((submitted + 1))
        done
    done
done

if (( DRY_RUN )); then
    echo "Planned ${submitted} evaluation jobs."
else
    echo "Submitted ${submitted} evaluation jobs."
fi

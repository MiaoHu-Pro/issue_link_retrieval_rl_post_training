# Running the 28 SFT jobs

For choosing A100, H100, H200, L40S, L4, or scavenger partitions, see `docs/gpu_partition_selection_guide.md`.

This guide contains all 28 independent SFT job commands:

\[
7\text{ dataset configurations}
\times 2\text{ initializations}
\times 2\text{ tasks}
=28\text{ jobs}.
\]

The dataset configurations are Apache, Jira, RedHat, MongoDB, Qt, Mojang, and the temperature-balanced mixture of all six repositories. The two initializations are Qwen3.5-9B-Base and the completed CPT adapter. The two tasks are set retrieval and pointwise classification.

## 1. Before submitting

Run from the project root on the server:

```bash
cd ~/scratch/its_project/issue_link_retrieval_rl_post_training
mkdir -p logs
```

The launcher expects:

```text
Base model:
~/scratch/llms_model/ilr_llms/base/Qwen3.5-9B-Base

CPT adapter:
~/scratch/llms_model/ilr_llms/adapters/qwen3.5-9b-cpt-all-v1

Conda environment:
ilr_rl_post_training_env
```

The commands below are **10-step smoke tests**. Each command uses `--run-suffix smoke`, so its adapter and checkpoints cannot overwrite or block the later production run.

`--run-suffix smoke` is necessary for the commands in this guide because they deliberately run only 10 steps. It is an output-isolation label; it does not change the model, data, loss, or optimizer. For a production run, remove `--run-suffix smoke` and the three short-run overrides, or replace the suffix with a seed label such as `seed-42` for final repeated experiments.

### Slurm options must precede the script

Resource and log-naming arguments belong to `sbatch` and must be written before the shell-script path:

The required order is `sbatch [Slurm options] scripts/training/submit_train_sft_qwen35.sh [training options]`. Every command below follows this order and has a unique job name.

This creates a log such as:

```text
logs/sft-set-base-apache-smoke-1602345.out
```

Do not put `--partition`, `--gres`, `--time`, `--job-name`, or `--output` after `submit_train_sft_qwen35.sh`. Arguments after the script path are application arguments and are forwarded to the Python trainer.

Use this job-name convention:

```text
sft-{set|pw}-{base|cpt}-{apache|jira|redhat|mongodb|qt|mojang|all}-{smoke|prod|seed-N}
```

Examples:

```text
sft-set-cpt-apache-smoke
sft-pw-base-redhat-prod
sft-set-cpt-all-seed-42
```

The commands explicitly use `--data-version v1_full`. These pools are suitable for software smoke tests and the v1 lexical-retrieval ablation. For the final paper protocol, create the corrected v2 pools and change every command to `--data-version v2`.

## 2. Apache: four jobs

### A1-SR — Base, set retrieval, Apache

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-set-base-apache-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization base \
  --dataset Apache \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

### A2-SR — CPT, set retrieval, Apache

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-set-cpt-apache-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization cpt \
  --dataset Apache \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

### A1-PW — Base, pointwise, Apache

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-pw-base-apache-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization base \
  --dataset Apache \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

### A2-PW — CPT, pointwise, Apache

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-pw-cpt-apache-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization cpt \
  --dataset Apache \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

## 3. Jira: four jobs

### J1-SR — Base, set retrieval, Jira

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-set-base-jira-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization base \
  --dataset Jira \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

### J2-SR — CPT, set retrieval, Jira

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-set-cpt-jira-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization cpt \
  --dataset Jira \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

### J1-PW — Base, pointwise, Jira

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-pw-base-jira-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization base \
  --dataset Jira \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

### J2-PW — CPT, pointwise, Jira

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-pw-cpt-jira-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization cpt \
  --dataset Jira \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

## 4. RedHat: four jobs

### R1-SR — Base, set retrieval, RedHat

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-set-base-redhat-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization base \
  --dataset RedHat \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

### R2-SR — CPT, set retrieval, RedHat

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-set-cpt-redhat-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization cpt \
  --dataset RedHat \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

### R1-PW — Base, pointwise, RedHat

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-pw-base-redhat-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization base \
  --dataset RedHat \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

### R2-PW — CPT, pointwise, RedHat

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-pw-cpt-redhat-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization cpt \
  --dataset RedHat \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

## 5. MongoDB: four jobs

### MDB1-SR — Base, set retrieval, MongoDB

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-set-base-mongodb-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization base \
  --dataset MongoDB \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

### MDB2-SR — CPT, set retrieval, MongoDB

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-set-cpt-mongodb-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization cpt \
  --dataset MongoDB \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

### MDB1-PW — Base, pointwise, MongoDB

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-pw-base-mongodb-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization base \
  --dataset MongoDB \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

### MDB2-PW — CPT, pointwise, MongoDB

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-pw-cpt-mongodb-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization cpt \
  --dataset MongoDB \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

## 6. Qt: four jobs

### Q1-SR — Base, set retrieval, Qt

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-set-base-qt-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization base \
  --dataset Qt \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

### Q2-SR — CPT, set retrieval, Qt

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-set-cpt-qt-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization cpt \
  --dataset Qt \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

### Q1-PW — Base, pointwise, Qt

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-pw-base-qt-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization base \
  --dataset Qt \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

### Q2-PW — CPT, pointwise, Qt

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-pw-cpt-qt-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization cpt \
  --dataset Qt \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

## 7. Mojang: four jobs

### MJ1-SR — Base, set retrieval, Mojang

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-set-base-mojang-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization base \
  --dataset Mojang \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

### MJ2-SR — CPT, set retrieval, Mojang

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-set-cpt-mojang-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization cpt \
  --dataset Mojang \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

### MJ1-PW — Base, pointwise, Mojang

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-pw-base-mojang-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization base \
  --dataset Mojang \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

### MJ2-PW — CPT, pointwise, Mojang

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-pw-cpt-mojang-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization cpt \
  --dataset Mojang \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

## 8. All six repositories: four jobs

The `all` runs stream all six repositories and use temperature-balanced sampling with exponent 0.5 by default.

### T1-SR — Base, set retrieval, all repositories

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-set-base-all-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization base \
  --dataset all \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

### T2-SR — CPT, set retrieval, all repositories

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-set-cpt-all-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization cpt \
  --dataset all \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

### T1-PW — Base, pointwise, all repositories

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-pw-base-all-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization base \
  --dataset all \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

### T2-PW — CPT, pointwise, all repositories

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=sft-pw-cpt-all-smoke \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization cpt \
  --dataset all \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

## 9. Check submitted and running jobs

```bash
squeue -u "${USER}"
```

Inspect one log using its Slurm job ID:

```bash
tail -f logs/qwen35-sft-JOB_ID.out
```

Cancel a specific failed or incorrect job if necessary:

```bash
scancel JOB_ID
```

Do not cancel a healthy job only because model loading or the first streaming batch takes several minutes.

## 10. Smoke-test output directories

Examples of isolated smoke outputs are:

```text
~/scratch/llms_model/ilr_llms/adapters/qwen3.5-9b-base-sft-set-apache-v1-full-smoke
~/scratch/llms_model/ilr_llms/adapters/qwen3.5-9b-cpt-sft-set-apache-v1-full-smoke
~/scratch/llms_model/ilr_llms/adapters/qwen3.5-9b-base-sft-pointwise-apache-v1-full-smoke
~/scratch/llms_model/ilr_llms/adapters/qwen3.5-9b-cpt-sft-pointwise-apache-v1-full-smoke
```

Checkpoints use the parallel `checkpoints/` hierarchy. Each adapter directory should contain `run_config.json`; a completed run also contains adapter weights, tokenizer files, and `validation_metrics.json`.

## 11. Convert a smoke command into a production command

After all relevant smoke tests pass, remove these four lines from a command:

```text
--run-suffix smoke
--max-steps 10
--eval-steps 5
--save-steps 5
```

For example, convert the Apache CPT set-retrieval command in Section 2 by retaining its task, initialization, dataset, and data-version arguments while removing the four smoke arguments above.

The launcher then uses these initial defaults:

| Task | Maximum sequence length | Maximum steps | Evaluation interval | Checkpoint interval |
|---|---:|---:|---:|---:|
| Set retrieval | 4096 | 1000 | 250 | 250 |
| Pointwise | 2048 | 2000 | 250 | 250 |

These are engineering starting points. Select final steps, learning rate, candidate-pool version, and stopping checkpoint using validation data. Do not select them from test results.

For final paper runs, replace the smoke suffix with a seed-specific suffix and set the matching seed, such as `--run-suffix seed-42 --seed 42`. Use matching `seed-43` and `seed-44` runs only after the v2 data and final validation-selected hyperparameters have been frozen.

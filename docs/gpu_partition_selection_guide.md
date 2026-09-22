# GPU and Slurm Partition Selection Guide

## 1. Recommendation for the current SFT code

The current Qwen3.5-9B SFT implementation is a **single-GPU trainer**. It places the model on CUDA device 0 and launches one Python training process. Request exactly one GPU:

```text
--gres=gpu:1
```

Requesting two, four, or eight GPUs will not accelerate the current code and will leave the additional GPUs unused. Multi-GPU training would require a separate implementation using DDP, FSDP, or DeepSpeed, removal of the fixed `device_map={"": 0}`, and a distributed launcher such as `torchrun` or `srun` with one process per GPU.

For the current 28 SFT jobs:

| Workload | Recommended GPU | Reason |
|---|---|---|
| Pointwise, 2,048 tokens, LoRA | One A100 80 GB | Sufficient memory and good throughput |
| Set retrieval, 4,096 tokens, LoRA | One A100 80 GB | Default supported configuration |
| Set retrieval with larger batch/context | One H200 142 GB | More activation memory |
| Fast set-retrieval production run | One H100 or H200 | Higher throughput when access and queue time permit |
| QLoRA smoke test | One A100, L40S, or possibly L4 | Lower memory, but slower and dependent on bitsandbytes/CUDA support |
| Future GRPO/PPO | H200 or multi-GPU H100/H200 after code changes | Policy rollouts and reference models need more memory |

The default recommendation is therefore:

```text
Pointwise SFT:     a100, one A100 80 GB
Set-retrieval SFT: a100, one A100 80 GB
Large/slow set run: one H200 if the A100 estimate approaches 60 hours
```

## 2. How Slurm selects the GPU type

The GPU type is selected by the Slurm partition. The existing launcher contains:

```bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
```

Therefore, an ordinary submission uses one A100:

```bash
sbatch scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization cpt \
  --dataset Apache \
  --data-version v1_full \
  --run-suffix smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

To use a different GPU partition, put Slurm resource options **before the script path**. Command-line `sbatch` resource options override the `#SBATCH` defaults inside the script.

The same rule applies to `--job-name` and `--output`. Options placed after the script path are passed to the SFT application and will cause an "unrecognized arguments" error if they are Slurm options.

General form:

```bash
sbatch \
  --partition=PARTITION_NAME \
  --gres=gpu:1 \
  --time=HH:MM:SS \
  scripts/training/submit_train_sft_qwen35.sh \
  --task TASK \
  --initialization INITIALIZATION \
  --dataset DATASET
```

Options after the script path belong to the SFT launcher. Options before the script path belong to Slurm.

## 3. Available partitions and suitability

### `a100`

Hardware: dual A100 nodes, 80 GB per GPU. Maximum walltime: 60 hours.

This is the main partition for the current SFT jobs. Use it for pointwise and set retrieval.

```bash
sbatch \
  --partition=a100 \
  --gres=gpu:1 \
  --time=60:00:00 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization cpt \
  --dataset Apache \
  --data-version v1_full
```

The launcher already contains these resource defaults, so the shorter command is equivalent.

### `maths_a100`

Hardware: dual A100 node, 80 GB per GPU. Maximum walltime: 60 hours. Priority is for Mathematical Sciences researchers.

Use only if your account is entitled to this partition:

```bash
sbatch \
  --partition=maths_a100 \
  --gres=gpu:1 \
  --time=60:00:00 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization base \
  --dataset Apache \
  --data-version v1_full
```

### `swarm_a100`

Hardware: four A100 SXM4 GPUs per node, 80 GB each, with NVLink. Maximum walltime: 120 hours. Private access applies.

For the current single-GPU code, request one GPU. The 120-hour limit is useful for a long run even though NVLink is not used:

```bash
sbatch \
  --partition=swarm_a100 \
  --gres=gpu:1 \
  --time=120:00:00 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization cpt \
  --dataset all \
  --data-version v1_full
```

Use this only if your research group has access.

### `swarm_h100`

Hardware: eight H100 GPUs per node, 80 GB each, with NVLink/NVSwitch. Maximum walltime: 120 hours. Private access applies.

An H100 should provide better training throughput than an A100 for BF16 transformer workloads. Request one GPU with the current trainer:

```bash
sbatch \
  --partition=swarm_h100 \
  --gres=gpu:1 \
  --time=120:00:00 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization cpt \
  --dataset all \
  --data-version v1_full
```

Use it when you have eligible access and the expected reduction in runtime justifies using the scarcer partition.

### `quad_h200`

Hardware: four H200 NVL GPUs per node, 142 GB each. Maximum walltime: 60 hours.

This is a strong choice for 4,096-token set retrieval, larger batch sizes, or experiments that run out of memory on an A100:

```bash
sbatch \
  --partition=quad_h200 \
  --gres=gpu:1 \
  --time=60:00:00 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization cpt \
  --dataset Apache \
  --data-version v1_full \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4
```

The example preserves the default effective batch size of eight while testing whether batch size two improves throughput. Confirm memory use and examples per second with a smoke run before adopting it.

### `dual_h200`

Hardware: two H200 NVL GPUs per node, 142 GB each. Maximum walltime: 60 hours.

Use it in the same way as `quad_h200`; the current job still requests one GPU:

```bash
sbatch \
  --partition=dual_h200 \
  --gres=gpu:1 \
  --time=60:00:00 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization base \
  --dataset RedHat \
  --data-version v1_full
```

### `i7_h200`

Hardware: four H200 GPUs per node, 142 GB each, with high system-memory capacity. Access uses the `i7_h200` partition. Maximum walltime should be checked with `scontrol show partition i7_h200` before submission if it is not stated in the current service documentation.

```bash
sbatch \
  --partition=i7_h200 \
  --gres=gpu:1 \
  --time=60:00:00 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization cpt \
  --dataset Mojang \
  --data-version v1_full
```

The large node RAM can help with indexing and dataset preparation, although the streaming SFT trainer itself requests only 80 GB system memory by default.

### `l40`

Hardware: L40S GPUs with 42 GB VRAM. Maximum walltime: 60 hours.

Use QLoRA rather than LoRA. Pointwise training is more suitable than 4,096-token set retrieval:

```bash
sbatch \
  --partition=l40 \
  --gres=gpu:1 \
  --time=60:00:00 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization base \
  --dataset Qt \
  --data-version v1_full \
  --method qlora
```

Run a smoke test first. The launcher imports bitsandbytes only when QLoRA model loading requires it.

### `l4`

Hardware: L4 GPUs with 24 GB VRAM. Maximum walltime: 60 hours.

This is not recommended for the main Qwen3.5-9B LoRA study. It may support a constrained QLoRA smoke test with batch size one and a shorter context:

```bash
sbatch \
  --partition=l4 \
  --gres=gpu:1 \
  --time=02:00:00 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization base \
  --dataset Qt \
  --data-version v1_full \
  --method qlora \
  --max-seq-length 1024 \
  --run-suffix l4-smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

Results from a 1,024-token L4 run are not directly comparable with the standard 2,048-token pointwise configuration.

### `mi300x`

Hardware: MI300X GPUs with 192 GB VRAM. Maximum walltime: 60 hours.

Do not use this partition with the current environment or launcher. The current stack assumes NVIDIA CUDA and uses bitsandbytes for QLoRA. MI300X requires a validated ROCm PyTorch environment and may require changes to quantization, model loading, device checks, and the Slurm launcher. Treat MI300X support as a separate engineering task.

## 4. Scavenger partitions

The following partitions are preemptible with a 12-hour maximum:

```text
scavenger_mathsa100
scavenger_4a100
scavenger_8h100
scavenger_l4
```

They are appropriate for smoke tests, profiling, evaluation, or short resumable training segments. They are risky for an uninterrupted production run.

Example H100 scavenger smoke test:

```bash
sbatch \
  --partition=scavenger_8h100 \
  --gres=gpu:1 \
  --time=02:00:00 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization cpt \
  --dataset Apache \
  --data-version v1_full \
  --run-suffix h100-smoke \
  --max-steps 10 \
  --eval-steps 5 \
  --save-steps 5
```

For longer scavenger experiments, reduce `--save-steps` so useful checkpoints exist before preemption. Resume explicitly from a saved Trainer checkpoint:

```text
--resume-from-checkpoint /absolute/path/to/checkpoint-N
```

## 5. Suggested allocation of the 28 jobs

Start with four Apache smoke tests on one A100 each:

```text
A1-SR, A2-SR, A1-PW, A2-PW
```

After those pass:

| Job family | Preferred partition | Alternative |
|---|---|---|
| Six repository-specific pointwise Base runs | `a100` | `l40` with QLoRA |
| Six repository-specific pointwise CPT runs | `a100` | H100/H200 if queue is shorter |
| Six repository-specific set Base runs | `a100` | `quad_h200` or `dual_h200` |
| Six repository-specific set CPT runs | `a100` | H100/H200 for the largest corpora |
| All-six pointwise Base/CPT | `a100` | H100/H200 |
| All-six set Base/CPT | H100/H200 if available | `a100`, monitoring the 60-hour limit |

Queue waiting time can outweigh raw GPU speed. Check the current queues before choosing:

```bash
sinfo -o "%P %a %l %D %G"
squeue -u "${USER}"
```

If permitted by the service, `squeue --start -j JOB_ID` may provide an estimated start time.

## 6. Benchmark before choosing final resources

GPU names alone do not determine total runtime. Sequence length, model architecture, checkpointing, tokenization, JSON streaming, and evaluation frequency also matter. Run 20–50 training steps on each candidate partition and record:

```text
seconds per step
examples per second
tokens per second
maximum allocated GPU memory
GPU utilization
validation time
```

Monitor an allocated job with:

```bash
srun --jobid=JOB_ID --overlap nvidia-smi
```

Estimate training time as:

\[
T_{train}\approx T_{startup}+
(N_{steps}\times T_{step})+
(N_{eval}\times T_{eval})+
T_{checkpoint}.
\]

Do not start a production run whose conservative estimate exceeds the partition walltime. Use a 120-hour eligible partition, reduce the validation frequency, reduce the selected maximum steps based on validation, or divide the run into resumable checkpoints.

## 7. Memory tuning order

If an A100 LoRA set-retrieval run is out of memory, change one factor at a time:

1. Keep per-device batch size at one.
2. Reduce maximum sequence length only if the candidate records still fit and the experiment is recorded as a different configuration.
3. Switch from LoRA to QLoRA.
4. Move to one H200.
5. Implement distributed training before requesting multiple GPUs.

Do not request multiple GPUs with the current code as a memory workaround. The fixed single-device model loading prevents those GPUs from sharing the model.

## 8. Reproducibility rules

- Compare Base and CPT on the same GPU type whenever possible.
- Keep sequence length, effective batch size, maximum steps, data version, and evaluation frequency identical within each Base-versus-CPT comparison.
- Record the physical GPU name printed by the launcher, not only the partition name.
- Record package versions, Slurm job ID, seed, runtime, maximum memory, and final checkpoint.
- A run moved from A100 to H200 is comparable for quality if numerical precision and all training hyperparameters remain fixed, but runtime and cost must be reported separately.
- Do not mix LoRA and QLoRA inside a direct initialization comparison unless method is an explicit experimental factor.

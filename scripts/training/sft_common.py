#!/usr/bin/env python3
"""Shared streaming SFT implementation for issue-link selection tasks."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any


REPOSITORIES = ("apache", "jira", "redhat", "mongodb", "qt", "mojang")
TASK_SLUG = {"set_retrieval": "set", "pointwise": "pointwise"}


def stable_fraction(value: str) -> float:
    number = int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)
    return number / float(16**16 - 1)


def build_parser(task: str) -> argparse.ArgumentParser:
    project = Path(__file__).resolve().parents[2]
    model_root = Path(os.environ.get("ILR_MODEL_ROOT", "~/scratch/llms_model/ilr_llms")).expanduser()
    default_length = 4096 if task == "set_retrieval" else 2048
    default_steps = 1000 if task == "set_retrieval" else 2000
    parser = argparse.ArgumentParser(
        description=f"Stream and train Qwen3.5 for the {task} issue-link SFT task."
    )
    parser.add_argument("--project-root", type=Path, default=project)
    parser.add_argument("--model-root", type=Path, default=model_root)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--cpt-adapter-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--run-suffix", default="",
        help="Optional output-name suffix, for example smoke or seed-42.",
    )
    parser.add_argument("--initialization", choices=("base", "cpt"), required=True)
    parser.add_argument("--dataset", type=str.lower, choices=(*REPOSITORIES, "all"), required=True)
    parser.add_argument(
        "--data-version", default="v1_full",
        help="Directory suffix, e.g. v1_full resolves apache_v1_full. Use v2 for corrected pools.",
    )
    parser.add_argument("--method", choices=("lora", "qlora"), default="lora")
    parser.add_argument("--max-seq-length", type=int, default=default_length)
    parser.set_defaults(fixed_step_default=default_steps)
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Explicit fixed optimizer-step budget. If omitted, --dataset all makes one exhaustive pass.",
    )
    parser.add_argument(
        "--training-mode", choices=("auto", "fixed_steps", "exhaustive"), default="auto",
        help="auto uses exhaustive for dataset=all unless --max-steps is supplied; other runs stay fixed-step.",
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-eval-samples", type=int, default=1000)
    parser.add_argument("--shuffle-buffer-size", type=int, default=4096)
    parser.add_argument("--dataloader-num-workers", type=int, default=1)
    parser.add_argument("--sampling", choices=("proportional", "temperature"), default="temperature")
    parser.add_argument("--sampling-temperature", type=float, default=0.5)
    parser.add_argument(
        "--negative-keep-probability", type=float, default=None,
        help="Pointwise negative retention. Defaults to 1.0 for exhaustive mode and 0.10 otherwise.",
    )
    parser.add_argument(
        "--empty-keep-probability", type=float, default=None,
        help="Set retrieval only: retain this fraction of empty-target training rows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--auto-resume", action=argparse.BooleanOptionalAction, default=True,
        help="Resume the newest checkpoint in the resolved checkpoint directory when present.",
    )
    parser.add_argument(
        "--overwrite-output", action="store_true",
        help="Allow final adapter files for this exact run name to be replaced after successful training.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def normalize_paths(args: argparse.Namespace, task: str) -> argparse.Namespace:
    args.project_root = args.project_root.expanduser().resolve()
    args.model_root = args.model_root.expanduser().resolve()
    args.model_path = (args.model_path or args.model_root / "base/Qwen3.5-9B-Base").expanduser().resolve()
    args.cpt_adapter_path = (
        args.cpt_adapter_path or args.model_root / "adapters/qwen3.5-9b-cpt-all-v1"
    ).expanduser().resolve()
    dataset_slug = args.dataset
    version_slug = args.data_version.replace("/", "-").replace("_", "-")
    run_name = f"qwen3.5-9b-{args.initialization}-sft-{TASK_SLUG[task]}-{dataset_slug}-{version_slug}"
    if args.run_suffix:
        safe_suffix = args.run_suffix.replace("/", "-").replace("_", "-")
        run_name += f"-{safe_suffix}"
    args.output_dir = (args.output_dir or args.model_root / "adapters" / run_name).expanduser().resolve()
    args.checkpoint_dir_was_explicit = args.checkpoint_dir is not None
    args.checkpoint_dir = (
        args.checkpoint_dir or args.model_root / "checkpoints" / run_name
    ).expanduser().resolve()
    return args


def selected_repositories(dataset: str) -> tuple[str, ...]:
    return REPOSITORIES if dataset == "all" else (dataset,)


def dataset_paths(args: argparse.Namespace, task: str) -> dict[str, dict[str, Path]]:
    root = args.project_root / "data/training/sft" / task
    result: dict[str, dict[str, Path]] = {}
    for repository in selected_repositories(args.dataset):
        directory = root / f"{repository}_{args.data_version}"
        paths = {split: directory / f"{split}.jsonl" for split in ("train", "validation", "test")}
        paths["manifest"] = directory / "manifest.json"
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing SFT dataset files:\n  " + "\n  ".join(missing))
        result[repository] = paths
    return result


def read_manifests(paths: dict[str, dict[str, Path]], task: str) -> dict[str, dict[str, Any]]:
    prefix = "set" if task == "set_retrieval" else "point"
    manifests = {}
    for repository, repository_paths in paths.items():
        manifest = json.loads(repository_paths["manifest"].read_text(encoding="utf-8"))
        for split in ("train", "validation", "test"):
            key = f"{prefix}_{split}"
            if key not in manifest.get("counts", {}):
                raise ValueError(f"{repository_paths['manifest']} lacks counts.{key}")
        manifests[repository] = manifest
    return manifests


def resolve_training_plan(args: argparse.Namespace, task: str, manifests) -> argparse.Namespace:
    """Resolve an exact one-pass all-repository run while leaving other runs fixed-step."""
    explicit_steps = args.max_steps is not None
    if args.training_mode == "auto":
        args.training_mode = "fixed_steps" if explicit_steps or args.dataset != "all" else "exhaustive"
    if args.training_mode == "exhaustive" and args.dataset != "all":
        raise ValueError("Exhaustive mode is restricted to --dataset all; repository-specific jobs are unchanged")
    if args.training_mode == "exhaustive" and explicit_steps:
        raise ValueError("Do not combine --training-mode exhaustive with --max-steps")

    args.negative_keep_probability = (
        args.negative_keep_probability
        if args.negative_keep_probability is not None
        else (1.0 if args.training_mode == "exhaustive" else 0.10)
    )
    args.empty_keep_probability = (
        args.empty_keep_probability if args.empty_keep_probability is not None else 1.0
    )
    prefix = "set" if task == "set_retrieval" else "point"
    args.retained_records_by_repository = {}
    for repository, manifest in manifests.items():
        total = int(manifest["counts"][f"{prefix}_train"])
        if task == "pointwise" and args.training_mode == "exhaustive":
            positives = int(manifest.get("eligible_gold", {}).get("train", -1))
            if positives < 0:
                raise ValueError(f"{repository} manifest lacks eligible_gold.train")
            negatives = total - positives
            retained = positives + round(args.negative_keep_probability * negatives)
        else:
            retained = total
        args.retained_records_by_repository[repository] = retained
    args.records_per_pass = sum(args.retained_records_by_repository.values())
    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if args.training_mode == "exhaustive":
        if args.empty_keep_probability != 1.0:
            raise ValueError("Exhaustive mode requires --empty-keep-probability 1.0")
        if task == "set_retrieval" and args.negative_keep_probability != 1.0:
            raise ValueError("--negative-keep-probability applies only to pointwise training")
        if task == "pointwise" and args.negative_keep_probability < 1.0 and args.dataloader_num_workers != 1:
            raise ValueError("Exact pointwise negative selection requires --dataloader-num-workers 1")
        args.max_steps = math.ceil(args.records_per_pass / effective_batch)
        # Do not reuse checkpoints from the former short fixed-step all-data run.
        if not args.checkpoint_dir_was_explicit:
            args.checkpoint_dir = args.checkpoint_dir.with_name(args.checkpoint_dir.name + "-exhaustive")
    else:
        args.max_steps = args.max_steps or args.fixed_step_default
    args.logging_steps = args.logging_steps or (
        max(10, math.ceil(args.max_steps / 1000)) if args.training_mode == "exhaustive" else 10
    )
    args.eval_steps = args.eval_steps or (
        max(250, math.ceil(args.max_steps / 20)) if args.training_mode == "exhaustive" else 250
    )
    args.save_steps = args.save_steps or (
        min(1000, args.eval_steps) if args.training_mode == "exhaustive" else 250
    )
    return args


def newest_checkpoint(directory: Path) -> Path | None:
    candidates = []
    if directory.is_dir():
        for path in directory.glob("checkpoint-*"):
            try:
                step = int(path.name.rsplit("-", 1)[1])
            except ValueError:
                continue
            if path.is_dir():
                candidates.append((step, path))
    return max(candidates, default=(0, None))[1]


def sampling_probabilities(
    repositories: tuple[str, ...], manifests: dict[str, dict[str, Any]], task: str,
    sampling: str, temperature: float,
) -> list[float] | None:
    if len(repositories) == 1 or sampling == "proportional":
        return None
    if not 0.0 <= temperature <= 1.0:
        raise ValueError("--sampling-temperature must be between 0 and 1")
    prefix = "set" if task == "set_retrieval" else "point"
    weights = [float(manifests[repo]["counts"][f"{prefix}_train"]) ** temperature for repo in repositories]
    total = sum(weights)
    return [weight / total for weight in weights]


def import_training_stack():
    try:
        import torch
        import transformers
        from datasets import concatenate_datasets, interleave_datasets, load_dataset
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise SystemExit(
            "Install training dependencies with the project training extra. "
            f"Missing package: {exc}"
        ) from exc
    return torch, transformers, load_dataset, interleave_datasets, concatenate_datasets, LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


def load_model(args, torch, transformers, LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training):
    if not (args.model_path / "config.json").is_file():
        raise FileNotFoundError(f"Base model is missing: {args.model_path}")
    if args.initialization == "cpt" and not (args.cpt_adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"CPT adapter is missing: {args.cpt_adapter_path}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {"dtype": torch.bfloat16, "trust_remote_code": args.trust_remote_code}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}
    if args.method == "qlora":
        if not torch.cuda.is_available():
            raise SystemExit("QLoRA requires CUDA")
        model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model_class = getattr(transformers, "AutoModelForImageTextToText", transformers.AutoModelForCausalLM)
    try:
        model = model_class.from_pretrained(args.model_path, **model_kwargs)
    except (ValueError, OSError):
        model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    except TypeError:
        model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = model_class.from_pretrained(args.model_path, **model_kwargs)

    if args.method == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model.enable_input_require_grads()

    if args.initialization == "cpt":
        # Continue from a copied in-memory CPT adapter and save the resulting
        # CPT+SFT adapter to a new directory. The original CPT files stay intact.
        model = PeftModel.from_pretrained(model, args.cpt_adapter_path, is_trainable=True)
    else:
        lora = LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules="all-linear",
        )
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer


def token_id_list(value) -> list[int]:
    """Normalize Transformers chat/tokenizer outputs to one flat ID list."""
    if hasattr(value, "keys") and "input_ids" in value:
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = list(value[0])
    if not isinstance(value, list) or any(isinstance(item, (list, tuple)) for item in value):
        raise TypeError(f"Expected one flat token-ID sequence, got {type(value).__name__}")
    return [int(token_id) for token_id in value]


def chat_template(tokenizer, messages, add_generation_prompt: bool) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "return_tensors": None,
    }
    try:
        result = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        return token_id_list(result)
    except TypeError:
        try:
            result = tokenizer.apply_chat_template(messages, **kwargs)
            return token_id_list(result)
        except (AttributeError, ValueError):
            pass
    except (AttributeError, ValueError):
        pass

    # Base tokenizers do not always ship a chat template. Keep a deterministic
    # fallback whose prompt rendering is an exact prefix of the full rendering.
    role_names = {"system": "System", "user": "User", "assistant": "Assistant"}
    text = "".join(
        f"{role_names.get(message['role'], message['role'].title())}:\n{message['content']}\n"
        for message in messages
    )
    if add_generation_prompt:
        text += "Assistant:\n"
    elif messages and messages[-1]["role"] == "assistant":
        text += tokenizer.eos_token or ""
    return token_id_list(tokenizer(text, add_special_tokens=True))


def truncate_text(tokenizer, text: str, token_limit: int) -> str:
    if token_limit <= 0:
        return ""
    ids = token_id_list(tokenizer(
        text, add_special_tokens=False, truncation=True, max_length=token_limit
    ))
    return tokenizer.decode(ids, skip_special_tokens=True)


def compact_set_prompt(
    record: dict[str, Any], tokenizer, max_seq_length: int,
    completion_token_reserve: int = 128,
) -> list[dict[str, str]]:
    candidates = record["candidate_records"]
    # Use a fixed completion reserve rather than the gold completion length, so
    # prompt truncation cannot reveal target-set size during training/evaluation.
    content_budget = max(256, max_seq_length - completion_token_reserve - 768)
    query_budget = min(512, max(128, content_budget // 5))
    candidate_budget = max(24, (content_budget - query_budget) // max(1, len(candidates)))
    query_text = truncate_text(tokenizer, record["query_text"], query_budget)
    rows = [
        f"Query issue {record['query_uid']}:\n{query_text}",
        f"Relation: {record['relation']}",
        f"Relation definition: {record['relation_definition']}",
        "Candidates:",
    ]
    for candidate in candidates:
        text = truncate_text(tokenizer, candidate["text"], candidate_budget)
        rows.append(f"{candidate['label']}: {text}")
    return [record["prompt"][0], {"role": "user", "content": "\n".join(rows)}]


def encode_record(record: dict[str, Any], tokenizer, task: str, max_seq_length: int) -> dict[str, list[int]]:
    completion = record["completion"]
    compact_limit = max_seq_length
    for _ in range(6):
        prompt_messages = (
            compact_set_prompt(record, tokenizer, compact_limit)
            if task == "set_retrieval" else record["prompt"]
        )
        prompt_ids = chat_template(tokenizer, prompt_messages, add_generation_prompt=True)
        full_ids = chat_template(tokenizer, prompt_messages + completion, add_generation_prompt=False)
        if full_ids[: len(prompt_ids)] == prompt_ids:
            completion_ids = full_ids[len(prompt_ids):]
        else:
            completion_ids = token_id_list(tokenizer(
                completion[0]["content"] + (tokenizer.eos_token or ""), add_special_tokens=False
            ))
        overflow = len(prompt_ids) + len(completion_ids) - max_seq_length
        if task != "set_retrieval" or overflow <= 0:
            break
        compact_limit = max(1024, compact_limit - overflow - 128)
    if len(completion_ids) >= max_seq_length:
        raise ValueError("Completion alone exceeds --max-seq-length")
    if task == "set_retrieval" and len(prompt_ids) + len(completion_ids) > max_seq_length:
        raise ValueError(
            f"Could not preserve every set candidate within {max_seq_length} tokens for {record['query_id']}"
        )
    if task == "pointwise":
        prompt_ids = prompt_ids[: max_seq_length - len(completion_ids)]
    input_ids = prompt_ids + completion_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + completion_ids.copy(),
    }


def keep_training_record(record: dict[str, Any], args, task: str) -> bool:
    if task == "pointwise" and int(record["label"]) == 0:
        candidate = record.get("candidate", {})
        key = f"{args.seed}:{record['query_id']}:{candidate.get('issue_uid', candidate.get('label', ''))}"
        return stable_fraction(key) < args.negative_keep_probability
    if task == "set_retrieval" and not record.get("gold_candidate_labels"):
        return stable_fraction(f"{args.seed}:{record['query_id']}") < args.empty_keep_probability
    return True


def exact_pointwise_selector(args, repository: str, manifest):
    """Keep all positives and an exact, deterministic fraction of negatives."""
    total = int(manifest["counts"]["point_train"])
    positives = int(manifest["eligible_gold"]["train"])
    negative_total = total - positives
    negative_target = round(args.negative_keep_probability * negative_total)
    if negative_target >= negative_total:
        return lambda row: True
    phase_key = f"{args.seed}:{repository}:pointwise-negative-phase"
    accumulator = int(hashlib.sha256(phase_key.encode("utf-8")).hexdigest()[:16], 16) % negative_total

    def select(row):
        nonlocal accumulator
        if int(row["label"]) == 1:
            return True
        accumulator += negative_target
        if accumulator >= negative_total:
            accumulator -= negative_total
            return True
        return False

    return select


def load_streams(
    args, task, tokenizer, paths, manifests,
    load_dataset, interleave_datasets, concatenate_datasets,
):
    repositories = selected_repositories(args.dataset)

    def make_stream(repository: str, split: str):
        stream = load_dataset(
            "json", data_files=str(paths[repository][split]), split="train", streaming=True
        )
        if split == "train":
            if task == "pointwise" and args.training_mode == "exhaustive":
                stream = stream.filter(exact_pointwise_selector(args, repository, manifests[repository]))
            else:
                stream = stream.filter(lambda row: keep_training_record(row, args, task))
            stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)
        columns = list(stream.features) if stream.features is not None else None
        stream = stream.map(
            lambda row: encode_record(row, tokenizer, task, args.max_seq_length),
            remove_columns=columns,
        )
        return stream

    train_streams = [make_stream(repo, "train") for repo in repositories]
    valid_streams = [make_stream(repo, "validation") for repo in repositories]
    probabilities = sampling_probabilities(
        repositories, manifests, task, args.sampling, args.sampling_temperature
    )
    if len(repositories) == 1:
        train, valid = train_streams[0], valid_streams[0]
    elif args.training_mode == "exhaustive":
        # Concatenation visits every row exactly once. Probabilistic
        # interleaving can repeat exhausted repositories and is therefore not
        # suitable for a one-pass coverage claim.
        train = concatenate_datasets(train_streams)
        valid = interleave_datasets(
            valid_streams, probabilities=None, seed=args.seed,
            stopping_strategy="first_exhausted",
        )
        probabilities = None
    else:
        train = interleave_datasets(
            train_streams, probabilities=probabilities, seed=args.seed,
            stopping_strategy="all_exhausted",
        )
        valid = interleave_datasets(
            valid_streams, probabilities=None, seed=args.seed,
            stopping_strategy="first_exhausted",
        )
    return train, valid.take(args.max_eval_samples), probabilities


class CompletionOnlyCollator:
    def __init__(self, tokenizer, torch):
        self.tokenizer = tokenizer
        self.torch = torch

    def __call__(self, features):
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_mask, labels = [], [], []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.tokenizer.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": self.torch.tensor(input_ids, dtype=self.torch.long),
            "attention_mask": self.torch.tensor(attention_mask, dtype=self.torch.long),
            "labels": self.torch.tensor(labels, dtype=self.torch.long),
        }


def training_arguments(args, torch, transformers):
    values = {
        "output_dir": str(args.checkpoint_dir),
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_total_limit": args.save_total_limit,
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "tf32": torch.cuda.is_available(),
        "gradient_checkpointing": True,
        "dataloader_num_workers": args.dataloader_num_workers,
        "remove_unused_columns": False,
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "logging_first_step": True,
        "lr_scheduler_type": "cosine",
    }
    accepted = inspect.signature(transformers.TrainingArguments.__init__).parameters
    strategy_name = "eval_strategy" if "eval_strategy" in accepted else "evaluation_strategy"
    values[strategy_name] = "steps"
    if "warmup_ratio" in accepted:
        values["warmup_ratio"] = args.warmup_ratio
    elif "warmup_steps" in accepted:
        values["warmup_steps"] = max(1, round(args.max_steps * args.warmup_ratio))
    return transformers.TrainingArguments(**{key: value for key, value in values.items() if key in accepted})


def dry_run_report(args, task, paths, manifests):
    repositories = selected_repositories(args.dataset)
    probabilities = None if args.training_mode == "exhaustive" else sampling_probabilities(
        repositories, manifests, task, args.sampling, args.sampling_temperature
    )
    report = {
        "status": "dry-run",
        "task": task,
        "initialization": args.initialization,
        "dataset": args.dataset,
        "training_mode": args.training_mode,
        "records_per_pass": args.records_per_pass,
        "resolved_max_steps": args.max_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "data_version": args.data_version,
        "model_path": str(args.model_path),
        "cpt_adapter_path": str(args.cpt_adapter_path) if args.initialization == "cpt" else None,
        "output_dir": str(args.output_dir),
        "checkpoint_dir": str(args.checkpoint_dir),
        "files": {repo: {key: str(value) for key, value in repo_paths.items()} for repo, repo_paths in paths.items()},
        "counts": {repo: manifests[repo]["counts"] for repo in repositories},
        "candidate_recall": {repo: manifests[repo].get("candidate_recall") for repo in repositories},
        "sampling_probabilities": dict(zip(repositories, probabilities)) if probabilities else None,
        "warning": "v1_full natural pools are pilot inputs; use corrected v2 pools for final paper runs.",
    }
    print(json.dumps(report, indent=2))


def run(task: str) -> None:
    args = normalize_paths(build_parser(task).parse_args(), task)
    paths = dataset_paths(args, task)
    manifests = read_manifests(paths, task)
    args = resolve_training_plan(args, task, manifests)
    if not 0.0 <= args.negative_keep_probability <= 1.0:
        raise ValueError("--negative-keep-probability must be between 0 and 1")
    if not 0.0 <= args.empty_keep_probability <= 1.0:
        raise ValueError("--empty-keep-probability must be between 0 and 1")
    if args.max_steps <= 0:
        raise ValueError("Streaming training requires --max-steps greater than zero")
    if args.dry_run:
        dry_run_report(args, task, paths, manifests)
        return
    if args.training_mode == "exhaustive" and args.resume_from_checkpoint is None and args.auto_resume:
        checkpoint = newest_checkpoint(args.checkpoint_dir)
        if checkpoint is not None:
            args.resume_from_checkpoint = str(checkpoint)
            print(f"Auto-resuming from {checkpoint}", flush=True)
    if (
        args.output_dir.exists() and any(args.output_dir.iterdir())
        and not args.resume_from_checkpoint and not args.overwrite_output
    ):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    stack = import_training_stack()
    torch, transformers, load_dataset, interleave_datasets, concatenate_datasets, LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training = stack
    model, tokenizer = load_model(
        args, torch, transformers, LoraConfig, PeftModel, get_peft_model,
        prepare_model_for_kbit_training,
    )
    train, valid, probabilities = load_streams(
        args, task, tokenizer, paths, manifests,
        load_dataset, interleave_datasets, concatenate_datasets,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args) | {
        "task": task,
        "resolved_data_files": {
            repo: {key: str(value) for key, value in repo_paths.items()} for repo, repo_paths in paths.items()
        },
        "sampling_probabilities": probabilities,
        "training_mode": args.training_mode,
        "records_per_pass": args.records_per_pass,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
    }
    serialized_config = json.dumps(run_config, default=str, indent=2) + "\n"
    # Checkpoints own the in-progress configuration. When replacing an adapter,
    # leave the completed adapter directory untouched until training succeeds.
    (args.checkpoint_dir / "run_config.json").write_text(serialized_config, encoding="utf-8")
    if not args.overwrite_output or not (args.output_dir / "adapter_model.safetensors").is_file():
        (args.output_dir / "run_config.json").write_text(serialized_config, encoding="utf-8")
    trainer = transformers.Trainer(
        model=model,
        args=training_arguments(args, torch, transformers),
        train_dataset=train,
        eval_dataset=valid,
        data_collator=CompletionOnlyCollator(tokenizer, torch),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    (args.output_dir / "run_config.json").write_text(serialized_config, encoding="utf-8")
    metrics = trainer.evaluate()
    (args.output_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {task} adapter to {args.output_dir}")

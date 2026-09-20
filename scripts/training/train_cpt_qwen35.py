#!/usr/bin/env python3
"""Train a LoRA/QLoRA CPT adapter for Qwen3.5 on all prepared corpora."""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import sys

REPOSITORIES = ("redhat_v1", "apache_v1", "jira_v1", "mongodb_v1", "qt_v1", "mojang_v1")


def args():
    project = Path(__file__).resolve().parents[2]
    model_root = Path(os.environ.get("ILR_MODEL_ROOT", "~/scratch/llms_model/ilr_llms")).expanduser()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=project)
    p.add_argument("--model-path", type=Path, default=model_root / "base/Qwen3.5-9B-Base")
    p.add_argument("--output-dir", type=Path, default=model_root / "adapters/qwen3.5-9b-cpt-all-v1")
    p.add_argument("--checkpoint-dir", type=Path, default=model_root / "checkpoints/qwen3.5-9b-cpt-all-v1")
    p.add_argument("--method", choices=("lora", "qlora"), default="qlora")
    p.add_argument("--repositories", nargs="+", choices=REPOSITORIES, default=list(REPOSITORIES))
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--per-device-eval-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--dataloader-num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def files(root, repos):
    train, valid = [], []
    for repo in repos:
        d = root / "data/training/cpt" / repo
        train += sorted(str(x) for x in d.glob("train-*.jsonl"))
        valid += sorted(str(x) for x in d.glob("validation-*.jsonl"))
    if not train or not valid:
        raise FileNotFoundError("CPT JSONL shards are missing")
    return train, valid


def imports():
    try:
        import torch
        import transformers
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise SystemExit(
            "Install training dependencies first:\n"
            "  uv pip install torch transformers datasets peft accelerate\n"
            "  uv pip install bitsandbytes  # required for --method qlora\n"
            f"Missing package: {exc}"
        ) from exc
    return torch, transformers, load_dataset, LoraConfig, get_peft_model


def load_data(load_dataset, root, repos):
    train_files, valid_files = files(root, repos)
    train = load_dataset("json", data_files={"train": train_files}, split="train")
    valid = load_dataset("json", data_files={"validation": valid_files}, split="validation")
    for ds in (train, valid):
        if ds.column_names != ["text"]:
            raise ValueError(f"Expected only text column, got {ds.column_names}")
    return train, valid, train_files, valid_files


def load_model(args, torch, transformers, LoraConfig, get_peft_model):
    if not args.model_path.is_dir() or not (args.model_path / "config.json").exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Transformers 5.x uses `dtype`; keep a compatibility fallback below for
    # older model implementations that still only accept `torch_dtype`.
    model_kwargs = {"dtype": torch.bfloat16, "trust_remote_code": args.trust_remote_code}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}
    if args.method == "qlora":
        if not torch.cuda.is_available():
            raise SystemExit("QLoRA requires CUDA")
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    try:
        cls = transformers.AutoModelForImageTextToText
    except AttributeError:
        cls = transformers.AutoModelForCausalLM
    try:
        model = cls.from_pretrained(args.model_path, **model_kwargs)
    except TypeError as first_error:
        legacy_kwargs = dict(model_kwargs)
        legacy_kwargs["torch_dtype"] = legacy_kwargs.pop("dtype")
        try:
            model = cls.from_pretrained(args.model_path, **legacy_kwargs)
        except (ValueError, TypeError, OSError):
            raise first_error
    except (ValueError, OSError):
        if cls is transformers.AutoModelForCausalLM:
            raise
        model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    if args.method == "qlora":
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.enable_input_require_grads()
    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules="all-linear")
    return get_peft_model(model, lora), tokenizer


def main():
    a = args()
    root = a.project_root.resolve()
    a.model_path, a.output_dir, a.checkpoint_dir = (x.expanduser().resolve() for x in (a.model_path, a.output_dir, a.checkpoint_dir))
    train_files, valid_files = files(root, a.repositories)
    if a.dry_run:
        print(json.dumps({"status": "dry-run", "model_exists": a.model_path.is_dir(), "train_shards": len(train_files), "validation_shards": len(valid_files), "model_path": str(a.model_path), "output_dir": str(a.output_dir)}, indent=2))
        return
    if not a.model_path.is_dir():
        raise FileNotFoundError(f"Model not found: {a.model_path}")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "run_config.json").write_text(json.dumps(vars(a), default=str, indent=2) + "\n")
    torch, transformers, load_dataset, LoraConfig, get_peft_model = imports()
    train, valid, _, _ = load_data(load_dataset, root, a.repositories)
    model, tokenizer = load_model(a, torch, transformers, LoraConfig, get_peft_model)

    def tokenize(batch):
        texts = [text + (tokenizer.eos_token or "") for text in batch["text"]]
        return tokenizer(texts, add_special_tokens=True, truncation=False)

    def pack(batch):
        joined = {key: sum(batch[key], []) for key in batch}
        usable = (len(joined["input_ids"]) // a.max_seq_length) * a.max_seq_length
        return {key: [values[i:i + a.max_seq_length] for i in range(0, usable, a.max_seq_length)] for key, values in joined.items()}

    train = train.map(tokenize, batched=True, remove_columns=["text"], desc="Tokenizing train corpus")
    valid = valid.map(tokenize, batched=True, remove_columns=["text"], desc="Tokenizing validation corpus")
    train = train.map(pack, batched=True, desc="Packing train tokens")
    valid = valid.map(pack, batched=True, desc="Packing validation tokens")
    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    argument_values = {
        "output_dir": str(a.checkpoint_dir), "per_device_train_batch_size": a.per_device_train_batch_size,
        "per_device_eval_batch_size": a.per_device_eval_batch_size, "gradient_accumulation_steps": a.gradient_accumulation_steps,
        "learning_rate": a.learning_rate, "num_train_epochs": a.num_train_epochs, "max_steps": a.max_steps,
        "logging_steps": a.logging_steps, "save_steps": a.save_steps, "eval_steps": a.eval_steps,
        "save_strategy": "steps", "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "gradient_checkpointing": True, "dataloader_num_workers": a.dataloader_num_workers,
        "remove_unused_columns": False, "report_to": "none", "seed": a.seed, "logging_first_step": True,
    }
    # Transformers renamed evaluation_strategy to eval_strategy in newer releases
    # and some cluster environments expose neither warmup_ratio nor eval_strategy.
    accepted = inspect.signature(transformers.TrainingArguments.__init__).parameters
    argument_values["eval_strategy" if "eval_strategy" in accepted else "evaluation_strategy"] = "steps"
    if "warmup_ratio" in accepted:
        argument_values["warmup_ratio"] = a.warmup_ratio
    elif "warmup_steps" in accepted:
        import math
        steps_per_epoch = math.ceil(len(train) / (a.per_device_train_batch_size * a.gradient_accumulation_steps))
        total_steps = a.max_steps if a.max_steps > 0 else math.ceil(steps_per_epoch * a.num_train_epochs)
        argument_values["warmup_steps"] = max(1, round(total_steps * a.warmup_ratio))
    ta = transformers.TrainingArguments(**{k: v for k, v in argument_values.items() if k in accepted})
    trainer = transformers.Trainer(model=model, args=ta, train_dataset=train, eval_dataset=valid, data_collator=collator)
    trainer.train(resume_from_checkpoint=a.resume_from_checkpoint)
    trainer.save_model(str(a.output_dir))
    tokenizer.save_pretrained(str(a.output_dir))
    print(f"Saved CPT adapter to {a.output_dir}")


if __name__ == "__main__":
    main()

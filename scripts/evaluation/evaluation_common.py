#!/usr/bin/env python3
"""Shared loading, data, parsing, and metrics for SFT evaluation."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, TextIO


REPOSITORIES = ("apache", "jira", "redhat", "mongodb", "qt", "mojang")
TASK_SLUG = {"set_retrieval": "set", "pointwise": "pointwise"}
RANK_CUTOFFS = (1, 5, 10, 20)


def build_parser(task: str) -> argparse.ArgumentParser:
    project = Path(__file__).resolve().parents[2]
    model_root = Path(os.environ.get("ILR_MODEL_ROOT", "~/scratch/llms_model/ilr_llms")).expanduser()
    parser = argparse.ArgumentParser(description=f"Evaluate the {task} issue-link SFT model.")
    parser.add_argument("--project-root", type=Path, default=project)
    parser.add_argument("--model-root", type=Path, default=model_root)
    parser.add_argument("--model-path", type=Path, default=None, help="Base model directory.")
    parser.add_argument("--adapter-path", type=Path, default=None, help="Override the inferred SFT adapter.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--initialization", choices=("base", "cpt"), required=True)
    parser.add_argument(
        "--model-dataset", "--dataset", dest="model_dataset", type=str.lower,
        choices=(*REPOSITORIES, "all"), required=True,
        help="Repository mixture used to train the adapter.",
    )
    parser.add_argument(
        "--eval-dataset", type=str.lower, choices=(*REPOSITORIES, "all"), default=None,
        help="Repository evaluated; defaults to --model-dataset.",
    )
    parser.add_argument("--data-version", default="v1_full")
    parser.add_argument("--adapter-suffix", default="", help="Training run suffix, e.g. smoke-fix1.")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max-seq-length", type=int, default=4096 if task == "set_retrieval" else 2048)
    parser.add_argument("--max-samples", type=int, default=0, help="Maximum queries (0 evaluates all).")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save-predictions", action=argparse.BooleanOptionalAction, default=True,
        help="Write gzip-compressed per-example predictions (default: true).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace, task: str) -> argparse.Namespace:
    args.project_root = args.project_root.expanduser().resolve()
    args.model_root = args.model_root.expanduser().resolve()
    args.model_path = (args.model_path or args.model_root / "base/Qwen3.5-9B-Base").expanduser().resolve()
    args.eval_dataset = args.eval_dataset or args.model_dataset
    version = args.data_version.replace("/", "-").replace("_", "-")
    adapter_name = f"qwen3.5-9b-{args.initialization}-sft-{TASK_SLUG[task]}-{args.model_dataset}-{version}"
    if args.adapter_suffix:
        adapter_name += "-" + args.adapter_suffix.replace("/", "-").replace("_", "-")
    args.adapter_path = (args.adapter_path or args.model_root / "adapters" / adapter_name).expanduser().resolve()
    shard = f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}" if args.num_shards > 1 else "full"
    args.output_dir = (
        args.output_dir
        or args.project_root / "experiment_results/evaluation" / adapter_name
        / f"{args.eval_dataset}-{args.data_version}" / args.split / shard
    ).expanduser().resolve()
    if args.max_seq_length <= 0 or args.max_samples < 0:
        raise ValueError("Sequence length must be positive and max samples cannot be negative")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require --num-shards > 0 and 0 <= --shard-index < --num-shards")
    return args


def selected_repositories(dataset: str) -> tuple[str, ...]:
    return REPOSITORIES if dataset == "all" else (dataset,)


def evaluation_paths(args: argparse.Namespace, task: str) -> dict[str, Path]:
    root = args.project_root / "data/training/sft" / task
    paths = {
        repository: root / f"{repository}_{args.data_version}" / f"{args.split}.jsonl"
        for repository in selected_repositories(args.eval_dataset)
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing evaluation datasets:\n  " + "\n  ".join(missing))
    return paths


def set_gold_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.project_root / "data/training/sft/set_retrieval"
    paths = {
        repository: root / f"{repository}_{args.data_version}" / f"{args.split}.jsonl"
        for repository in selected_repositories(args.eval_dataset)
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Pointwise ranking requires set-retrieval gold files:\n  " + "\n  ".join(missing))
    return paths


def repository_from_record(record: dict[str, Any]) -> str:
    uid = str(record.get("query_uid", ""))
    return uid.split(":", 1)[0].lower() if ":" in uid else "unknown"


def query_in_shard(query_id: str, num_shards: int, shard_index: int) -> bool:
    value = int(hashlib.sha256(query_id.encode("utf-8")).hexdigest()[:16], 16)
    return value % num_shards == shard_index


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc


def import_inference_stack():
    try:
        import torch
        import transformers
        from peft import PeftModel
    except ImportError as exc:
        raise SystemExit(f"Evaluation dependency is missing: {exc}") from exc
    return torch, transformers, PeftModel


def load_model_and_tokenizer(args: argparse.Namespace):
    torch, transformers, PeftModel = import_inference_stack()
    if not (args.model_path / "config.json").is_file():
        raise FileNotFoundError(f"Base model is missing: {args.model_path}")
    if not (args.adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"SFT adapter is missing: {args.adapter_path}")
    tokenizer_path = args.adapter_path if (args.adapter_path / "tokenizer_config.json").is_file() else args.model_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=args.trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    kwargs: dict[str, Any] = {"dtype": torch.bfloat16, "trust_remote_code": args.trust_remote_code}
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
    if args.load_in_4bit:
        if not torch.cuda.is_available():
            raise SystemExit("--load-in-4bit requires CUDA")
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    model_class = getattr(transformers, "AutoModelForImageTextToText", transformers.AutoModelForCausalLM)
    try:
        model = model_class.from_pretrained(args.model_path, **kwargs)
    except (ValueError, OSError):
        model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = model_class.from_pretrained(args.model_path, **kwargs)
    model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    return torch, model, tokenizer


def model_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except AttributeError:
        return next(model.parameters()).device


def prepare_output(args: argparse.Namespace, task: str, paths: dict[str, Path]) -> TextIO | None:
    metrics_path = args.output_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"Evaluation already exists: {metrics_path}; pass --overwrite to replace it")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "task": task,
        "resolved_data_files": {repository: str(path) for repository, path in paths.items()},
        "started_unix": time.time(),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, default=str, indent=2) + "\n", encoding="utf-8"
    )
    if not args.save_predictions:
        return None
    return gzip.open(args.output_dir / "predictions.jsonl.gz", "wt", encoding="utf-8")


def write_prediction(handle: TextIO | None, value: dict[str, Any]) -> None:
    if handle is not None:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def save_metrics(args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_tail_output(text: str) -> tuple[list[str], dict[str, Any]]:
    """Extract `tails` from the first JSON object and retain parse diagnostics."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1] if len(lines) >= 3 and lines[-1].strip() == "```" else lines[1:])
    start = stripped.find("{")
    diagnostics: dict[str, Any] = {"parse_ok": False, "parse_error": None}
    if start < 0:
        diagnostics["parse_error"] = "no_json_object"
        return [], diagnostics
    try:
        value, _ = json.JSONDecoder().raw_decode(stripped[start:])
    except (json.JSONDecodeError, TypeError) as exc:
        diagnostics["parse_error"] = f"invalid_json: {exc}"
        return [], diagnostics
    tails = value.get("tails") if isinstance(value, dict) else None
    if not isinstance(tails, list) or any(not isinstance(item, str) for item in tails):
        diagnostics["parse_error"] = "tails_must_be_a_string_list"
        return [], diagnostics
    diagnostics["parse_ok"] = True
    return [item.strip() for item in tails if item.strip()], diagnostics


def deduplicate(items: Iterable[str]) -> tuple[list[str], list[str]]:
    unique, duplicates, seen = [], [], set()
    for item in items:
        if item in seen:
            duplicates.append(item)
        else:
            unique.append(item)
            seen.add(item)
    return unique, duplicates


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def f_score(tp: int, fp: int, fn: int, beta: float = 1.0) -> float:
    beta2 = beta * beta
    return safe_div((1 + beta2) * tp, (1 + beta2) * tp + beta2 * fn + fp)


def average_precision(ranked: list[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    found, total = set(), 0.0
    for rank, item in enumerate(ranked, 1):
        if item in gold and item not in found:
            found.add(item)
            total += len(found) / rank
    return total / len(gold)


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    return next((1.0 / rank for rank, item in enumerate(ranked, 1) if item in gold), 0.0)


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(ranked[:k], 1) if item in gold)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(gold)) + 1))
    return safe_div(dcg, ideal)


class RetrievalMetrics:
    """Aggregate set and ranking metrics without retaining query records."""

    def __init__(self) -> None:
        self.n = 0
        self.candidate_total = self.full_gold = self.pool_gold = self.candidate_hits = 0
        self.predicted = self.tp_full = self.tp_pool = 0
        self.full_exact = self.pool_exact = self.empty_gold = self.empty_correct = 0
        self.macro = defaultdict(float)
        self.cutoff = {k: defaultdict(float) for k in RANK_CUTOFFS}

    def update(
        self, ranked: list[str], predicted: set[str], full_gold: set[str],
        pool_gold: set[str], candidate_set: set[str],
    ) -> None:
        self.n += 1
        self.candidate_total += len(candidate_set)
        self.full_gold += len(full_gold)
        self.pool_gold += len(pool_gold)
        self.candidate_hits += len(candidate_set & full_gold)
        self.macro["candidate_recall"] += safe_div(len(candidate_set & full_gold), len(full_gold))
        tp_full = len(predicted & full_gold)
        tp_pool = len(predicted & pool_gold)
        self.predicted += len(predicted)
        self.tp_full += tp_full
        self.tp_pool += tp_pool
        self.full_exact += predicted == full_gold
        self.pool_exact += predicted == pool_gold
        if not full_gold:
            self.empty_gold += 1
            self.empty_correct += not predicted
        for name, gold, tp in (("full", full_gold, tp_full), ("pool", pool_gold, tp_pool)):
            fp, fn = len(predicted - gold), len(gold - predicted)
            self.macro[f"{name}_precision"] += safe_div(tp, tp + fp)
            self.macro[f"{name}_recall"] += safe_div(tp, tp + fn)
            self.macro[f"{name}_f1"] += f_score(tp, fp, fn)
            self.macro[f"{name}_f2"] += f_score(tp, fp, fn, beta=2.0)
        self.macro["map"] += average_precision(ranked, full_gold)
        self.macro["mrr"] += reciprocal_rank(ranked, full_gold)
        for k in RANK_CUTOFFS:
            top = set(ranked[:k])
            hits = len(top & full_gold)
            self.cutoff[k]["recall"] += safe_div(hits, len(full_gold))
            self.cutoff[k]["hits"] += float(hits > 0)
            self.cutoff[k]["ndcg"] += ndcg_at_k(ranked, full_gold, k)

    def finalize(self) -> dict[str, Any]:
        full_fp = self.predicted - self.tp_full
        full_fn = self.full_gold - self.tp_full
        pool_fp = self.predicted - self.tp_pool
        pool_fn = self.pool_gold - self.tp_pool
        return {
            "queries": self.n,
            "candidates": self.candidate_total,
            "gold_tails": self.full_gold,
            "gold_tails_in_pool": self.pool_gold,
            "candidate_recall_micro": safe_div(self.candidate_hits, self.full_gold),
            "predicted_tails": self.predicted,
            "end_to_end_micro": {
                "precision": safe_div(self.tp_full, self.predicted),
                "recall": safe_div(self.tp_full, self.full_gold),
                "f1": f_score(self.tp_full, full_fp, full_fn),
                "f2": f_score(self.tp_full, full_fp, full_fn, beta=2.0),
                "exact_set_accuracy": safe_div(self.full_exact, self.n),
            },
            "candidate_pool_micro": {
                "precision": safe_div(self.tp_pool, self.predicted),
                "recall": safe_div(self.tp_pool, self.pool_gold),
                "f1": f_score(self.tp_pool, pool_fp, pool_fn),
                "f2": f_score(self.tp_pool, pool_fp, pool_fn, beta=2.0),
                "exact_set_accuracy": safe_div(self.pool_exact, self.n),
            },
            "macro": {key: safe_div(value, self.n) for key, value in sorted(self.macro.items())},
            "ranking": {
                f"recall@{k}": safe_div(values["recall"], self.n)
                for k, values in self.cutoff.items()
            } | {
                f"hits@{k}": safe_div(values["hits"], self.n)
                for k, values in self.cutoff.items()
            } | {
                f"ndcg@{k}": safe_div(values["ndcg"], self.n)
                for k, values in self.cutoff.items()
            },
            "empty_gold": {
                "queries": self.empty_gold,
                "empty_prediction_accuracy": safe_div(self.empty_correct, self.empty_gold),
            },
        }


def binary_metrics(scores: list[float], labels: list[int], threshold: float) -> dict[str, Any]:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels):
        predicted = score >= threshold
        tp += predicted and label == 1
        fp += predicted and label == 0
        tn += (not predicted) and label == 0
        fn += (not predicted) and label == 1
    return {
        "examples": len(labels), "positive_labels": sum(labels), "threshold": threshold,
        "accuracy": safe_div(tp + tn, len(labels)), "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn), "f1": f_score(tp, fp, fn),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "auroc": roc_auc(scores, labels), "average_precision": binary_average_precision(scores, labels),
    }


def roc_auc(scores: list[float], labels: list[int]) -> float:
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        return 0.0
    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = fp = previous_tp = previous_fp = area = 0.0
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        group_pos = group_neg = 0
        while index < len(pairs) and pairs[index][0] == score:
            group_pos += pairs[index][1]
            group_neg += 1 - pairs[index][1]
            index += 1
        tp += group_pos
        fp += group_neg
        area += (fp - previous_fp) * (tp + previous_tp) / 2.0
        previous_tp, previous_fp = tp, fp
    return area / (positives * negatives)


def binary_average_precision(scores: list[float], labels: list[int]) -> float:
    positives = sum(labels)
    if not positives:
        return 0.0
    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = fp = 0
    total = 0.0
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        previous_tp = tp
        while index < len(pairs) and pairs[index][0] == score:
            tp += pairs[index][1]
            fp += 1 - pairs[index][1]
            index += 1
        total += ((tp - previous_tp) / positives) * safe_div(tp, tp + fp)
    return total


def best_f1_threshold(scores: list[float], labels: list[int]) -> tuple[float, float]:
    """Select a threshold only on validation data; ties favor the higher threshold."""
    if not scores:
        return 0.5, 0.0
    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    positives, tp, fp, best_threshold, best_f1 = sum(labels), 0, 0, pairs[0][0], 0.0
    index = 0
    while index < len(pairs):
        threshold = pairs[index][0]
        while index < len(pairs) and pairs[index][0] == threshold:
            tp += pairs[index][1]
            fp += 1 - pairs[index][1]
            index += 1
        score = f_score(tp, fp, positives - tp)
        if score > best_f1:
            best_threshold, best_f1 = threshold, score
    return float(best_threshold), float(best_f1)


def seed_everything(seed: int, torch=None) -> None:
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

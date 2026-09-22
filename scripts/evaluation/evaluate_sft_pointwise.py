#!/usr/bin/env python3
"""Score candidate validity and evaluate pointwise SFT as classification and retrieval."""
from __future__ import annotations

import itertools
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from sft_common import chat_template, token_id_list  # noqa: E402

from evaluation_common import (  # noqa: E402
    RetrievalMetrics, best_f1_threshold, binary_metrics, build_parser,
    evaluation_paths, iter_jsonl, load_model_and_tokenizer, model_device,
    normalize_args, prepare_output, query_in_shard, repository_from_record,
    save_metrics, seed_everything, set_gold_paths, write_prediction,
)


TRUE_RESPONSE = [{"role": "assistant", "content": '{"valid":true}'}]
FALSE_RESPONSE = [{"role": "assistant", "content": '{"valid":false}'}]


def completion_ids(tokenizer, prompt_messages, response, prompt_ids: list[int]) -> list[int]:
    full_ids = chat_template(tokenizer, prompt_messages + response, add_generation_prompt=False)
    if full_ids[: len(prompt_ids)] == prompt_ids:
        return full_ids[len(prompt_ids):]
    return token_id_list(tokenizer(
        response[0]["content"] + (tokenizer.eos_token or ""), add_special_tokens=False
    ))


def score_records(records, tokenizer, model, torch, device, max_seq_length: int, normalization: str):
    sequences, metadata = [], []
    for record_index, record in enumerate(records):
        prompt_ids = chat_template(tokenizer, record["prompt"], add_generation_prompt=True)
        answer_ids = [
            completion_ids(tokenizer, record["prompt"], response, prompt_ids)
            for response in (TRUE_RESPONSE, FALSE_RESPONSE)
        ]
        for answer_index, completion in enumerate(answer_ids):
            if len(completion) >= max_seq_length:
                raise ValueError("Pointwise completion exceeds --max-seq-length")
            # This matches SFT preprocessing: preserve the beginning of the
            # prompt and every supervised completion token.
            truncated_prompt = prompt_ids[: max_seq_length - len(completion)]
            metadata.append((record_index, answer_index, len(truncated_prompt), completion))
            sequences.append(truncated_prompt + completion)
    width = max(map(len, sequences))
    padded = [seq + [tokenizer.pad_token_id] * (width - len(seq)) for seq in sequences]
    masks = [[1] * len(seq) + [0] * (width - len(seq)) for seq in sequences]
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)
    values = [[0.0, 0.0] for _ in records]
    for row, (record_index, answer_index, start, completion) in enumerate(metadata):
        target = torch.tensor(completion, dtype=torch.long, device=device)
        positions = torch.arange(start - 1, start + len(completion) - 1, device=device)
        value = log_probs[row, positions, target].sum()
        if normalization == "mean":
            value = value / len(completion)
        values[record_index][answer_index] = float(value.item())
    results = []
    for valid_value, invalid_value in values:
        high = max(valid_value, invalid_value)
        probability = math.exp(valid_value - high) / (
            math.exp(valid_value - high) + math.exp(invalid_value - high)
        )
        results.append((probability, valid_value, invalid_value))
    return results


def load_gold(paths: dict[str, Path]) -> dict[str, set[str]]:
    result = {}
    for path in paths.values():
        for record in iter_jsonl(path):
            result[record["query_id"]] = set(record.get("gold_tail_uids", []))
    return result


def grouped_records(path: Path):
    return itertools.groupby(iter_jsonl(path), key=lambda row: row["query_id"])


def main() -> None:
    parser = build_parser("pointwise")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--select-threshold", action="store_true",
        help="Select the maximum-F1 threshold; allowed only on validation.",
    )
    parser.add_argument("--score-normalization", choices=("sum", "mean"), default="sum")
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Candidates per forward pass; each creates valid and invalid sequences.",
    )
    args = normalize_args(parser.parse_args(), "pointwise")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between zero and one")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.select_threshold and args.split != "validation":
        raise ValueError("--select-threshold is validation-only; pass the frozen value to test with --threshold")
    paths = evaluation_paths(args, "pointwise")
    gold_paths = set_gold_paths(args)
    if args.dry_run:
        print(json.dumps({
            "task": "pointwise", "adapter_path": str(args.adapter_path),
            "output_dir": str(args.output_dir), "data_files": {k: str(v) for k, v in paths.items()},
            "gold_files": {k: str(v) for k, v in gold_paths.items()},
        }, indent=2))
        return

    prediction_handle = prepare_output(args, "pointwise", paths)
    gold_by_query = load_gold(gold_paths)
    torch, model, tokenizer = load_model_and_tokenizer(args)
    seed_everything(args.seed, torch)
    device = model_device(model)
    query_results = []
    classification = defaultdict(lambda: {"scores": [], "labels": []})
    processed_queries = processed_candidates = 0
    started = time.perf_counter()
    try:
        for repository, path in paths.items():
            for query_id, rows_iterator in grouped_records(path):
                rows = list(rows_iterator)
                if not query_in_shard(query_id, args.num_shards, args.shard_index):
                    continue
                if args.max_samples and processed_queries >= args.max_samples:
                    break
                scored = []
                for batch_start in range(0, len(rows), args.batch_size):
                    batch = rows[batch_start:batch_start + args.batch_size]
                    before = time.perf_counter()
                    batch_scores = score_records(
                        batch, tokenizer, model, torch, device,
                        args.max_seq_length, args.score_normalization,
                    )
                    per_candidate_latency = (time.perf_counter() - before) / len(batch)
                    for record, (probability, valid_logp, invalid_logp) in zip(batch, batch_scores):
                        candidate = record["candidate"]
                        label = int(record["label"])
                        scored.append((probability, candidate["rank"], candidate["issue_uid"], record, per_candidate_latency))
                        for key in ("overall", repository_from_record(record)):
                            classification[key]["scores"].append(probability)
                            classification[key]["labels"].append(label)
                        write_prediction(prediction_handle, {
                            "repository": repository, "query_id": query_id,
                            "query_uid": record["query_uid"], "relation": record["relation"],
                            "candidate_label": candidate["label"], "candidate_uid": candidate["issue_uid"],
                            "retriever_rank": candidate["rank"], "gold_label": label,
                            "valid_probability": probability, "valid_log_likelihood": valid_logp,
                            "invalid_log_likelihood": invalid_logp, "latency_seconds": per_candidate_latency,
                        })
                        processed_candidates += 1
                scored.sort(key=lambda item: (-item[0], item[1], item[2]))
                query_results.append((repository, query_id, scored))
                processed_queries += 1
                if processed_queries % 100 == 0:
                    print(
                        f"Evaluated {processed_queries} queries / {processed_candidates} candidates "
                        f"in {time.perf_counter() - started:.1f}s", flush=True,
                    )
            if args.max_samples and processed_queries >= args.max_samples:
                break
    finally:
        if prediction_handle is not None:
            prediction_handle.close()

    threshold = args.threshold
    selection = None
    if args.select_threshold:
        threshold, best_f1 = best_f1_threshold(
            classification["overall"]["scores"], classification["overall"]["labels"]
        )
        selection = {"selected_threshold": threshold, "validation_f1": best_f1}
        (args.output_dir / "selected_threshold.json").write_text(
            json.dumps(selection, indent=2) + "\n", encoding="utf-8"
        )

    retrieval = defaultdict(RetrievalMetrics)
    latency = defaultdict(float)
    for repository, query_id, scored in query_results:
        ranked = [item[2] for item in scored]
        predicted = {item[2] for item in scored if item[0] >= threshold}
        full_gold = gold_by_query.get(query_id, set())
        candidate_set = set(ranked)
        pool_gold = full_gold & candidate_set
        repo_key = repository_from_record(scored[0][3]) if scored else repository
        for key in ("overall", repo_key):
            retrieval[key].update(ranked, predicted, full_gold, pool_gold, candidate_set)
            latency[key] += sum(item[4] for item in scored)

    result = {}
    for key, aggregate in sorted(retrieval.items()):
        values = aggregate.finalize()
        values["classification"] = binary_metrics(
            classification[key]["scores"], classification[key]["labels"], threshold
        )
        count = values["classification"]["examples"]
        values["scoring"] = {
            "normalization": args.score_normalization,
            "mean_candidate_latency_seconds": latency[key] / count if count else 0.0,
        }
        result[key] = values
    metrics = {
        "task": "pointwise", "split": args.split, "model_dataset": args.model_dataset,
        "eval_dataset": args.eval_dataset, "adapter_path": str(args.adapter_path),
        "elapsed_seconds": time.perf_counter() - started, "threshold_selection": selection,
        "results": result,
        "methodology": {
            "candidate_score": "softmax of conditional log likelihoods for fixed valid/invalid JSON",
            "ranking_ties": "original retriever rank then candidate UID",
            "gold_scope": "gold_tail_uids loaded from the matching set-retrieval split",
        },
    }
    save_metrics(args, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

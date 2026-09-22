#!/usr/bin/env python3
"""Generate tail sets and evaluate a relation-conditioned SFT adapter."""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from sft_common import chat_template, compact_set_prompt  # noqa: E402

from evaluation_common import (  # noqa: E402
    RetrievalMetrics, build_parser, deduplicate, evaluation_paths, iter_jsonl,
    load_model_and_tokenizer, model_device, normalize_args, parse_tail_output,
    prepare_output, query_in_shard, repository_from_record, save_metrics,
    seed_everything, write_prediction,
)


def prompt_ids_for_record(record, tokenizer, max_seq_length: int, max_new_tokens: int) -> list[int]:
    prompt_limit = max_seq_length
    for _ in range(8):
        messages = compact_set_prompt(
            record, tokenizer, prompt_limit, completion_token_reserve=max_new_tokens
        )
        prompt_ids = chat_template(tokenizer, messages, add_generation_prompt=True)
        if len(prompt_ids) + max_new_tokens <= max_seq_length:
            return prompt_ids
        prompt_limit = max(1024, prompt_limit - (len(prompt_ids) + max_new_tokens - max_seq_length) - 128)
    raise ValueError(
        f"Could not retain every candidate within {max_seq_length} tokens for {record['query_id']}"
    )


def main() -> None:
    parser = build_parser("set_retrieval")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = normalize_args(parser.parse_args(), "set_retrieval")
    if args.max_new_tokens <= 0 or args.max_new_tokens >= args.max_seq_length:
        raise ValueError("--max-new-tokens must be positive and below --max-seq-length")
    paths = evaluation_paths(args, "set_retrieval")
    if args.dry_run:
        print(json.dumps({
            "task": "set_retrieval", "adapter_path": str(args.adapter_path),
            "output_dir": str(args.output_dir), "data_files": {k: str(v) for k, v in paths.items()},
        }, indent=2))
        return

    prediction_handle = prepare_output(args, "set_retrieval", paths)
    torch, model, tokenizer = load_model_and_tokenizer(args)
    seed_everything(args.seed, torch)
    device = model_device(model)
    aggregates = defaultdict(RetrievalMetrics)
    diagnostics = defaultdict(lambda: defaultdict(float))
    processed = 0
    started = time.perf_counter()
    try:
        for repository, path in paths.items():
            for record in iter_jsonl(path):
                if not query_in_shard(record["query_id"], args.num_shards, args.shard_index):
                    continue
                if args.max_samples and processed >= args.max_samples:
                    break
                prompt_ids = prompt_ids_for_record(
                    record, tokenizer, args.max_seq_length, args.max_new_tokens
                )
                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids)
                before = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=input_ids, attention_mask=attention_mask,
                        do_sample=False, max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                latency = time.perf_counter() - before
                output_text = tokenizer.decode(generated[0, len(prompt_ids):], skip_special_tokens=True)
                parsed, parse_info = parse_tail_output(output_text)
                unique, duplicates = deduplicate(parsed)
                label_to_uid = {candidate["label"]: candidate["issue_uid"] for candidate in record["candidate_records"]}
                valid_labels = [label for label in unique if label in label_to_uid]
                invalid_labels = [label for label in unique if label not in label_to_uid]
                ranked_uids = [label_to_uid[label] for label in valid_labels]
                predicted = set(ranked_uids)
                full_gold = set(record.get("gold_tail_uids", []))
                pool_gold = {
                    label_to_uid[label] for label in record.get("gold_candidate_labels", [])
                    if label in label_to_uid
                }
                candidate_set = set(label_to_uid.values())
                keys = ("overall", repository_from_record(record))
                for key in keys:
                    aggregates[key].update(ranked_uids, predicted, full_gold, pool_gold, candidate_set)
                    diagnostics[key]["parse_errors"] += not parse_info["parse_ok"]
                    diagnostics[key]["invalid_labels"] += len(invalid_labels)
                    diagnostics[key]["duplicate_labels"] += len(duplicates)
                    diagnostics[key]["latency_seconds"] += latency
                    diagnostics[key]["generated_tokens"] += generated.shape[1] - len(prompt_ids)
                write_prediction(prediction_handle, {
                    "repository": repository, "query_id": record["query_id"],
                    "query_uid": record["query_uid"], "relation": record["relation"],
                    "raw_output": output_text, "parse_ok": parse_info["parse_ok"],
                    "parse_error": parse_info["parse_error"], "predicted_labels": valid_labels,
                    "predicted_tail_uids": ranked_uids, "invalid_labels": invalid_labels,
                    "duplicate_labels": duplicates, "gold_candidate_labels": record.get("gold_candidate_labels", []),
                    "gold_tail_uids": record.get("gold_tail_uids", []), "latency_seconds": latency,
                })
                processed += 1
                if processed % 100 == 0:
                    print(f"Evaluated {processed} set queries in {time.perf_counter() - started:.1f}s", flush=True)
            if args.max_samples and processed >= args.max_samples:
                break
    finally:
        if prediction_handle is not None:
            prediction_handle.close()

    results = {}
    for key, aggregate in sorted(aggregates.items()):
        result = aggregate.finalize()
        count = result["queries"]
        info = diagnostics[key]
        result["generation"] = {
            "parse_error_rate": info["parse_errors"] / count if count else 0.0,
            "invalid_labels": int(info["invalid_labels"]),
            "duplicate_labels": int(info["duplicate_labels"]),
            "mean_latency_seconds": info["latency_seconds"] / count if count else 0.0,
            "mean_generated_tokens": info["generated_tokens"] / count if count else 0.0,
        }
        results[key] = result
    metrics = {
        "task": "set_retrieval", "split": args.split, "model_dataset": args.model_dataset,
        "eval_dataset": args.eval_dataset, "adapter_path": str(args.adapter_path),
        "elapsed_seconds": time.perf_counter() - started, "results": results,
        "methodology": {
            "decoding": "greedy", "gold_scope": "gold_tail_uids",
            "pool_scope": "gold tails present in the natural candidate pool",
            "invalid_identifiers": "excluded from retrieval metrics and reported separately",
        },
    }
    save_metrics(args, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

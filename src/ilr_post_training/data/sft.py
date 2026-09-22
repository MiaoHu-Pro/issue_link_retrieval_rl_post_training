"""Build relation-conditioned pointwise and set-retrieval SFT data.

The builder uses a deterministic lexical candidate retriever.
It never inserts
gold tails into a natural candidate pool; candidate recall is reported instead.
Unrecorded candidates are operational negatives under the closed-world label
regime and are marked as such in sidecar fields.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import tomllib

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.+-]{1,}", re.I)
REPOSITORIES = ("Apache", "Jira", "RedHat", "MongoDB", "Qt", "Mojang")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def parse_date(value: str):
    return datetime.strptime(value, DATE_FORMAT)


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.casefold()))


def read_issues(path: Path):
    issues, duplicate_keys = {}, 0
    with path.open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 8:
                raise ValueError(f"Invalid issue row {path}:{row_number}")
            source_id, key, title, project, issue_type, status, description, created = fields
            date = parse_date(created)
            text = "\n".join(x for x in (f"Title: {title}" if title else "", f"Description: {description}" if description else "") if x)
            record = {"issue_uid": f"{path.parent.name}:{key}", "key": key, "source_id": source_id,
                      "title": title, "description": description, "text": text, "project": project,
                      "issue_type": issue_type, "status": status, "created": created, "date": date,
                      "tokens": tokens(text)}
            if key in issues:
                duplicate_keys += 1
                if (record["created"], record["source_id"], record["text"]) < (issues[key]["created"], issues[key]["source_id"], issues[key]["text"]):
                    issues[key] = record
            else:
                issues[key] = record
    return issues, {"source_rows": sum(1 for _ in path.open(encoding="utf-8")), "unique_keys": len(issues), "duplicate_key_rows": duplicate_keys}


def read_links(path: Path):
    groups = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        declared = int(next(handle).strip())
        rows = 0
        for line in handle:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 4:
                raise ValueError(f"Invalid link row {path}")
            head, relation, tail, created = fields
            parse_date(created)
            groups[(head, relation)].add(tail)
            rows += 1
    if declared != rows:
        raise ValueError(f"Link count mismatch in {path}: {declared} != {rows}")
    return groups, rows


def read_relations(path: Path):
    relations = {}
    with path.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"Invalid relation row {path}")
            name, definition, relation_id = fields
            relations[name] = {"relation_id": relation_id, "name": name, "definition": definition}
    return relations


class LexicalRetriever:
    def __init__(self, issues: dict, max_posting: int = 50000):
        self.issues = issues
        self.keys = sorted(issues)
        self.newest = sorted(self.keys, key=lambda key: (issues[key]["date"], key), reverse=True)
        self.postings = defaultdict(list)
        for key in self.keys:
            for term in issues[key]["tokens"]:
                self.postings[term].append(key)
        self.n = len(self.keys)
        self.idf = {term: math.log((self.n + 1) / (len(posting) + 1)) + 1.0
                    for term, posting in self.postings.items() if len(posting) <= max_posting}
        self.postings = {term: posting for term, posting in self.postings.items() if term in self.idf}

    def retrieve(self, query, query_date, relation, limit):
        scores = defaultdict(float)
        query_terms = tokens(query["text"] + " " + relation)
        for term in query_terms:
            weight = self.idf.get(term)
            if weight is None:
                continue
            for key in self.postings[term]:
                candidate = self.issues[key]
                if key != query["key"] and candidate["date"] < query_date:
                    scores[key] += weight
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
        if len(ranked) < limit:
            used = {key for key, _ in ranked}
            fallback = []
            for key in self.newest:
                if key != query["key"] and key not in used and self.issues[key]["date"] < query_date:
                    fallback.append((key, 0.0))
                    if len(fallback) >= limit - len(ranked):
                        break
            ranked += fallback
        return ranked


def split_name(query_key, relation, source):
    if source == "test":
        return "test"
    bucket = int(digest(f"sft-v1:{query_key}:{relation}"), 16) % 100
    return "validation" if bucket < 10 else "train"


def candidate_record(issue, label, rank, score):
    return {"label": label, "issue_uid": issue["issue_uid"], "issue_key": issue["key"],
            "text": issue["text"], "rank": rank, "retriever_score": round(score, 6)}


def render_prompt(query, relation, definition, candidates):
    rows = [f"Query issue {query['issue_uid']}:\n{query['text']}", f"Relation: {relation}", f"Relation definition: {definition}", "Candidates:"]
    rows.extend(f"{item['label']}: {item['text']}" for item in candidates)
    return "\n".join(rows)


def build_repository(project_root: Path, repository: str, args):
    raw = project_root / "data/raw" / repository
    issue_path = raw / "ID_Name_Project_Type_Status_sMention_Time.txt"
    issues, issue_stats = read_issues(issue_path)
    relations = read_relations(raw / "relation2id.txt")
    train_groups, train_rows = read_links(raw / "train.txt")
    test_groups, test_rows = read_links(raw / "test.txt")
    retriever = LexicalRetriever(issues, args.max_posting)
    version = repository.lower() + "_v1" + (f"_{args.version_suffix}" if args.version_suffix else "")
    root = project_root / "data/training/sft"
    set_root, point_root = root / "set_retrieval" / version, root / "pointwise" / version
    if set_root.exists() or point_root.exists():
        raise FileExistsError(f"SFT version already exists for {repository}; remove only after deliberate review or use a new version")
    with tempfile.TemporaryDirectory(prefix=f"sft-{repository.lower()}-", dir=project_root / "cache") as temp:
        stage = Path(temp)
        for task in ("set_retrieval", "pointwise"):
            for split in ("train", "validation", "test"):
                (stage / task / version).mkdir(parents=True, exist_ok=True)
        counts = Counter(); recall = Counter(); missing = Counter(); query_counts = Counter()
        all_groups = [("train", key, rel, tails) for (key, rel), tails in train_groups.items()]
        all_groups += [("test", key, rel, tails) for (key, rel), tails in test_groups.items()]
        files = {}
        for task in ("set_retrieval", "pointwise"):
            for split in ("train", "validation", "test"):
                files[task, split] = (stage / task / version / f"{split}.jsonl").open("w", encoding="utf-8")
        try:
            for group_index, (source, key, relation, gold_keys) in enumerate(all_groups):
                if args.max_groups and group_index >= args.max_groups:
                    break
                if group_index and group_index % 1000 == 0:
                    print(f"{repository}: processed {group_index}/{len(all_groups)} query groups", flush=True)
                query = issues.get(key)
                relation_info = relations.get(relation, {"name": relation, "definition": "Recorded relation label; definition unavailable."})
                if query is None:
                    missing["query_issue_missing"] += 1
                    continue
                valid_gold = {tail for tail in gold_keys if tail in issues and issues[tail]["date"] < query["date"]}
                ranked = retriever.retrieve(query, query["date"], relation, args.candidate_count)
                ranked_keys = {candidate_key for candidate_key, _ in ranked}
                eligible_gold = valid_gold & ranked_keys
                split = split_name(key, relation, source)
                labels = [f"C{i:03d}" for i in range(1, len(ranked) + 1)]
                candidates = [candidate_record(issues[candidate_key], label, i, score) for i, ((candidate_key, score), label) in enumerate(zip(ranked, labels), 1)]
                target_labels = [item["label"] for item in candidates if item["issue_key"] in eligible_gold]
                record_base = {"query_id": f"{repository}:{key}:{relation}", "query_uid": query["issue_uid"],
                               "query_text": query["text"], "relation": relation, "relation_definition": relation_info["definition"],
                               "candidate_records": candidates, "gold_candidate_labels": target_labels,
                               "gold_tail_uids": sorted(query["issue_uid"].split(":")[0] + ":" + x for x in valid_gold),
                               "candidate_pool_size": len(candidates), "candidate_pool_mode": "natural_lexical",
                               "judgment_regime": "closed_world_recorded_links", "split": split}
                set_record = {"prompt": [{"role": "system", "content": "Select all supplied candidates satisfying the requested relation. Return only JSON with a unique tails array. An empty array is allowed."}, {"role": "user", "content": render_prompt(query, relation, relation_info["definition"], candidates)}], "completion": [{"role": "assistant", "content": json.dumps({"tails": target_labels}, separators=(",", ":"))}], **record_base}
                files["set_retrieval", split].write(encode(set_record) + "\n")
                counts["set_" + split] += 1
                recall[split] += len(eligible_gold)
                missing[split + "_gold"] += len(valid_gold - eligible_gold)
                query_counts[split] += len(valid_gold)
                for item in candidates:
                    label = item["label"]
                    is_positive = item["issue_key"] in valid_gold
                    point = {"prompt": [{"role": "system", "content": "Classify whether the candidate satisfies the requested relation. Return only JSON with a valid boolean."}, {"role": "user", "content": render_prompt(query, relation, relation_info["definition"], [item])}], "completion": [{"role": "assistant", "content": json.dumps({"valid": is_positive}, separators=(",", ":"))}], "query_id": record_base["query_id"], "query_uid": query["issue_uid"], "relation": relation, "relation_definition": relation_info["definition"], "candidate": item, "label": int(is_positive), "judgment_regime": "closed_world_recorded_links", "split": split}
                    files["pointwise", split].write(encode(point) + "\n")
                    counts["point_" + split] += 1
        finally:
            for handle in files.values():
                handle.close()
        metadata = {"repository": repository, "version": version, "candidate_count": args.candidate_count, "max_posting": args.max_posting,
                    "retriever": "lexical IDF overlap with creation-time eligibility", "split_policy": "train groups hashed 90/10; all test groups held out",
                    "issue_stats": issue_stats, "train_link_rows": train_rows, "test_link_rows": test_rows, "relation_count": len(relations),
                    "counts": dict(counts), "eligible_gold": dict(recall), "missing_gold": dict(missing), "gold_totals": dict(query_counts),
                    "candidate_recall": {split: (recall[split] / query_counts[split] if query_counts[split] else 0.0) for split in ("train", "validation", "test")},
                    "label_regime": "closed_world_recorded_links", "unknown_candidates_are_not_verified_negatives": True}
        (stage / "set_retrieval" / version / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (stage / "pointwise" / version / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        set_root.parent.mkdir(parents=True, exist_ok=True); point_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage / "set_retrieval" / version, set_root)
        shutil.copytree(stage / "pointwise" / version, point_root)
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument("--repositories", nargs="+", choices=REPOSITORIES, default=list(REPOSITORIES))
    p.add_argument("--candidate-count", type=int, default=32)
    p.add_argument("--max-posting", type=int, default=50000)
    p.add_argument("--max-groups", type=int, default=0, help="Development limit per repository; 0 means all groups")
    p.add_argument("--version-suffix", default="", help="Append a version suffix, for example full or pilot")
    args = p.parse_args(); args.project_root = args.project_root.resolve()
    for repository in args.repositories:
        print(f"Building SFT data for {repository}", flush=True)
        metadata = build_repository(args.project_root, repository, args)
        print(json.dumps({"repository": repository, "counts": metadata["counts"], "candidate_recall": metadata["candidate_recall"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

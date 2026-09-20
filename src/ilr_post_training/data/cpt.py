"""Build versioned CPT JSONL from legacy issue snapshots using only the stdlib."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import re
import sqlite3
import tempfile
import tomllib
import unicodedata

ISSUE_FILE = "ID_Name_Project_Type_Status_sMention_Time.txt"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
PLACEHOLDERS = {"", "none", "null", "nan", "no name", "no description"}


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_date(value):
    parsed = datetime.strptime(value, DATE_FORMAT)
    if parsed.strftime(DATE_FORMAT) != value:
        raise ValueError(f"Noncanonical timestamp: {value!r}")
    return parsed


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_membership(path):
    result = {}
    count = 0
    with Path(path).open(encoding="utf-8") as handle:
        declared = int(next(handle).strip())
        for number, line in enumerate(handle, 2):
            row = line.rstrip("\r\n").split("\t")
            if len(row) != 3 or not row[0] or not row[1]:
                raise ValueError(f"Invalid membership row {path}:{number}")
            key, source_id, date = row
            parse_date(date)
            entry = (source_id, date)
            if key in result and result[key] != entry:
                raise ValueError(f"Conflicting membership identity: {key}")
            result[key] = entry
            count += 1
    if count != declared:
        raise ValueError(f"Membership header mismatch: {path}: {declared} != {count}")
    return result, {"declared_rows": declared, "rows": count, "unique_ids": len(result), "duplicate_rows": count - len(result)}


def read_query_heads(path):
    heads = set()
    with Path(path).open(encoding="utf-8") as handle:
        declared = int(next(handle).strip())
        count = 0
        for line in handle:
            row = line.rstrip("\r\n").split("\t")
            if len(row) != 4:
                raise ValueError(f"Invalid query/link row in {path}")
            parse_date(row[3])
            heads.add(row[0])
            count += 1
    if count != declared:
        raise ValueError(f"Query header mismatch in {path}")
    return heads


class Sanitizer:
    """Use known issue-project prefixes; preserve ordinary numbers and punctuation."""
    def __init__(self, keys, description_words=50):
        prefixes = sorted({key.rsplit("-", 1)[0] for key in keys if re.fullmatch(r".+-\d+", key)}, key=lambda x: (-len(x), x))
        self.issue_pattern = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in prefixes) + r")-\d+\b", re.I) if prefixes else re.compile(r"(?!)")
        self.description_words = description_words

    def field(self, text, flags):
        original = text
        text = unicodedata.normalize("NFKC", text)
        text = "".join(" " if unicodedata.category(c).startswith("C") else c for c in text)
        text = " ".join(text.split())
        if text != original:
            flags.add("normalized_unicode_whitespace_controls")
        if text.casefold() in PLACEHOLDERS:
            if text:
                flags.add("placeholder_removed")
            return ""
        text, count = re.subn(r"https?://[^\s]*(?:/browse/|/issues/)[^\s]+", "[ISSUE_REF]", text, flags=re.I)
        if count:
            flags.add("issue_url_masked")
        text, count = self.issue_pattern.subn("[ISSUE_REF]", text)
        if count:
            flags.add("issue_id_masked")
        # Remove only explicit relation-reference spans, never generic domain terms.
        text, count = re.subn(r"\b(?:is\s+)?(?:a\s+)?(?:duplicate\s+of|duplicates|blocked\s+by|blocks|relates\s+to|depends\s+on)\s+\[ISSUE_REF\]", " ", text, flags=re.I)
        if count:
            flags.add("explicit_link_reference_removed")
        return " ".join(text.split())

    def __call__(self, title, description):
        flags = set()
        title = self.field(title, flags)
        description = self.field(description, flags)
        if description and description.casefold() == title.casefold():
            description = ""
            flags.add("description_equals_title_removed")
        content = title + " " + description
        meaningful = re.sub(r"\[ISSUE_REF\]", "", content)
        if not any(c.isalnum() for c in meaningful):
            return "", "", sorted(flags | {"empty_after_cleaning"})
        if not title:
            flags.add("description_only")
        if not description:
            flags.add("title_only")
        if len(content.split()) <= 5:
            flags.add("short_text")
        parts = (["Title: " + title] if title else []) + (["Description: " + description] if description else [])
        desc_hash = digest(description.casefold()) if len(description.split()) >= self.description_words else ""
        return "\n".join(parts), desc_hash, sorted(flags)


def split_for(group_hash, config):
    bucket = int(digest(config["validation_seed"] + ":" + group_hash), 16) % config["bucket_count"]
    return "validation" if bucket < config["validation_buckets"] else "train"


class Groups:
    def __init__(self):
        self.parent = {}

    def find(self, node):
        self.parent.setdefault(node, node)
        trail = []
        while self.parent[node] != node:
            trail.append(node)
            node = self.parent[node]
        for member in trail:
            self.parent[member] = node
        return node

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def validate_export(corpus, processed, splits, config):
    """Independently reread every output row and compare it to canonical SQLite."""
    connection = sqlite3.connect(Path(processed) / "canonical_issues.sqlite")
    connection.row_factory = sqlite3.Row
    seen_hashes, seen_ids, group_splits, description_splits = set(), set(), {}, {}
    counts = Counter()
    for split in ("train", "validation"):
        membership_ids = {}
        with (Path(splits) / f"{split}_issue_ids.jsonl").open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row["issue_uid"] in membership_ids:
                    raise ValueError("Duplicate output split identity")
                membership_ids[row["issue_uid"]] = row["representative_uid"]
        output_ids = {}
        for shard in sorted(Path(corpus).glob(f"{split}-*.jsonl")):
            with shard.open(encoding="utf-8") as docs, (Path(corpus) / "provenance" / shard.name).open(encoding="utf-8") as sidecar:
                from itertools import zip_longest
                for row_index, (line, meta_line) in enumerate(zip_longest(docs, sidecar)):
                    if line is None or meta_line is None:
                        raise ValueError("Document/provenance row mismatch")
                    doc, meta = json.loads(line), json.loads(meta_line)
                    if set(doc) != {"text"} or not doc["text"]:
                        raise ValueError("Invalid model document schema")
                    h = digest(doc["text"].casefold())
                    if h != meta["text_hash"] or h in seen_hashes:
                        raise ValueError("Text hash mismatch or duplicate text")
                    seen_hashes.add(h)
                    if meta["row_index"] != row_index or meta["split"] != split:
                        raise ValueError("Invalid row index/split")
                    group = meta["group_hash"]
                    if split_for(group, config) != split or group_splits.setdefault(group, split) != split:
                        raise ValueError("Group leakage or nondeterministic split")
                    for uid in [meta["issue_uid"], *meta["dedup_aliases"]]:
                        if uid in seen_ids:
                            raise ValueError("Identity appears twice in export")
                        seen_ids.add(uid)
                        canonical = connection.execute("SELECT * FROM issues WHERE uid=?", (uid,)).fetchone()
                        if canonical is None or canonical["membership"] != "train" or canonical["outcome"] != split:
                            raise ValueError("Exported nontraining identity")
                        if canonical["created"] > config["cutoff"] or canonical["text_hash"] != h:
                            raise ValueError("Temporal or canonical content mismatch")
                        if canonical["text"] != doc["text"] and digest(canonical["text"].casefold()) != h:
                            raise ValueError("Canonical text mismatch")
                        desc = canonical["desc_hash"]
                        if desc and description_splits.setdefault(desc, split) != split:
                            raise ValueError("Substantive description crosses splits")
                        output_ids[uid] = meta["issue_uid"]
                    counts[split] += 1
                    counts[split + "_words"] += len(doc["text"].split())
        if output_ids != membership_ids:
            raise ValueError("Split ledger does not match corpus and aliases")
    heldout_hashes = {r[0] for r in connection.execute("SELECT hash FROM heldout_text")}
    if seen_hashes & heldout_hashes:
        raise ValueError("Held-out exact text contamination")
    heldout_desc = {r[0] for r in connection.execute("SELECT hash FROM heldout_description")}
    if set(description_splits) & heldout_desc:
        raise ValueError("Held-out substantive description contamination")
    expected = connection.execute("SELECT count(*) FROM issues WHERE outcome IN ('train','validation')").fetchone()[0]
    if expected != len(seen_ids):
        raise ValueError("Canonical/output identity coverage mismatch")
    connection.close()
    return {"passed": True, "documents": dict(counts), "represented_issue_ids": len(seen_ids), "exact_text_overlap": 0, "substantive_description_overlap": 0}


def build(project_root, config_path):
    project_root, config_path = Path(project_root).resolve(), Path(config_path).resolve()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    version, repository = config["version"], config["repository"]
    if not re.fullmatch(r"[a-z0-9_]+", version):
        raise ValueError("Version must be a simple lowercase identifier")
    if not 0 < config["validation_buckets"] < config["bucket_count"] or config["shard_size"] < 1:
        raise ValueError("Invalid split/shard configuration")
    if config["sanitization_version"] != "conservative-v1" or config["near_duplicate_policy"] != "disabled_pending_review":
        raise ValueError("Unsupported sanitation/near-duplicate policy")
    parse_date(config["cutoff"])
    raw = project_root / config["raw_directory"]
    targets = {"corpus": project_root / "data/training/cpt" / version,
               "processed": project_root / "data/processed/cpt" / version,
               "splits": project_root / "data/splits/cpt" / version}
    if any(p.exists() for p in targets.values()):
        raise FileExistsError("Version already exists; select a new version rather than overwriting")
    source_paths = [raw / name for name in (ISSUE_FILE, "train_entity2id.txt", "test_entity2id.txt", "test.txt")]
    cutoff_path = raw.parent / "cut_off_time.txt"
    if cutoff_path.exists():
        source_paths.append(cutoff_path)
        matching = [line for line in cutoff_path.read_text().splitlines() if line.startswith(repository + "\t")]
        if not matching or config["cutoff"][:10] not in matching[0]:
            raise ValueError("Configuration cutoff disagrees with source cutoff document")
    sources = [{"path": str(p.relative_to(project_root)), "bytes": p.stat().st_size, "sha256": file_hash(p)} for p in source_paths]
    train, train_stats = read_membership(raw / "train_entity2id.txt")
    heldout, heldout_stats = read_membership(raw / "test_entity2id.txt")
    if train.keys() & heldout.keys():
        raise ValueError("Train/held-out identity overlap")
    if any(date > config["cutoff"] for _, date in train.values()) or any(date <= config["cutoff"] for _, date in heldout.values()):
        raise ValueError("Membership violates cutoff")
    heads = read_query_heads(raw / "test.txt")
    if heads & train.keys() or not heads <= heldout.keys():
        raise ValueError("Held-out query heads do not match held-out membership")
    sanitizer = Sanitizer(train.keys() | heldout.keys(), config["substantive_description_words"])
    cache = project_root / "cache"
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cpt-build-", dir=cache) as temporary:
        stage = {key: Path(temporary) / key for key in targets}
        for p in stage.values():
            p.mkdir()
        (stage["corpus"] / "provenance").mkdir()
        connection = sqlite3.connect(stage["processed"] / "canonical_issues.sqlite")
        connection.execute("CREATE TABLE issues(uid TEXT PRIMARY KEY, source_id TEXT, created TEXT, project TEXT, issue_type TEXT, membership TEXT, source_rows TEXT, raw_hash TEXT, text TEXT, text_hash TEXT, desc_hash TEXT, flags TEXT, reason TEXT, outcome TEXT)")
        connection.execute("CREATE TABLE heldout_text(hash TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE heldout_description(hash TEXT PRIMARY KEY)")
        stats = Counter()
        flag_counts = Counter()
        seen_keys = set()
        samples = defaultdict(list)
        with (stage["processed"] / "quarantine_rows.jsonl").open("w", encoding="utf-8") as quarantine, (raw / ISSUE_FILE).open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                stats["source_rows"] += 1
                row = line.rstrip("\r\n").split("\t")
                if len(row) != 8:
                    stats["malformed_rows"] += 1
                    quarantine.write(encode({"source_row": number, "reason": "invalid_field_count", "raw_line_hash": digest(line)}) + "\n")
                    continue
                source_id, key, title, project, issue_type, _status, description, created = row
                uid = repository + ":" + key
                membership = "train" if key in train else "heldout" if key in heldout else "unassigned"
                stats[membership + "_source_rows"] += 1
                seen_keys.add(key)
                text, desc_hash, flags = sanitizer(title, description)
                text_hash = digest(text.casefold()) if text else ""
                reason = ""
                try:
                    parse_date(created)
                except ValueError:
                    reason = "invalid_date"
                expected = train.get(key) or heldout.get(key)
                if expected and expected != (source_id, created):
                    reason = "membership_metadata_mismatch"
                if not key or not source_id:
                    reason = "missing_identity"
                if membership == "unassigned":
                    reason = "unassigned_identity"
                if membership == "heldout":
                    if text_hash:
                        connection.execute("INSERT OR IGNORE INTO heldout_text VALUES (?)", (text_hash,))
                    if desc_hash:
                        connection.execute("INSERT OR IGNORE INTO heldout_description VALUES (?)", (desc_hash,))
                raw_hash = digest(encode(row))
                previous = connection.execute("SELECT raw_hash, source_rows, reason FROM issues WHERE uid=?", (uid,)).fetchone()
                if previous:
                    stats["duplicate_identity_rows"] += 1
                    source_rows = json.loads(previous[1]) + [number]
                    conflict = previous[0] != raw_hash
                    stats["conflicting_duplicate_rows"] += int(conflict)
                    connection.execute("UPDATE issues SET source_rows=?, reason=? WHERE uid=?", (encode(source_rows), "conflicting_identity" if conflict else previous[2], uid))
                    if conflict:
                        quarantine.write(encode({"issue_uid": uid, "source_row": number, "reason": "conflicting_identity", "raw_line_hash": raw_hash}) + "\n")
                    continue
                connection.execute("INSERT INTO issues VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (uid, source_id, created, project, issue_type, membership, encode([number]), raw_hash, text, text_hash, desc_hash, encode(flags), reason, "pending"))
                if stats["source_rows"] % 100000 == 0:
                    print(f"Parsed {stats['source_rows']:,} issue rows", flush=True)
        if (train.keys() | heldout.keys()) - seen_keys:
            raise ValueError("Membership IDs lack parseable issue rows; inspect source data")
        connection.commit()
        heldout_hashes = {r[0] for r in connection.execute("SELECT hash FROM heldout_text")}
        heldout_desc = {r[0] for r in connection.execute("SELECT hash FROM heldout_description")}
        connection.row_factory = sqlite3.Row
        representatives = {}
        aliases = defaultdict(list)
        groups = Groups()
        desc_to_hash = {}
        exclusions = Counter()
        candidates = connection.execute("SELECT * FROM issues ORDER BY created,uid").fetchall()
        with (stage["processed"] / "exclusions.jsonl").open("w", encoding="utf-8") as excluded:
            for row in candidates:
                stats["unique_identities"] += 1
                reason = row["reason"]
                if not reason and row["membership"] == "heldout":
                    reason = "heldout_identity"
                if not reason and not row["text"]:
                    reason = "empty_after_cleaning"
                if not reason and row["text_hash"] in heldout_hashes:
                    reason = "heldout_exact_text_overlap"
                if not reason and row["desc_hash"] and row["desc_hash"] in heldout_desc:
                    reason = "heldout_substantive_description_overlap"
                if row["membership"] == "train":
                    stats["unique_train_identities"] += 1
                    flag_counts.update(json.loads(row["flags"]))
                if reason:
                    exclusions[reason] += 1
                    if row["membership"] == "train":
                        stats["excluded_train_identities"] += 1
                    excluded.write(encode({"issue_uid": row["uid"], "source_rows": json.loads(row["source_rows"]), "reason": reason, "membership": row["membership"]}) + "\n")
                    connection.execute("UPDATE issues SET outcome='excluded',reason=? WHERE uid=?", (reason, row["uid"]))
                    continue
                h = row["text_hash"]
                groups.find(h)
                if row["desc_hash"]:
                    other = desc_to_hash.setdefault(row["desc_hash"], h)
                    groups.union(h, other)
                if h in representatives:
                    aliases[h].append(row["uid"])
                    stats["exact_text_duplicate_extra_identities"] += 1
                else:
                    representatives[h] = row
        del candidates
        counts = Counter()
        composition = {"train": Counter(), "validation": Counter()}
        with ExitStack() as stack:
            id_files = {s: stack.enter_context((stage["splits"] / f"{s}_issue_ids.jsonl").open("w", encoding="utf-8")) for s in ("train", "validation")}
            duplicate_file = stack.enter_context((stage["processed"] / "duplicate_groups.jsonl").open("w", encoding="utf-8"))
            doc_files, meta_files = {}, {}
            for h, row in representatives.items():
                group_hash = groups.find(h)
                split = split_for(group_hash, config)
                index = counts[split]
                shard_index, row_index = divmod(index, config["shard_size"])
                filename = f"{split}-{shard_index:05d}.jsonl"
                if row_index == 0:
                    if split in doc_files:
                        doc_files[split].close()
                        meta_files[split].close()
                    doc_files[split] = stack.enter_context((stage["corpus"] / filename).open("w", encoding="utf-8"))
                    meta_files[split] = stack.enter_context((stage["corpus"] / "provenance" / filename).open("w", encoding="utf-8"))
                alias_ids = aliases[h]
                meta = {"row_index": row_index, "issue_uid": row["uid"], "source_rows": json.loads(row["source_rows"]), "source_file": str((raw / ISSUE_FILE).relative_to(project_root)), "created_at": row["created"], "text_hash": h, "group_hash": group_hash, "split": split, "quality_flags": json.loads(row["flags"]), "dedup_aliases": alias_ids}
                doc_files[split].write(encode({"text": row["text"]}) + "\n")
                meta_files[split].write(encode(meta) + "\n")
                for uid in [row["uid"], *alias_ids]:
                    id_files[split].write(encode({"issue_uid": uid, "representative_uid": row["uid"], "text_hash": h, "group_hash": group_hash}) + "\n")
                    connection.execute("UPDATE issues SET outcome=? WHERE uid=?", (split, uid))
                if alias_ids:
                    duplicate_file.write(encode({"text_hash": h, "representative_uid": row["uid"], "alias_uids": alias_ids, "split": split}) + "\n")
                counts[split] += 1
                counts[split + "_words"] += len(row["text"].split())
                composition[split][row["project"]] += 1
                for flag in ["general", *json.loads(row["flags"])]:
                    if len(samples[flag]) < 5:
                        samples[flag].append({"issue_uid": row["uid"], "text": row["text"], "split": split})
        stats["documents"] = len(representatives)
        stats["split_groups"] = len({groups.find(h) for h in representatives})
        if stats["unique_train_identities"] != stats["excluded_train_identities"] + stats["exact_text_duplicate_extra_identities"] + stats["documents"]:
            raise ValueError("Training identity counts do not reconcile")
        if not counts["train"] or not counts["validation"]:
            raise ValueError("Empty CPT partition; adjust fixture size/configuration")
        connection.commit()
        connection.close()
        validation = validate_export(stage["corpus"], stage["processed"], stage["splits"], config)
        if validation["documents"] != dict(counts):
            raise ValueError("Exported counts disagree with builder")
        # Verify raw inputs did not change during the build.
        for source in sources:
            if file_hash(project_root / source["path"]) != source["sha256"]:
                raise ValueError("Source changed during preparation")
        quality = {"counts": dict(stats), "output_counts": dict(counts), "exclusions": dict(exclusions), "train_quality_flags": dict(flag_counts), "project_document_counts": {s: dict(c) for s, c in composition.items()}, "sample_documents": dict(samples), "validation": validation, "near_duplicates": "Not detected/removed by similarity; exact full text and substantive exact descriptions only.", "tokenization": "Not performed; no tokenizer dependency or model download."}
        write_json(stage["corpus"] / "quality_report.json", quality)
        card = f"""# {repository} CPT — {version}

Retrospective processed-snapshot domain text, title/description only. Cutoff: {config['cutoff']} (source timezone unspecified).

- Training documents: {counts['train']:,}; validation documents: {counts['validation']:,}.
- Validation uses deterministic hash groups, approximately {100 * config['validation_buckets'] / config['bucket_count']:.1f}%.
- Exact full-text copies are deduplicated; substantive identical descriptions stay in one split.
- Held-out IDs, exact documents, and substantive identical descriptions are excluded.
- Read only train-*.jsonl for training and validation-*.jsonl for CPT validation. Each record has only `text`.
- Sidecars, quality reports, and canonical SQLite are provenance, never model input.
- Original exports were already cleaned and descriptions truncated; no historical edits can be reconstructed.
- No near-duplicate similarity filtering or guarantee of removal of every semantic link reference.
- No tokenization or packing yet. Word counts are not tokenizer counts. No chat template is applied.
- Corpus preparation does not establish data licensing or foundation-model pretraining provenance; retain original source attribution and terms.

See manifest.json and quality_report.json for exact counts, hashes, configuration, and limitations.
"""
        (stage["corpus"] / "dataset_card.md").write_text(card, encoding="utf-8")
        artifacts = []
        for key, directory in stage.items():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    artifacts.append({"path": str((targets[key] / path.relative_to(directory)).relative_to(project_root)), "bytes": path.stat().st_size, "sha256": file_hash(path)})
        manifest = {"schema_version": 1, "repository": repository, "version": version, "config": config, "config_sha256": file_hash(config_path), "builder_sha256": file_hash(Path(__file__)), "python_version": platform.python_version(), "sources": sources, "membership": {"train": train_stats, "heldout": heldout_stats}, "counts": dict(stats), "output_counts": dict(counts), "validation": validation, "artifacts": artifacts, "canonical_format": "SQLite (stdlib, no Parquet dependency)", "tokenizer": None, "build_status": "validated"}
        write_json(stage["corpus"] / "manifest.json", manifest)
        # No user data is overwritten. Corpus is published last as completion marker.
        for key in ("processed", "splits", "corpus"):
            targets[key].parent.mkdir(parents=True, exist_ok=True)
            stage[key].rename(targets[key])
    print(json.dumps({"corpus": str(targets["corpus"]), "counts": dict(counts), "validation": "passed"}, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    build(args.project_root, args.config)

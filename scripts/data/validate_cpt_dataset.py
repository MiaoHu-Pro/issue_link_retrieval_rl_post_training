#!/usr/bin/env python3
"""Check hashes, source membership, and every document/provenance row in a CPT corpus."""
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from ilr_post_training.data.cpt import file_hash, read_membership, validate_export


def validate(project_root, version):
    root = Path(project_root).resolve()
    corpus = root / "data/training/cpt" / version
    manifest = json.loads((corpus / "manifest.json").read_text())
    for entry in manifest["sources"] + manifest["artifacts"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest path escapes project")
        if path.stat().st_size != entry["bytes"] or file_hash(path) != entry["sha256"]:
            raise ValueError(f"Artifact/source checksum mismatch: {entry['path']}")
    raw = root / manifest["config"]["raw_directory"]
    train, _ = read_membership(raw / "train_entity2id.txt")
    heldout, _ = read_membership(raw / "test_entity2id.txt")
    repo = manifest["repository"]
    for split in ("train", "validation"):
        with (root / "data/splits/cpt" / version / f"{split}_issue_ids.jsonl").open() as handle:
            for line in handle:
                uid = json.loads(line)["issue_uid"]
                prefix, key = uid.split(":", 1)
                if prefix != repo or key not in train or key in heldout:
                    raise ValueError("Output ID violates original membership")
    result = validate_export(corpus, root / "data/processed/cpt" / version, root / "data/splits/cpt" / version, manifest["config"])
    if result != manifest["validation"] or result["documents"] != manifest["output_counts"]:
        raise ValueError("Manifest counts differ from revalidated export")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="redhat_v1")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    print(json.dumps(validate(args.project_root, args.version), indent=2))

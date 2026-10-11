"""Pin TAB sources, reuse the board curator, and fail closed on split contamination.

Run with --curator pointing to klusai-datasets/scripts/curate_tab.py (read only).
All downloads and curated rows stay in the assigned worktree; only hashes/counts are committed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import urllib.request
from pathlib import Path

SOURCE = "https://github.com/NorskRegnesentral/text-anonymization-benchmark"
REVISION = "558e09e26d6b36f5f78440074e6a233946d98bd9"
HF_REVISION = "58faa4fa572cd8064818a6e1f8ca6e361b959c00"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit_splits(splits: dict[str, list[dict]]) -> dict:
    result = {}
    for a, b in itertools.combinations(splits, 2):
        ids_a = {r["doc_id"] for r in splits[a]}
        ids_b = {r["doc_id"] for r in splits[b]}
        texts_a = {sha(" ".join(r["text"].split()).encode()) for r in splits[a]}
        texts_b = {sha(" ".join(r["text"].split()).encode()) for r in splits[b]}
        result[f"{a}_vs_{b}"] = {
            "doc_id_overlap": len(ids_a & ids_b),
            "normalized_text_overlap": len(texts_a & texts_b),
        }
        if ids_a & ids_b or texts_a & texts_b:
            raise ValueError(f"Contamination: {a} overlaps {b}: {result}")
    return result


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curator", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/res104-tab"))
    ap.add_argument("--manifest", type=Path, default=Path("docs/res104-tab-manifest.json"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("tab_curator", args.curator)
    curator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(curator)
    raw_base = f"https://raw.githubusercontent.com/NorskRegnesentral/text-anonymization-benchmark/{REVISION}"
    for name in ["LICENSE.txt", "README.md", "echr_train.json", "echr_dev.json", "echr_test.json"]:
        dst = args.out / name
        if not dst.exists():
            partial = dst.with_suffix(dst.suffix + ".partial")
            urllib.request.urlretrieve(f"{raw_base}/{name}", partial)
            partial.replace(dst)
    license_text = (args.out / "LICENSE.txt").read_text()
    if (
        "The MIT License" not in license_text
        or "TAB is released under an MIT License" not in (args.out / "README.md").read_text()
    ):
        raise ValueError("Upstream MIT verification failed; do not train")
    raw = {s: json.loads((args.out / f"echr_{s}.json").read_text()) for s in ("train", "dev", "test")}
    for split, docs in raw.items():
        if any(doc["dataset_type"] != split for doc in docs):
            raise ValueError(f"Unexpected upstream split label in {split}")
    audit = audit_splits(raw)
    curated, stats = {}, {}
    for split in raw:
        curated[split], stats[split] = curator.curate_split(args.out / f"echr_{split}.json")
        stats[split]["source_sha256"] = sha((args.out / f"echr_{split}.json").read_bytes())
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in curated[split])
        (args.out / f"{split}.jsonl").write_text(payload)
        stats[split]["jsonl_sha256"] = sha(payload.encode())
    parquet = hf_hub_download(
        "klusai/europriv-bench",
        "tab-echr-legal-en-v1/test-00000-of-00001.parquet",
        repo_type="dataset",
        revision=HF_REVISION,
    )
    board = pq.read_table(parquet).to_pylist()
    cols = ("doc_id", "text", "spans", "annotator")

    def canonical(rows):
        return {r["doc_id"]: {k: r[k] for k in cols} for r in rows}

    if canonical(board) != canonical(curated["test"]):
        raise ValueError("Curated upstream test differs from the pinned harness test")
    audit["train_dev_vs_harness"] = audit_splits({"train": raw["train"], "dev": raw["dev"], "board": board})
    manifest = {
        "issue": "RES-104",
        "status": "license_and_document_split_gate_passed",
        "source": SOURCE,
        "revision": REVISION,
        "license": "MIT",
        "license_url": f"{SOURCE}/blob/{REVISION}/LICENSE.txt",
        "license_sha256": sha(license_text.encode()),
        "license_notice_local": str(args.out / "LICENSE.txt"),
        "attribution": "Pilan et al. (2022), Text Anonymization Benchmark; Norsk Regnesentral",
        "training_and_redistribution": "MIT; preserve copyright and permission notice",
        "board": {
            "repo": "klusai/europriv-bench",
            "config": "tab-echr-legal-en-v1",
            "split": "test",
            "revision": HF_REVISION,
            "parquet_sha256": sha(Path(parquet).read_bytes()),
            "curated_test_exact_match": True,
            "n": len(board),
        },
        "curator_file_sha256": sha(args.curator.read_bytes()),
        "curator_mapping_sha256": sha(Path(curator.map_tab_document.__code__.co_filename).read_bytes()),
        "splits": stats,
        "disjointness": audit,
        "selection": "TAB dev only; fixed epochs/LR/recipe before test scoring; no test-based arm selection",
        "registered_recipe": {
            "arm": "TAB train only",
            "base": "FacebookAI/xlm-roberta-large",
            "base_license": "MIT",
            "base_url": "https://huggingface.co/FacebookAI/xlm-roberta-large",
            "base_revision": "c23d21b0620b635a76227c604d44e43a9f0ee389",
            "seeds": [0, 1, 2],
            "epochs": 3,
            "learning_rate": 0.0003,
            "lora_rank": 16,
            "batch_size": 16,
            "dtype": "float32",
            "device": "mps",
            "max_length": 256,
            "overflow_stride": 0,
            "selection_metric": "minimum TAB-dev loss",
            "optional_cjeu_mix": "not run",
            "cjeu_baseline": "rescore existing RES-72 checkpoints seeds 0/1/2",
        },
        "not_verified": ["Subject identity disjointness across cases", "Near-duplicate passages across judgments"],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

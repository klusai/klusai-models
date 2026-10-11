#!/usr/bin/env python3
"""Prepare the RES-97 KP recipe + TAB TRAIN only; never publish or read TAB test for training."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import click
from train_xlmr560m_res97 import LANGUAGES, _balanced_downsample

from klusai.privacy.models.training.token_classification import carve_heldout_general, template_disjoint_split

TAB_URL = "https://raw.githubusercontent.com/NorskRegnesentral/text-anonymization-benchmark/master/echr_train.json"
TAB_SHA256 = "4aba41f8ac305ff9e93dd6f0bbc16756e57e9ace396c827931fab70e18d8c6a6"


def write_jsonl(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@click.command()
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--pool-per-language", type=click.IntRange(min=10), default=10000)
@click.option("--kp-max", type=click.IntRange(min=8), default=40000)
@click.option("--tab-max", type=click.IntRange(min=1), default=None)
@click.option("--tab-file", type=click.Path(exists=True, path_type=Path), default=None)
@click.option(
    "--source-manifest",
    type=click.Path(exists=True, path_type=Path),
    default="conf/res100-data-sources.json",
    help="Pinned source revisions and hashes.",
)
@click.option("--seed", type=int, default=0, help="Fixed DATA carve seed, independent of training seed.")
def main(output_dir, pool_per_language, kp_max, tab_max, tab_file, source_manifest, seed):
    import pyarrow.parquet as pq
    from datasets import Dataset
    from huggingface_hub import HfApi, hf_hub_download

    from klusai.privacy.datasets.data.curate import map_tab_document

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "recipe": "RES-97 eight-language carve/balance/in-training split + TAB train",
        "seed": seed,
        "pool_per_language": pool_per_language,
        "kp_max": kp_max,
        "sources": [],
        "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
    }
    manifest_path = output_dir / "manifest.json"
    pinned = {s["repo"]: s for s in json.loads(source_manifest.read_text())["sources"] if "repo" in s}
    all_rows = []
    for lang in LANGUAGES:
        repo = f"klusai/ds-kp-general-{lang}-50k"
        info = HfApi().dataset_info(repo, revision=pinned[repo]["revision"])
        license_id = (info.card_data or {}).get("license", "unknown")
        if license_id.lower() not in {"mit", "apache-2.0", "cc0-1.0", "cc-by-4.0"}:
            raise ValueError(f"STOP: unapproved/unknown training and redistribution license for {repo}: {license_id}")
        files = sorted(
            s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet") and "train" in s.rfilename
        )
        if not files:
            raise ValueError(f"No training parquet found for {repo}")
        rows = []
        hashes = {}
        for filename in files:
            path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=info.sha))
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
            if hashes[filename] != pinned[repo]["sha256"].get(filename):
                raise ValueError(f"Parquet checksum differs from the reviewed manifest: {repo}/{filename}")
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
                rows.extend(batch.to_pylist()[: max(0, pool_per_language - len(rows))])
                if len(rows) >= pool_per_language:
                    break
            if len(rows) >= pool_per_language:
                break
        all_rows.extend({"text": r["text"], "spans": r["spans"], "language": lang} for r in rows)
        manifest["sources"].append(
            {
                "repo": repo,
                "url": f"https://huggingface.co/datasets/{repo}",
                "license": license_id,
                "revision": info.sha,
                "split": "train",
                "pool_rows": len(rows),
                "sha256": hashes,
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        click.echo(f"{lang}: {len(rows)} rows")
    pool, heldout, carve = carve_heldout_general(
        Dataset.from_list(all_rows), heldout_templates_per_language=1, seed=seed
    )
    balanced, counts = _balanced_downsample(pool, max_samples=kp_max, language_key="language", seed=seed)
    train, validation, split = template_disjoint_split(balanced, eval_fraction=0.04, seed=seed)
    if tab_file is None:
        tab_file = output_dir / "echr_train.json"
        if not tab_file.exists():
            urllib.request.urlretrieve(TAB_URL, tab_file)
    if hashlib.sha256(tab_file.read_bytes()).hexdigest() != TAB_SHA256:
        raise ValueError("TAB train checksum changed; review provenance before using new data")
    tab_rows, dropped = [], []
    for doc in json.loads(tab_file.read_text()):
        try:
            tab_rows.append(map_tab_document(doc))
        except ValueError:
            dropped.append(doc["doc_id"])
    if tab_max:
        tab_rows = tab_rows[:tab_max]
    # Preserve the curated train document IDs for overlap audits, even though the model uses text/spans.
    manifest["sources"].append(
        {
            "url": TAB_URL,
            "license": "MIT",
            "split": "train",
            "license_url": "https://github.com/NorskRegnesentral/text-anonymization-benchmark/blob/master/LICENSE.txt",
            "sha256": TAB_SHA256,
            "rows": len(tab_rows),
            "doc_ids": [r["doc_id"] for r in tab_rows],
            "misaligned_dropped": dropped,
        }
    )
    manifest.update(
        {
            "carve": carve,
            "balanced_per_language": counts,
            "validation_split": split,
            "kp_train_rows": len(train),
            "tab_train_rows": len(tab_rows),
        }
    )
    manifest["outputs"] = {}
    for name, rows in (("train", list(train) + tab_rows), ("heldout-kp", heldout), ("validation-kp", validation)):
        manifest["outputs"][name] = {"rows": len(rows), "sha256": write_jsonl(output_dir / f"{name}.jsonl", rows)}
    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    click.echo(json.dumps({k: v for k, v in manifest.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()

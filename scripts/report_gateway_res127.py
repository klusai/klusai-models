"""Build the RES-127 scorecard from harness receipts; never select on test scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from report_res125 import interval, paired, privacy_verdict, utility_verdict
from score_gateway_res127 import CONFIGS, PROTOCOL, ROOT
from scorecard_res125 import digest, write


def build(scratch):
    contract_url = (
        f"https://github.com/klusai/europriv-bench/blob/{PROTOCOL['harness_commit']}/docs/gateway-contract.md"
    )
    records = {}
    for path in sorted((scratch / "results").glob("*.json")):
        record = json.loads(path.read_text())
        records[record["name"], record["config"]] = record
    runs = {}
    for path in sorted((scratch / "runs").glob("*/result.json")):
        run = json.loads(path.read_text())
        if "padding-probe" not in path.parent.name:
            runs[path.parent.name] = run
    expected = ["mdeberta-280m-seed0"]
    for seed in (0, 1):
        run = runs.get(f"mdeberta-280m-seed{seed}")
        if run is None:
            break
        seconds = run.get("wall_seconds_total", run.get("seconds_total", run["seconds_this_session"]))
        if seconds >= 10800:
            break
        expected.append(f"mdeberta-280m-seed{seed + 1}")
    expected += [name for name in runs if name not in expected]
    comparisons, floors = [], []
    for name in runs:
        for config in CONFIGS:
            record = records.get((name, config))
            if record is None:
                continue
            if config in ("ro-realskeleton", "pl-realskeleton"):
                metric = record["legacy"]["national_id_leakage"]
                floors.append(
                    {
                        "candidate": name,
                        "config": config,
                        "metric": metric,
                        "verdict": "unverified"
                        if not metric["decode_bearing_total"]
                        else "tie"
                        if metric["leak_rate"] == 0
                        else "loss",
                    }
                )
            for baseline in PROTOCOL["comparators"]:
                reference = records.get((baseline, config))
                if reference is None:
                    continue
                deltas = paired(record, reference)
                privacy = privacy_verdict(
                    record["residual_leakage_v2"], reference["residual_leakage_v2"], deltas["v2_all"]["ci95"]
                )
                utility = utility_verdict(deltas["over_redaction"]["ci95"])
                comparisons.append(
                    {
                        "candidate": name,
                        "config": config,
                        "comparator": baseline,
                        "deltas": deltas,
                        "privacy": privacy,
                        "utility": utility,
                        "f1_scope": record.get(
                            "f1_scope", {"full_split_available": True, "scored_documents": record["n_documents"]}
                        ),
                        "joint_numeric_gate": privacy == "win" and utility in ("win", "tie"),
                    }
                )
    missing = [
        {"name": name, "config": config}
        for name in [*expected, *PROTOCOL["comparators"]]
        for config in CONFIGS
        if (name, config) not in records
    ]
    worst = []
    for name in [*runs, *PROTOCOL["comparators"]]:
        gateway = [records[name, config] for config in PROTOCOL["gateway_configs"] if (name, config) in records]
        if gateway:
            r = max(gateway, key=lambda r: r["residual_leakage_v2"]["all_leak_rate"])
            worst.append(
                {
                    "name": name,
                    "completed_languages": len(gateway),
                    "language": r["language"],
                    "metric": r["residual_leakage_v2"],
                }
            )
    gateway_comparisons = [c for c in comparisons if c["config"] in PROTOCOL["gateway_configs"]]
    numeric_fail = any(c["privacy"] == "loss" or c["utility"] == "loss" for c in gateway_comparisons)
    anchor_fail = any(f["verdict"] == "loss" for f in floors)
    complete_gateway = set(runs) == {f"mdeberta-280m-seed{s}" for s in range(3)} and len(
        gateway_comparisons
    ) == 3 * 8 * len(PROTOCOL["comparators"])
    privacy_pass = complete_gateway and all(c["privacy"] == "win" for c in gateway_comparisons)
    paired_pass = complete_gateway and all(c["deltas"]["v2_all"]["ci95"][1] < 0 for c in gateway_comparisons)
    utility_pass = complete_gateway and all(c["utility"] in ("win", "tie") for c in gateway_comparisons)
    floors_pass = len(floors) == 6 and all(f["verdict"] == "tie" for f in floors)
    diagnostic_failures = [
        c
        for c in comparisons
        if c["config"] not in PROTOCOL["gateway_configs"] and (c["privacy"] == "loss" or c["utility"] == "loss")
    ]
    gateway_summary = []
    for comparator in PROTOCOL["comparators"]:
        cs = [c for c in gateway_comparisons if c["comparator"] == comparator]
        gateway_summary.append(
            {
                "comparator": comparator,
                "completed_seed_language_checks": len(cs),
                "required_seed_language_checks": 24,
                "privacy_passes": sum(c["privacy"] == "win" for c in cs),
                "utility_passes": sum(c["utility"] in ("win", "tie") for c in cs),
                "joint_numeric_passes": sum(c["joint_numeric_gate"] for c in cs),
            }
        )
    verdicts = [
        [
            "§1 exact residual v2",
            "measured" if records else "unverified",
            "Harness exact-character coverage; identifier subjects, not empirical attacker re-identification",
        ],
        [
            "§2 operator and gold pins",
            "recorded",
            "Raw native offsets, operator receipts and ordered gold hashes in scratch",
        ],
        [
            "§2 deployed operator parity and replacement-content attestation",
            "unverified",
            "Pinned local operator experiment; replacement fragment-copy audit and deployed parity not established",
        ],
        [
            "§3 legacy RO/PL zero floors",
            "loss" if anchor_fail else "tie" if len(floors) == 2 * len(runs) and runs else "unverified",
            "Every observed new decode-bearing national-ID leak fails; see anchor table",
        ],
        [
            "§3.1 three training seeds + disjoint Wilson",
            "pass"
            if privacy_pass
            else "loss"
            if any(c["privacy"] == "loss" for c in gateway_comparisons)
            else "unverified",
            f"{len(runs)} completed candidate seeds; each gateway seed/language/comparator must pass separately",
        ],
        [
            "§3.2 paired cluster delta",
            "pass" if paired_pass else "measured" if comparisons else "unverified",
            "2000 paired resamples, RNG125; gateway delta upper<0 required; all deltas in JSON",
        ],
        [
            "§3.3 no-worse utility + anchors",
            "pass"
            if utility_pass and floors_pass
            else "loss"
            if anchor_fail or any(c["utility"] == "loss" for c in gateway_comparisons)
            else "unverified",
            "Gateway non-PII non-whitespace character delta upper<=0, plus preserved legacy RO/PL zero floors",
        ],
        [
            "§3.4 all eight languages",
            "pass"
            if privacy_pass and utility_pass and floors_pass
            else "loss"
            if numeric_fail or anchor_fail
            else "unverified",
            "Numeric gateway gates only; reviewed/disjoint gold is the separate gate below; no pooled acceptance",
        ],
        ["§3.4 reviewed/disjoint gold", "unverified", "Native review and legacy semantic disjointness remain open"],
        [
            "§4 target-CPU short-lane load protocol",
            "unverified",
            "No open-loop three-by-ten-minute trials, frozen <=256-token replay, or 16 offered req/s load point",
        ],
        [
            "§4 p95 point and bootstrap upper <=250 ms",
            "unverified",
            "No end-to-end HTTP latency trials; training/inference runtime is not serving latency",
        ],
        ["§4 successful completed throughput >=15 req/s in every trial", "unverified", "No registered load trials"],
        ["§4 errors + timeouts + overload refusals <=0.1%", "unverified", "No offered-request failure accounting"],
        ["§4 1000 ms deadline and fail-closed dispatch", "unverified", "No gateway deadline or cancellation trials"],
        [
            "§4 zero silent bypass/truncation",
            "unverified",
            "Gateway evaluation uses full-input overflow; anchors retain 512-token cap; deployment untested",
        ],
        [
            "§4 long-input lane, admission and mixed-traffic interference",
            "unverified",
            "No separate 257-2048-token SLO or unsupported-payload refusal/load trials",
        ],
        [
            "Integration, rollback, pseudonym maps and semantic utility",
            "unverified",
            "Local detector/operator experiment only",
        ],
    ]
    output_records = []
    for record in records.values():
        public = {k: v for k, v in record.items() if k != "counts"}
        if record.get("f1_scope", {}).get("full_split_available") is False:
            public["conditional_entity_f1"] = public["entity_f1"]
            public["entity_f1"] = None
            public["bootstrap_ci95"] = {
                ("conditional_entity_f1" if k == "entity_f1" else k): v for k, v in record["bootstrap_ci95"].items()
            }
        public["legacy_support"] = {}
        for key, metric in record["legacy"].items():
            field = {
                "national_id_leakage": "decode_bearing_total",
                "name_in_context_leakage": "name_subjects_total",
            }.get(key)
            if field:
                public["legacy_support"][key] = {"available": bool(metric[field]), "n": int(metric[field])}
            else:
                public["legacy_support"][key] = {
                    group: {
                        "available": bool(metric[f"{group}_subjects_total"]),
                        "n": int(metric[f"{group}_subjects_total"]),
                    }
                    for group in ("all", "direct", "quasi")
                }
        output_records.append(public)
    prepared = {p.parent.name: json.loads(p.read_text()) for p in (scratch / "prepared").glob("*/manifest.json")}
    amendments = json.loads((ROOT / "docs/res127-training-amendments.json").read_text())
    public_training = {
        name: {
            **{k: v for k, v in run.items() if k not in {"training_metrics_not_quality", "best_dev_loss_not_quality"}},
            "training_receipt_sha256": digest(scratch / "runs" / name / "result.json"),
            "timing_basis": json.loads((scratch / "runs" / name / "timing.json").read_text()),
        }
        for name, run in runs.items()
    }
    output = {
        "protocol": PROTOCOL,
        "contract_source": contract_url,
        "training_amendments": amendments,
        "training": public_training,
        "data": prepared,
        "records": output_records,
        "comparisons": comparisons,
        "legacy_floors": floors,
        "worst_language": worst,
        "gateway_numeric_summary": gateway_summary,
        "non_gateway_diagnostic_failures": diagnostic_failures,
        "contract_verdicts": verdicts,
        "missing": missing,
        "pending_training": [name for name in expected if name not in runs],
        "not_run": {
            "xlmr-560m": "Optional architecture deferred before candidate test scoring; "
            "local compute prioritized for three mDeBERTa seeds and the complete evaluation matrix"
        },
        "status": "experimental; no gateway promotion",
        "receipt_manifest": {p.name: digest(p) for p in sorted((scratch / "results").glob("*.json"))},
    }
    if (scratch / "presidio-runtime.json").exists():
        output["presidio_runtime_inspection"] = json.loads((scratch / "presidio-runtime.json").read_text())
    if (scratch / "res126-baseline-drift.json").exists():
        output["res126_baseline_drift"] = json.loads((scratch / "res126-baseline-drift.json").read_text())
    if (scratch / "source-formatting-audit.json").exists():
        output["training_source_formatting_audit"] = json.loads((scratch / "source-formatting-audit.json").read_text())
    if (scratch / "national-id-v2-audit.json").exists():
        output["posthoc_national_id_coverage"] = json.loads((scratch / "national-id-v2-audit.json").read_text())
    output["baseline_reproduction"] = {
        name: json.loads(path.read_text())
        for name in ("presidio", "shield")
        if (path := scratch / f"{name}-gateway-parity.json").exists()
    }
    write(ROOT / "docs/res127-encoder-scorecard.json", output)
    lines = [
        "# RES-127 — gateway-trained encoder scorecard",
        "",
        "**Experimental; no gateway promotion.** Checkpoints are selected only by gateway-dev token loss.",
        "Quality numbers below come only from the EuroPriv harness. No test-based recipe or seed selection.",
        "",
        "**Full gateway entity F1 is unavailable:** distinct entities can share a whitespace token (e.g. CSV).",
        "Gateway F1 below is a conditional diagnostic on representable rows, with its n shown. Every conversation",
        "remains in residual-v2 and utility scoring. This limitation does not alter the primary endpoint.",
        "",
        "[Protocol](res127-protocol.json) · "
        "[Complete results, paired deltas and provenance](res127-encoder-scorecard.json)",
        "",
        "## Gateway numerical checks",
        "",
        "Each language has 1,000 conversations / 4,506 identifier subjects. The required checks cover three",
        "independent training seeds and eight languages separately; these counts never pool subjects or seeds.",
        "Privacy requires disjoint Wilson intervals and a paired bootstrap delta upper<0; utility requires",
        "a paired delta upper<=0. Review/disjointness and deployment gates remain separate and unverified.",
        "",
        "| Comparator | Completed / required seed-language checks | Privacy passes | Utility passes | Both |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in gateway_summary:
        lines.append(
            f"| {row['comparator']} | {row['completed_seed_language_checks']} / 24 | "
            f"{row['privacy_passes']} | {row['utility_passes']} | {row['joint_numeric_passes']} |"
        )
    lines += [
        "",
        "## Training and measurement",
        "",
        "Training: all 40,000 gateway TRAIN rows plus the original 150,000 RO/EN/PL rows; batch 16, fp32, MPS,",
        "max length 512, LoRA rank 16, three epochs, length-grouped batches, padding rounded to 32 tokens,",
        "non-reentrant activation checkpointing and cache clearing every 25 steps.",
        "[Training memory amendment](res127-training-amendments.json); gateway dev: 1,600 conversations.",
        "Each 1,000-step checkpoint is eligible by dev loss; the best eligible checkpoint per seed is scored below.",
        "The final partial interval is not an eligible selection checkpoint. Dev loss is not a quality claim.",
        "",
        "Gateway: 512-token windows, stride 64, full conversations. Anchors: frozen RES-125 capped inference",
        "(512 tokens, stride 0), with unchanged matched kp-280m/Presidio/Shield receipts rescored by this harness.",
        "Gold remains complete for capped anchors. Gateway KP/stock operators mask the union of native offsets;",
        "Shield uses its pinned native replacement service. These are local experiments, "
        "not deployed gateway behavior.",
        "",
        "95% Wilson intervals describe identifier subjects; F1 and utility CIs use 2,000 whole-document/conversation",
        "bootstrap draws (RNG125). Comparative deltas use paired draws. Subjects and seeds are never pooled.",
        "Translations share families across languages; compare each language separately. No aggregate language win.",
        "",
        "| Candidate | Seed | Total elapsed seconds (including queue) | Resume | Selected checkpoint |",
        "|---|---:|---:|---|---|",
    ]
    for name, run in runs.items():
        seconds = run.get("wall_seconds_total", run.get("seconds_total", run["seconds_this_session"]))
        lines.append(
            f"| {name} | {run['seed']} | {seconds:.1f} | {bool(run['resumed_from'])} | "
            f"{Path(run['best_checkpoint']).name} |"
        )
    lines += [
        "",
        "## Harness results",
        "",
        "Each candidate row is one independently trained checkpoint; kp-280m is the single shipped checkpoint",
        "(its independent training seed is unknown). Presidio/Shield are one inference run per configuration.",
        "",
        "| Config | Model | Docs | Identifier subjects | F1 [bootstrap CI] | "
        "Residual v2 [Wilson CI] | Over-redaction [bootstrap CI] |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for config in CONFIGS:
        for name in [*runs, *PROTOCOL["comparators"]]:
            r = records.get((name, config))
            if r is None:
                lines.append(f"| {config} | {name} | — | — | unverified | unverified | unverified |")
                continue
            m = r["residual_leakage_v2"]
            f1 = interval(r["entity_f1"]["f1"], r["bootstrap_ci95"]["entity_f1"])
            if "f1_scope" in r:
                f1 += f" (subset n={r['f1_scope']['scored_documents']})"
            v2 = interval(m["all_leak_rate"], [m["all_leak_rate_ci_low"], m["all_leak_rate_ci_high"]])
            utility = interval(r["utility"]["over_redaction"], r["bootstrap_ci95"]["over_redaction"])
            lines.append(
                f"| {config} | {name} | {r['n_documents']} | {int(m['all_subjects_total'])} | {f1} | {v2} | {utility} |"
            )
    lines += [
        "",
        "## Worst gateway language",
        "",
        "| Model | Completed languages | Worst language | All-PII residual v2 [Wilson CI] |",
        "|---|---:|---|---|",
    ]
    for row in worst:
        m = row["metric"]
        value = interval(m["all_leak_rate"], [m["all_leak_rate_ci_low"], m["all_leak_rate_ci_high"]])
        lines.append(f"| {row['name']} | {row['completed_languages']} | {row['language']} | {value} |")
    lines += [
        "",
        "## Legacy national-ID anchors",
        "",
        "| Candidate | Config | Subjects | Leak rate [Wilson CI] | Verdict |",
        "|---|---|---:|---|---|",
    ]
    for row in floors:
        m = row["metric"]
        value = interval(m["leak_rate"], [m["leak_rate_ci_low"], m["leak_rate_ci_high"]])
        lines.append(
            f"| {row['candidate']} | {row['config']} | {m['decode_bearing_total']} | {value} | {row['verdict']} |"
        )
    audit = output.get("posthoc_national_id_coverage")
    if audit:
        lines += ["", "## Post-hoc exact-character national-ID diagnostic", "",
                  "This supplementary check was added after scoring to distinguish legacy any-overlap detection",
                  "from complete coverage. It does not change model selection, primary gold or acceptance endpoints.",
                  "Both scopes retain all 1,500 documents and the same native intervals; rates use 95% Wilson CIs.",
                  "The validator-valid scope uses the pinned legacy CNP/PESEL validators across all gold labels.",
                  "The broader NATIONAL_ID scope includes every span with that gold label, including other values.", ""]
        names = [*runs, *PROTOCOL["comparators"]]
        headers = ["Config / scope", "Subjects", *[n.removeprefix("mdeberta-280m-") for n in names]]
        lines += ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        groups = sorted({(r["config"], r["scope"]) for r in audit["results"]})
        for config, scope in groups:
            rs = {r["name"]: r for r in audit["results"] if r["config"] == config and r["scope"] == scope}
            n = int(next(iter(rs.values()))["residual_leakage_v2"]["all_subjects_total"])
            cells = [f"{config} / {scope}", str(n)]
            for name in names:
                m = rs[name]["residual_leakage_v2"]
                cells.append(interval(m["all_leak_rate"], [m["all_leak_rate_ci_low"], m["all_leak_rate_ci_high"]]))
            lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Non-gateway diagnostic losses",
        "",
        "These regressions limit general-domain conclusions despite any gateway improvements.",
        "Per-seed estimates, n and CIs appear above; all paired deltas are in JSON.",
        "",
        "| Config | Comparator | Seeds with privacy loss | Seeds with utility loss |",
        "|---|---|---|---|",
    ]
    for config, comparator in sorted({(c["config"], c["comparator"]) for c in diagnostic_failures}):
        cs = [c for c in diagnostic_failures if c["config"] == config and c["comparator"] == comparator]
        privacy_seeds = [str(runs[c["candidate"]]["seed"]) for c in cs if c["privacy"] == "loss"]
        utility_seeds = [str(runs[c["candidate"]]["seed"]) for c in cs if c["utility"] == "loss"]
        lines.append(
            f"| {config} | {comparator} | {', '.join(privacy_seeds) or '—'} | {', '.join(utility_seeds) or '—'} |"
        )
    lines += [
        "",
        "## Contract verdicts",
        "",
        f"Source: [frozen gateway contract v1]({contract_url}).",
        "",
        "| Contract line | Verdict | Evidence / limit |",
        "|---|---|---|",
    ]
    lines += [f"| {line} | {verdict} | {reason} |" for line, verdict, reason in verdicts]
    lines += [
        "",
        "The new-configuration numeric gates above concern gateway traffic and the legacy national-ID floors.",
        f"TAB/general/anchor diagnostics additionally contain {len(diagnostic_failures)} seed/config/comparator",
        "comparisons with a privacy or utility loss; every such row and its paired CI is retained in JSON.",
        "These diagnostics remain limitations, not gateway-wide or general-domain acceptance claims.",
    ]
    lines += [
        "",
        "A numeric per-checkpoint privacy win requires disjoint Wilson intervals AND paired delta upper<0.",
        "A privacy tie means superiority was not established; it is not a statistical equivalence finding.",
        "A scoped contract win additionally requires all three seeds, no-worse utility, preserved anchors and",
        "reviewed/disjoint gold. Missing review alone prevents a scoped win. Every DIRECT/QUASI/UNSPECIFIED",
        "slice and comparator delta is retained in JSON; empty support is unavailable, never a measured zero.",
        "",
        "## Limitations and reproduction",
        "",
        f"Missing model/config rows: {len(missing)}. Synthetic gold has no independent native review. The gateway",
        "TRAIN audit covers new train versus dev/test; legacy-corpus semantic disjointness is unverified.",
        "All original KP training rows remain; the general configurations are not certified clean held-out data.",
        "Training truncates at 512; gateway test inference screens overflow. The train export uses two newlines",
        "between parts; the harness uses its frozen single-newline coordinate policy on dev/test.",
        "No baseline repeat-variance, all-document CPU/MPS equivalence, serving latency, "
        "rollback or semantic-utility verification.",
        "",
        "Seed 0 resumed after execution-profile changes; equality to an uninterrupted "
        "training trajectory was not tested.",
        "The exact executed trainer is archived in scratch; the committed copy has an AST-identical line wrap.",
        "The JSON source-formatting audit records both hashes and the archived source "
        "needed for the original resume pins.",
        "",
        "Optional XLM-R-560M was deferred before candidate test scoring to complete the mDeBERTa replication",
        "and evaluation matrix. No XLM-R result or cross-architecture conclusion is claimed.",
        "",
        "Run `scripts/train_gateway_res127.py prepare` then `train` with pinned local base, data and scratch paths.",
        "Use the RES-126 harness on PYTHONPATH; `scripts/score_gateway_res127.py freeze` imports frozen RES-125",
        "anchors and loads local gateway test through the RES-126 override. `predict` scores each selected checkpoint",
        "and each gateway comparator; `scripts/report_gateway_res127.py` assembles this report without selection.",
        "`scripts/audit_national_ids_res127.py` replays the optional post-hoc national-ID coverage diagnostic.",
        "Exact commands, logs, checkpoints and native receipts are retained in the assigned RES-127 scratch directory.",
    ]
    (ROOT / "docs/res127-encoder-scorecard.md").write_text("\n".join(lines) + "\n")
    print(f"REPORT candidates={len(runs)} results={len(records)} missing={len(missing)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", required=True, type=Path)
    build(parser.parse_args().scratch)

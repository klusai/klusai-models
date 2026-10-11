"""Assemble immutable RES-125 receipts into a scorecard; no inference or model selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scorecard_res125 import PROTOCOL, ROOT, ci, digest, ratio_samples, write

FAMILIES = {
    "kp-280m": ["kp-280m"],
    "560m-legal": [f"560m-legal-s{s}" for s in range(3)],
    "560m-tab": [f"560m-tab-s{s}" for s in range(3)],
}
COMPARATORS = ("presidio", "shield")


def privacy_verdict(candidate, comparator, delta_ci, group="all"):
    """Contract's disjoint Wilson AND paired-bootstrap rule; no zero-floor wins."""
    if not candidate[f"{group}_available"] or not comparator[f"{group}_available"] or delta_ci is None:
        return "unverified"
    if (candidate[f"{group}_leak_rate_ci_high"] < comparator[f"{group}_leak_rate_ci_low"]
            and delta_ci[1] < 0):
        return "win"
    if (candidate[f"{group}_leak_rate_ci_low"] > comparator[f"{group}_leak_rate_ci_high"]
            and delta_ci[0] > 0):
        return "loss"
    return "tie"  # no established superiority; not statistical equivalence


def utility_verdict(delta_ci):
    if delta_ci is None:
        return "unverified"
    if delta_ci[1] <= 0:
        return "tie" if delta_ci == [0, 0] else "win"
    return "loss" if delta_ci[0] > 0 else "unverified"


def paired(candidate, reference):
    if candidate["gold_sha256"] != reference["gold_sha256"]:
        raise ValueError("Cannot pair different ordered gold rows")
    n = candidate["n_documents"]
    indices = np.random.default_rng(125).integers(n, size=(2000, n))
    out = {}
    for key, counts in candidate["counts"].items():
        ref = reference["counts"][key]
        if key != "entity_f1" and [c[1] for c in counts] != [c[1] for c in ref]:
            raise ValueError(f"Gold-defined denominators differ: {key}")
        a = ratio_samples(counts, indices, f1=key == "entity_f1")
        b = ratio_samples(ref, indices, f1=key == "entity_f1")
        totals_a, totals_b = np.asarray(counts).sum(axis=0), np.asarray(ref).sum(axis=0)
        if key == "entity_f1":
            point = candidate["entity_f1"]["f1"] - reference["entity_f1"]["f1"]
        else:
            point = float((totals_a[0] - totals_b[0]) / totals_a[1]) if totals_a[1] else None
        out[key] = {"candidate_minus_comparator": point, "ci95": ci(a - b)}
    if "over_redaction" in out:
        utility = out["over_redaction"]
        point, bounds = utility["candidate_minus_comparator"], utility["ci95"]
        out["retention"] = {"candidate_minus_comparator": -point if point is not None else None,
                            "ci95": [-bounds[1], -bounds[0]] if bounds is not None else None}
    return out


def acceptance(records, deltas, config, family, reference):
    names = FAMILIES[family]
    available = [records.get((name, config)) for name in names]
    base = records.get((reference, config))
    result = {"config": config, "candidate": family, "comparator": reference,
              "expected_seeds": len(names), "available_checkpoints": sum(r is not None for r in available),
              "n_documents": base["n_documents"] if base else None, "per_seed": []}
    if base is None or any(r is None for r in available):
        return {**result, "overall": "unverified", "reason": "Missing completed candidate/comparator receipt"}
    for record in available:
        delta = deltas[(record["name"], config, reference)]
        seed = {"name": record["name"], "all_v2": privacy_verdict(
            record["residual_leakage_v2"], base["residual_leakage_v2"], delta["v2_all"]["ci95"]),
            "utility": utility_verdict(delta["over_redaction"]["ci95"])}
        if config == "tab-echr-legal-en":
            seed["tab_direct_v2"] = privacy_verdict(
                record["residual_leakage_v2"], base["residual_leakage_v2"], delta["v2_direct"]["ci95"], "direct")
        if config in ("ro-realskeleton", "pl-realskeleton"):
            leak = record["legacy"]["national_id_leakage"]
            seed["legacy_zero_floor"] = "tie" if leak["leak_rate"] == 0 else "loss"
        result["per_seed"].append(seed)
    result["three_training_seeds"] = "unverified" if len(names) < 3 else "win"
    # The existing TAB split is document-disjoint, but cross-case subject/near-duplicate
    # disjointness remains unaudited; synthetic configs have additional native-review gates.
    result["reviewed_disjoint_gold"] = "unverified"
    result["gold_limitation"] = (
        "TAB document-disjoint; cross-case subject/near-duplicate disjointness not established"
        if config == "tab-echr-legal-en" else "dev synthetic gold; reviewed/disjoint gateway gold unavailable"
    )
    fail = any(s["all_v2"] == "loss" or s["utility"] == "loss" or s.get("legacy_zero_floor") == "loss"
               for s in result["per_seed"])
    result["overall"] = "loss" if fail else "unverified"
    result["reason"] = "Observed privacy/utility/anchor failure" if fail else "Contract gates not all established"
    return result


def interval(point, bounds):
    if point is None or bounds is None:
        return "unavailable"
    return f"{point:.4f} [{bounds[0]:.4f}, {bounds[1]:.4f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    records = {}
    for path in sorted((args.scratch / "results").glob("*.json")):
        result = json.loads(path.read_text())
        result["config_status"] = "real-external-gold" if result["config"] == "tab-echr-legal-en" else "dev"
        result["training_contamination_audit"] = "not established across all subject/scenario/translation families"
        result["checkpoint_count"] = 1 if result["identity"]["kind"] == "kp" else 0
        result["independent_training_seed_count"] = 1 if result["identity"].get("training_seed") is not None else None
        result["torch_inference_seed"] = 125
        result["completed_inference_runs"] = 1
        bounds = result["bootstrap_ci95"]["over_redaction"]
        result["utility"]["over_redaction_ci95"] = bounds
        result["utility"]["retention_ci95"] = [1 - bounds[1], 1 - bounds[0]] if bounds is not None else None
        result["legacy_support"] = {}
        for key, metric in result["legacy"].items():
            if key == "national_id_leakage":
                result["legacy_support"][key] = {"available": bool(metric["decode_bearing_total"]),
                                                  "n": int(metric["decode_bearing_total"])}
            elif key == "name_in_context_leakage":
                result["legacy_support"][key] = {"available": bool(metric["name_subjects_total"]),
                                                  "n": int(metric["name_subjects_total"])}
            else:
                result["legacy_support"][key] = {
                    group: {"available": bool(metric[f"{group}_subjects_total"]),
                            "n": int(metric[f"{group}_subjects_total"])} for group in ("all", "direct", "quasi")
                }
        records[(result["name"], result["config"])] = result
    deltas = {}
    comparisons = []
    for (name, config), result in records.items():
        if name in COMPARATORS:
            continue
        for comparator in COMPARATORS:
            if (comparator, config) not in records:
                continue
            delta = paired(result, records[(comparator, config)])
            deltas[name, config, comparator] = delta
            comparisons.append({"candidate": name, "config": config, "comparator": comparator, "deltas": delta})
    verdicts = [acceptance(records, deltas, config, family, comparator)
                for config in PROTOCOL["configs"] for family in FAMILIES for comparator in COMPARATORS]
    aggregates = []
    for config in PROTOCOL["configs"]:
        for family, names in FAMILIES.items():
            rs = [records[(name, config)] for name in names if (name, config) in records]
            if not rs:
                continue
            stats = {}
            for metric, values in {
                "f1": [r["entity_f1"]["f1"] for r in rs],
                "v2_all": [r["residual_leakage_v2"]["all_leak_rate"] for r in rs],
                "v2_direct": [r["residual_leakage_v2"]["direct_leak_rate"] for r in rs
                              if r["residual_leakage_v2"]["direct_available"]],
                "over_redaction": [r["utility"]["over_redaction"] for r in rs
                                   if r["utility"]["over_redaction"] is not None],
            }.items():
                stats[metric] = {"mean": float(np.mean(values)), "range": [min(values), max(values)]} if values else None
            aggregates.append({"config": config, "family": family, "checkpoint_count": len(rs),
                               "training_seeds": [r["identity"].get("training_seed") for r in rs], "metrics": stats})
    tab = [records[(n, "tab-echr-legal-en")] for n in FAMILIES["560m-tab"]
           if (n, "tab-echr-legal-en") in records]
    best = max(tab, key=lambda r: r["entity_f1"]["f1"])["name"] if tab else None
    best_residual = min(tab, key=lambda r: r["residual_leakage_v2"]["all_leak_rate"])["name"] if tab else None
    expected = [(n, c) for c in PROTOCOL["configs"] for n in [*sum(FAMILIES.values(), []), *COMPARATORS]]
    missing = [{"model": n, "config": c, "status": "unverified", "reason": "No complete receipt"}
               for n, c in expected if (n, c) not in records]
    eight_language = ("ro-realskeleton", "en", "de", "fr", "es", "it", "nl", "pl-realskeleton")
    worst_language = []
    for name in [*sum(FAMILIES.values(), []), *COMPARATORS]:
        rs = [records[(name, c)] for c in eight_language if (name, c) in records]
        if rs:
            worst = max(rs, key=lambda r: r["residual_leakage_v2"]["all_leak_rate"])
            worst_language.append({"model": name, "completed_languages": len(rs), "language": worst["language"],
                                   "config": worst["config"], "all_v2": worst["residual_leakage_v2"],
                                   "scope": "worst frozen core config; not representative gateway traffic"})
    training = []
    for path in sorted((args.scratch / "runs").glob("*-train.json")):
        record = json.loads(path.read_text())
        training.append({k: v for k, v in record.items() if k not in {"log_history", "dev_selection_loss"}})
    prior_tab = json.loads((ROOT / "docs/res104-tab-results.json").read_text())["results"]
    reproduction = []
    for family, prior in (("560m-legal", "cjeu"), ("560m-tab", "tab")):
        for seed in range(3):
            name = f"{family}-s{seed}"
            current = records.get((name, "tab-echr-legal-en"))
            if current is not None:
                old = prior_tab[f"{prior}-seed{seed}"]
                reproduction.append({"name": name,
                                     "same_weights_as_res104": current["identity"]["files"]["model.safetensors"]
                                     == old["weights_sha256"],
                                     "f1_delta_from_res104": current["entity_f1"]["f1"] - old["entity_f1"]["f1"]})
    result = {
        "schema": "res125-candidate-scorecard/v1", "issue": "RES-125", "freeze_date": "2026-10-11",
        "protocol": PROTOCOL,
        "protocol_sha256": digest(ROOT / "docs/res125-protocol.json"),
        "best_tab_seed": {"name": best, "selection": "post-hoc highest TAB test F1, descriptive only"},
        "best_tab_seed_by_primary_v2": {"name": best_residual,
                                        "selection": "post-hoc lowest TAB all-PII residual v2, descriptive only"},
        "results": [{k: v for k, v in r.items() if k != "counts"} for r in records.values()],
        "seed_spread": aggregates, "paired_comparisons": comparisons, "acceptance": verdicts, "failures": missing,
        "acceptance_failures": [{k: v[k] for k in ("config", "candidate", "comparator", "reason")}
                                for v in verdicts if v["overall"] == "loss"],
        "worst_core_language_per_checkpoint": worst_language, "rebuilt_training_provenance": training,
        "training_manifest_sha256": digest(args.scratch / "tab-manifest.json"),
        "inference_amendment": json.loads((args.scratch / "inference-amendment.json").read_text()),
        "device_handoff": (json.loads((args.scratch / "device-handoff.json").read_text())
                           if (args.scratch / "device-handoff.json").exists() else []),
        "stock_presidio_runtime": json.loads((args.scratch / "runtime-presidio.json").read_text()),
        "historical_presidio_f1_audit": json.loads((args.scratch / "historical-presidio-audit.json").read_text()),
        "res104_reproduction_audit": reproduction,
        "gateway_verdict": "unverified",
        "gateway_acceptance": [
            {"line": "reviewed disjoint gateway gold for all required languages", "verdict": "unverified",
             "required_languages": ["ro", "en", "de", "fr", "es", "it", "nl", "pl"],
             "reviewed_gateway_conversations": 0, "note": "Frozen benchmark rows are not gateway conversations"},
            {"line": "p95 <=250 ms at >=15 successful req/s", "verdict": "unverified",
             "registered_load_trials": 0, "requests_in_registered_load_trials": 0, "ci95": None},
            {"line": "deployed KP operator / Shield production asset parity", "verdict": "unverified"},
            {"line": "gateway integration, rollback and pseudonym restoration", "verdict": "unverified"},
        ],
        "not_verified": ["Gateway conversation gold and all-eight-language reviewed acceptance",
                         "Subject/scenario/translation disjointness for all candidates and corpora",
                         "Three independent shipped-280M training seeds (v2 seeds are a different recipe)",
                         "Baseline inference-repeat variability (one fresh run/config; historical F1 drift is recorded)",
                         "All-document CPU/MPS bitwise equivalence; per-config device is recorded, fp32 throughout",
                         "Deployed KP/stock operator and Shield production asset parity",
                         "No-truncation long-input service behavior (KP pipeline truncates at 512 tokens)",
                         "End-to-end p95 <=250ms at >=15 req/s; no HTTP load acceptance trials",
                         "Semantic utility, pseudonym restoration, gateway integration and rollback"],
        "sources": [{"url": "https://huggingface.co/datasets/klusai/europriv-bench", "license": "CC-BY-4.0; TAB MIT"},
                    {"url": "https://github.com/NorskRegnesentral/text-anonymization-benchmark",
                     "license": "MIT", "manifest": "res104-tab-manifest.json"},
                    {"url": "https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1m",
                     "license": "CC-BY-4.0", "attribution": "Ai4Privacy / Ai Suisse SA"}],
    }
    write(ROOT / "docs/res125-candidate-scorecard.json", result)
    decisions = []
    if best is not None and best_residual is not None and best != best_residual:
        cells = []
        for name in (best, best_residual):
            row = records[(name, "tab-echr-legal-en")]
            metric = row["residual_leakage_v2"]
            bounds = [metric["all_leak_rate_ci_low"], metric["all_leak_rate_ci_high"]]
            cells.append(f"`{name}` {interval(metric['all_leak_rate'], bounds)}")
        decisions.append(f"- Best F1 and lowest all-PII residual differ on TAB: residual v2 is {cells[0]} "
                         f"versus {cells[1]}. These are post-hoc descriptions, with "
                         f"{row['n_documents']} documents / {int(metric['all_subjects_total'])} subjects per seed "
                         "and Wilson CIs; all three seeds remain in acceptance.")
    for family in FAMILIES:
        for comparator in COMPARATORS:
            verdict = next(v for v in verdicts if v["config"] == "tab-echr-legal-en"
                           and v["candidate"] == family and v["comparator"] == comparator)
            seeds = verdict["per_seed"]
            if seeds:
                decisions.append(f"- TAB `{family}` vs `{comparator}`: all-PII v2 by checkpoint = "
                                 f"{', '.join(s['all_v2'] for s in seeds)}; utility = "
                                 f"{', '.join(s['utility'] for s in seeds)}; full gate **{verdict['overall']}**.")
    for config in ("ro-realskeleton", "pl-realskeleton"):
        for name in FAMILIES["560m-legal"]:
            row = records.get((name, config))
            if row is not None:
                leak = row["legacy"]["national_id_leakage"]
                if leak["decode_bearing_missed"]:
                    bounds = [leak["leak_rate_ci_low"], leak["leak_rate_ci_high"]]
                    decisions.append(f"- **Legacy anchor failure:** `{name}` / `{config}` leaks "
                                     f"{int(leak['decode_bearing_missed'])}/{int(leak['decode_bearing_total'])} "
                                     f"national-ID subjects, {interval(leak['leak_rate'], bounds)} (Wilson).")
    lines = ["# RES-125 — frozen candidate scorecard", "",
             "**Gateway acceptance: unverified.** Exact experimental operator receipts are scored on identical "
             "ordered gold for every completed row. Observed failures remain failures; no gateway promotion follows.",
             "", "Machine-readable [scorecard](res125-candidate-scorecard.json) and "
             "[pre-scoring protocol](res125-protocol.json). Rates are fractions; confidence intervals are 95%. "
             "Training-seed ranges are descriptive.",
             "", *decisions,
             "", "## Frozen scope and interpretation", "",
             "Harness `4e1d64097ff761db0256ad4f3f523c42a071ccc4`; core dataset "
             "`58faa4fa572cd8064818a6e1f8ca6e361b959c00`. OpenPII uses the merged RES-101 pinned "
             "validation loader: ten languages, 3,985 retained rows after 15 token-collision exclusions. "
             "Each JSON row records the ordered gold hash, dataset provenance, checkpoint/tokenizer hashes and runtime.",
             "", "KP and stock Presidio use a newly frozen local literal block-mask operator on native detector "
             "offsets. It executes before scoring and never expands partial tokens. Shield uses its pinned native "
             "DEFAULT replace service, including native types outside the detection crosswalk. These are experimental "
             "operator results, not a deployed KP gateway. Legacy F1/leakage retains the harness label mask; v2 and "
             "utility use all actual operator intervals without that mask. V1 and v2 are therefore not nested: "
             "stricter character coverage can still yield a lower rate when an out-of-gold predicted type was "
             "discarded by the legacy label filter but actually removed by the operator.",
             "", "Before any KP receipt or score, an unbounded shipped-tokenizer run was stopped and a "
             "512-token maximum (stride 0) was set for every KP candidate. The amendment is recorded in JSON. "
             "Full documents remain in gold; this is a frozen capped-inference experiment, including all "
             "truncation failures. Stock and Shield retain their native full-document processing.",
             "", "Encoder inference uses CPU or MPS fp32, recorded per result. Shared-host CPU contention led "
             "to a handoff after training: completed CPU configs were retained; incomplete configs were restarted "
             "in full on MPS with the same checkpoints and tokenizers, without mixing devices inside a receipt. "
             "Device equivalence over all documents is not asserted. The result is conditional on these frozen "
             "runtime receipts; no load/latency inference follows from wall time.",
             "", "Wilson intervals describe gold subjects. Comparative intervals use 2,000 paired whole-document "
             "bootstrap draws, RNG 125, clustering all subjects within each document. Utility is the fraction of gold "
             "non-PII non-whitespace characters removed; retention is its complement. Empty denominators are "
             "unavailable. Without explicit entity IDs, v2 counts exact-value identifier subjects, not unique people; "
             "missing sensitivity remains UNSPECIFIED. This measures residual identifiers, not attacker "
             "re-identification probability. Bootstrap CIs condition on each checkpoint; training seeds are never "
             "pooled into subject n.",
             "", "Stock EuroPriv Presidio runs English at threshold 0 with the benchmark adapter's entity "
             "whitelist/crosswalk. Shield uses EN/NL at 0.8, with Dutch fallback for "
             "other languages; its service pin is `763956e6da84eb755c168acf13c5d10b9239ca5c5415a8a30b4c806c8c55dd1f`. "
             "They remain separate comparators, with one inference run per config; baseline repeat variance is "
             "unverified. Small stock-Presidio F1 differences from the historical board are recorded in JSON; "
             "only fresh matched receipts enter comparisons. All source data is existing MIT/CC-BY-4.0 material; source URLs and "
             "licences are in JSON. No Piiranha training, paid APIs or cloud resources.",
             "", "The shipped 280M is one checkpoint with unknown independent training seed; KLU-106 v2 seeds "
             "cannot be relabeled as replicas of it. Its old full-data run took under two hours, but original "
             "base/data revisions and initialization state are not pinned, and the current trainer changes its "
             "historical split. A faithful additional shipped-recipe seed was not established; this requirement "
             "remains unverified rather than treating a changed recipe as replication. "
             "Legal seeds 0/1/2 are existing CJEU checkpoints. TAB seeds "
             "0/1/2 were rebuilt sequentially from the registered RES-104 train/dev recipe after the old worktree "
             "checkpoints were removed (prior runtime <2h/seed). TAB train/dev and test documents are disjoint; "
             "subject and near-duplicate passage disjointness is not established.",
             "", f"Descriptive best TAB-trained seed: **{best}** (highest TAB test F1; post-hoc, not an "
             f"acceptance selector). Lowest TAB all-PII residual v2: **{best_residual}**, also descriptive only. "
             "The representative table uses the F1-best seed across configs, plus legal seed0. Every seed's "
             "point estimate, Wilson/bootstrap CI, DIRECT/QUASI/UNSPECIFIED support and paired delta is in JSON.",
             "", "## Representative rows", "",
             "Legacy = TAB DIRECT any-overlap on TAB, decode-bearing national-ID on RO/PL real-skeleton, "
             "and name-in-context elsewhere (none is all-PII). V2 = all protected identifier subjects. "
             "All applicable legacy metrics and full TAB slices are also in JSON.",
             "", "| Config | Model | Docs / v2 subjects | F1 [CI] | Legacy [Wilson CI]; n | V2 all [Wilson CI] | Over-redaction [CI] |",
             "|---|---|---:|---|---|---|---|"]
    for config in PROTOCOL["configs"]:
        for name in ("kp-280m", "560m-legal-s0", best, *COMPARATORS):
            r = records.get((name, config))
            if not r:
                lines.append(f"| {config} | {name} | — | unverified | unverified | unverified | unverified |")
                continue
            v2 = r["residual_leakage_v2"]
            if config == "tab-echr-legal-en":
                legacy = r["legacy"]["tab_reid_leakage"]
                prefix, denom = "direct_", "direct_subjects_total"
            elif config in ("ro-realskeleton", "pl-realskeleton"):
                legacy = r["legacy"]["national_id_leakage"]
                prefix, denom = "", "decode_bearing_total"
            else:
                legacy = r["legacy"]["name_in_context_leakage"]
                prefix, denom = "name_", "name_subjects_total"
            legacy_text = (interval(legacy[prefix + "leak_rate"],
                                   [legacy[prefix + "leak_rate_ci_low"], legacy[prefix + "leak_rate_ci_high"]])
                           + f"; {int(legacy[denom])}" if legacy[denom] else "unavailable; 0")
            lines.append(f"| {config} | {name} | {r['n_documents']} / {int(v2['all_subjects_total'])} | "
                         f"{interval(r['entity_f1']['f1'], r['bootstrap_ci95']['entity_f1'])} | {legacy_text} | "
                         f"{interval(v2['all_leak_rate'], [v2['all_leak_rate_ci_low'], v2['all_leak_rate_ci_high']])} | "
                         f"{interval(r['utility']['over_redaction'], r['bootstrap_ci95']['over_redaction'])} |")
    lines += ["", "## Training seed spread", "", "Ranges describe seed variability, not confidence intervals.", "",
              "| Config | Candidate | Checkpoints | F1 mean [min,max] | V2 all mean [min,max] |",
              "|---|---|---:|---|---|"]
    for agg in aggregates:
        if agg["family"] == "kp-280m":
            continue
        f1, v2 = agg["metrics"]["f1"], agg["metrics"]["v2_all"]
        lines.append(f"| {agg['config']} | {agg['family']} | {agg['checkpoint_count']} | "
                     f"{interval(f1['mean'], f1['range'])} | {interval(v2['mean'], v2['range'])} |")
    lines += ["", "## Contract verdicts", "",
              "A privacy win requires disjoint v2 Wilson intervals **and** a paired delta upper bound <0 "
              "for all three candidate seeds, utility delta upper bound ≤0, preserved RO/PL legacy zero floors, "
              "and reviewed disjoint gold. `tie` means no established superiority, never statistical equivalence. "
              "Missing evidence means `unverified`; significant privacy/utility harm or a new anchor leak means `loss`.",
              "", "Acceptance lines below are in seed order (0/1/2; one shipped checkpoint for 280M). "
              "V2 columns are per-checkpoint statistical comparisons; the final column also applies utility, "
              "anchors, three-seed and gold gates. Reviewed gateway gold is unverified throughout. "
              "`—` means that acceptance line does not apply to the config.",
              "", "| Config | Candidate | Comparator | V2 all | TAB DIRECT v2 | Utility | Legacy zero floor | Overall |",
              "|---|---|---|---|---|---|---|---|"]
    for verdict in verdicts:
        cells = []
        for key in ("all_v2", "tab_direct_v2", "utility", "legacy_zero_floor"):
            values = [seed[key] for seed in verdict["per_seed"] if key in seed]
            cells.append(" / ".join(values) if values else "—")
        lines.append(f"| {verdict['config']} | {verdict['candidate']} | {verdict['comparator']} | "
                     f"{' | '.join(cells)} | {verdict['overall']} |")
    lines += ["", "| Gateway acceptance line | Verdict | Reason |", "|---|---|---|",
              "| Eight-language reviewed held-out privacy + utility | unverified | Dev/split gates remain; see failures above |",
              "| Three shipped-280M training seeds | unverified | Only one shipped-recipe checkpoint; v2 is distinct |",
              "| p95 ≤250 ms at ≥15 successful req/s | unverified | No contract HTTP/load trials |",
              "| Gateway integration, rollback, pseudonym maps | unverified | Detector/operator experiment only |",
              "", "## Worst language on the eight core configs", "",
              "These are heterogeneous benchmark strata, not eight-language gateway traffic. "
              "Worst means highest all-PII residual-v2 point estimate; JSON includes all support slices.", "",
              "| Checkpoint | Completed languages | Worst language | V2 all [Wilson CI] |",
              "|---|---:|---|---|"]
    for worst in worst_language:
        metric = worst["all_v2"]
        bounds = [metric["all_leak_rate_ci_low"], metric["all_leak_rate_ci_high"]]
        lines.append(f"| {worst['model']} | {worst['completed_languages']} | {worst['language']} | "
                     f"{interval(metric['all_leak_rate'], bounds)} |")
    lines += ["", "## Failures and limits", ""]
    if missing:
        lines += [f"- Missing completed receipt: `{r['model']}` / `{r['config']}`." for r in missing]
    else:
        lines += ["All 171 expected model/config runs completed; no configuration or seed was dropped."]
    lines += ["", *[f"- {limit}." for limit in result["not_verified"]], "",
              "KP uses the explicit 512-token truncation; full input rows and gold remain in "
              "every denominator. Shield's native chunking remains part of its frozen service. This may explain "
              "failures but does not remove them. The older TAB any-overlap win must not be called a v2 win.",
              "", "Reproduce with `scripts/scorecard_res125.py freeze`, `predict` per pinned candidate/baseline, "
              "then `scripts/report_res125.py --scratch <scratch>`; use the pinned harness on PYTHONPATH. "
              "Raw private receipts and logs live in the assigned RES-125 scratch directory; only aggregate "
              "scores, provenance and sufficient reproduction code are committed.", ""]
    (ROOT / "docs/res125-candidate-scorecard.md").write_text("\n".join(lines))
    print(f"Frozen {len(records)}/{len(expected)} rows; best TAB seed={best}; missing={len(missing)}")


if __name__ == "__main__":
    main()

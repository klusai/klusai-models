"""RES-127 frozen gold, resumable receipts and harness scoring; no model selection."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
from pathlib import Path

from score_res127_metrics import score_one
from scorecard_res125 import digest, literal_mask, write

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "docs/res127-protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())
CONFIGS = PROTOCOL["gateway_configs"] + PROTOCOL["anchor_configs"]


def freeze(args):
    from europriv_bench.gateway import load_gateway
    from europriv_bench.spec import EvalSpec

    for config in CONFIGS:
        target = args.scratch / "gold" / f"{config}.json"
        if config.startswith("gateway-"):
            lang = config.removeprefix("gateway-")
            spec_path = args.bench / "evaluations" / f"pii-detection-gateway-traffic-{lang}.yaml"
            spec = EvalSpec.from_yaml(spec_path)
            rows = load_gateway(spec)
            value = {"config": config, "language": lang, "dataset": rows.source_metadata,
                     "spec_sha256": digest(spec_path), "rows": rows}
            if target.exists() and json.loads(target.read_text()) != value:
                raise ValueError("Frozen gateway gold changed")
            write(target, value)
        else:
            source = args.prior / "gold" / f"{config}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and digest(target) != digest(source):
                raise ValueError("Frozen anchor gold changed")
            shutil.copyfile(source, target)
            for name in PROTOCOL["comparators"]:
                source = args.prior / "predictions" / f"{name}--{config}.json"
                target_pred = args.scratch / "predictions" / source.name
                target_pred.parent.mkdir(parents=True, exist_ok=True)
                if target_pred.exists() and digest(target_pred) != digest(source):
                    raise ValueError("Frozen comparator receipt changed")
                shutil.copyfile(source, target_pred)
                score_one(args.scratch, name, config)
        print(f"FROZEN {config}", flush=True)


def predict(args):
    import torch

    from europriv_bench.adapters import PresidioAdapter, ShieldAdapter, kp_entities_to_bioes

    torch.set_num_threads(4)
    torch.manual_seed(127)
    if args.name == "presidio":
        adapter = PresidioAdapter()
        identity = {"kind": "presidio", "language": "en", "threshold": 0.0,
                    "crosswalk": adapter.PRESIDIO_TO_KP}
    elif args.name == "shield":
        adapter = ShieldAdapter(base_url=args.shield_url, pin=json.loads(args.shield_pin.read_text()))
        identity = {"kind": "shield", "pin_sha256": adapter.client.pin_sha256}
    else:
        from transformers import pipeline

        pipe = pipeline("token-classification", model=str(args.model), aggregation_strategy="simple",
                        device=args.device)
        pipe.tokenizer.model_max_length = 512
        identity = {"kind": "kp", "checkpoint": str(args.model), "training_seed": args.seed,
                    "files": {p.name: digest(p) for p in sorted(args.model.iterdir()) if p.is_file()}}
    identity.update(protocol_sha256=digest(PROTOCOL_PATH), script_sha256=digest(__file__),
                    harness_commit=PROTOCOL["harness_commit"], device=args.device, dtype="float32",
                    inference_seed=127,
                    packages={k: importlib.metadata.version(k) for k in ("torch", "transformers", "numpy")})
    for config in args.configs or CONFIGS:
        gold_path = args.scratch / "gold" / f"{config}.json"
        frozen = json.loads(gold_path.read_text())
        rows = frozen["rows"]
        gateway = config.startswith("gateway-")
        if args.name == "shield":
            adapter.prepare([frozen["language"]])
        ident = {**identity, "gold_sha256": digest(gold_path), "config": config,
                 "max_input_tokens": 512, "stride": 64 if gateway else 0,
                 "operator": "native-replace" if args.name == "shield" else "literal-mask-union-v1"}
        target = args.scratch / "predictions" / f"{args.name}--{config}.json"
        saved = json.loads(target.read_text()) if target.exists() else {"identity": ident, "receipts": []}
        if saved["identity"] != ident:
            raise ValueError(f"Prediction identity drift: {target}")
        receipts = saved["receipts"]
        for offset in range(len(receipts), len(rows), 16):
            texts = [r["text"] for r in rows[offset:offset + 16]]
            if args.name == "shield":
                responses = adapter.client.analyze(texts)
                for text, response in zip(texts, responses, strict=True):
                    entities = [{"start": e["start"], "end": e["end"],
                                 "label": adapter.SHIELD_TO_KP[e["entity_type"]]}
                                for e in response["items"] if e["entity_type"] in adapter.SHIELD_TO_KP]
                    receipts.append({"tags": kp_entities_to_bioes(text, entities),
                                     "intervals": [[e["start"], e["end"]] for e in response["items"]],
                                     "native": response, "provenance": adapter.provenance()})
            else:
                if args.name == "presidio":
                    batches = [[{"start": int(e.start), "end": int(e.end),
                                 "label": adapter.PRESIDIO_TO_KP[e.entity_type]}
                                for e in adapter._analyzer().analyze(text=t, language="en",
                                                                    entities=adapter._entities, score_threshold=0.0)]
                               for t in texts]
                else:
                    kwargs = {"stride": 64} if gateway else {}
                    batches = [[{"start": int(e["start"]), "end": int(e["end"]), "label": e["entity_group"]}
                                for e in entities] for entities in pipe(texts, batch_size=16, **kwargs)]
                for text, entities in zip(texts, batches, strict=True):
                    spans = [[e["start"], e["end"]] for e in entities]
                    receipts.append({"tags": kp_entities_to_bioes(text, entities), "intervals": spans,
                                     "native": entities, "output": literal_mask(text, spans)})
            write(target, saved)
            print(f"{args.name}/{config}: {len(receipts)}/{len(rows)}", flush=True)
        score_one(args.scratch, args.name, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "predict"))
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--bench", type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--configs", nargs="+", choices=CONFIGS)
    parser.add_argument("--shield-url", default="http://127.0.0.1:11446")
    parser.add_argument("--shield-pin", type=Path)
    args = parser.parse_args()
    (freeze if args.action == "freeze" else predict)(args)


if __name__ == "__main__":
    main()

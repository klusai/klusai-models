# RES-120: kp as the razorBridge AI Gateway privacy layer

2026-10-11. **Not ready for the north-star claim.** The 560M legal checkpoint has a TAB DIRECT-proxy win
against stock Presidio; RO/PL national-ID leakage is tied at zero. Production Shield is a different
baseline, gateway chat does not currently invoke it, and CPU capacity remains unproven.

## 1. Shield today and the integration seam

Source snapshot: webos `8f4c04cd29f04d3bfe826c5396aa6c5343159058` (read-only `_main-instructions`);
europriv-bench `43fafaa24cc635c115f1e1cb210e8a2bcb869020`; models starting at `be50353`.

- Chat attachments and knowledge ingestion call `privacy/service.py:{maybe_anonymize,anonymize_text}`
  and `privacy/presidio/remote.py:{remote_anonymize_text,anonymize_many}`. Django loads no detector.
  `PRESIDIO_API_URL` selects the remote FastAPI service. Required cloud egress is separately guarded by
  `knowledge/privacy_policy.py`; `maybe_anonymize` can return original text with a failure outcome,
  so a new gateway caller must check the outcome, never forward its `.text` unconditionally. [W1], [W2]
- `services/presidio/presidio_service/engine.py:run_anonymization(text, language, entities, whitelist=None,
  pseudonymize=False, salt="") -> (text, items, pseudonyms)` chunks at sentence/word boundaries (~2k chars),
  analyzes, remaps offsets, trims whitespace, applies whitelist, unions overlaps, then applies operators.
  Threshold is **0.8**. `items` contain `entity_type,start,end,score,text`; response adds `recognizer_set`.
  `/anonymize-text` accepts those arguments; `/anonymize-batch` takes `texts` (up to 64), returns ordered
  `results` with per-item `failed` and reverse maps. Preserve this wire contract. [W3], [W4]
- Native NLP languages are **NL/EN only** (`nl_core_news_lg`, `en_core_web_lg`); unsupported codes normalize
  to NL. Only one language's NLP weights remain resident per worker; switching languages evicts them.
  Predefined Presidio rules plus spaCy PERSON/LOCATION/ORGANIZATION/NRP run with custom IBAN, identity
  documents and BSN checks. NL adds Flair `ner-dutch-large`, phone/postcode/cadastral, abbreviation and date
  recognizers plus address context. This is not eight-language support. Inventory: `/recognizers`. [W5]
- Pseudonymization is already implemented by service `operators.py`: `<TYPE_8hex>` from salted SHA-256,
  normalized entity text, and a `{token: original}` reverse map. kOS derives the secret salt using HMAC
  from a document/attachment seed, stores maps on `KnowledgeDocument`/`KluAttachment`, combines passage
  maps, sweeps consistent replacements and restores only for authorized display. Stored/model-facing
  answers keep tokens. Reuse this; `privacy/pseudonymize.py` is the unrelated screenshot DB scrubber.
  The kp SDK's `extract_pii` works, but SDK `deidentify`/`pseudonymize` still raise `NotImplementedError`.
  `kp-anon` surrogates are not a drop-in replacement for the existing kOS reverse-map contract. [W6], [M1]
- Deployment varies: service README describes Mac Studio `:11435` through a VPC/Tailscale bridge;
  the newer routing document identifies CPU as the screener of record and an optional admitted GPU
  accelerator. It retries identical payloads on CPU after GPU failure. Admission currently requires NL
  and a suitable Celery/deadline context; synchronous short-text/web calls stay on CPU. Batch 404/405
  falls back to per-text CPU calls. No in-Django model fallback; readiness and required egress fail closed.
  Do not assume the existing accelerator policy admits gateway requests. [W2], [W7], [W8]

Gateway trace (the old `verticals/rb` package is now `verticals/compute`):

```text
/api/v1/chat/completions -> gateway_auth_required -> gateway/views.py:chat_completions
  -> route/model/key policy + spend plan -> _handle_non_streaming -> _dispatch.dispatch_completion
  -> OR _handle_streaming -> backend streaming (separate branch, bypasses dispatch_completion)
/api/v1/compute/gate/chat/completions -> compute/api_views.py:gate_completions
  -> gate_views.py:completions -> gate-key auth/model/budget checks
  -> inference_execution.py:execute_compute_inference -> _execute_authorized_inference
  -> gateway/_dispatch.py:dispatch_completion -> provider + fallback + accounting
```

Neither chat entry path currently screens the submitted conversation. Provider residency and zero-data-
retention controls are separate from PII removal. Existing attachment Shield does not imply gateway Shield.
Evidence: [gateway views][W9], [gate views][W10], [executor][W11], [dispatch][W12], [API routes][W13].

**Proposed engine seam:** introduce a detector interface inside `run_anonymization` at the current
`get_analyzer()/analyzer.analyze(...)` boundary: `detect(text, language, entities, threshold) ->
list[RecognizerResult]` using original Python character offsets. Implement `presidio` and `kp` engines;
select via a proposed service flag `PRIVACY_ENGINE=presidio|kp` (default Presidio), with a separately
scored `kp+rules` candidate retaining checksum/structured recognizers and replacing expensive semantic NER.
Use `KpModelAdapter.predict_spans` initially; version a crosswalk (EMAIL→EMAIL_ADDRESS, PHONE→PHONE_NUMBER,
DATE→DATE_TIME, ADDRESS→LOCATION, ORG_PARTY→ORGANIZATION). Retain explicit generic KP identifier/secret
types where subtype is unknown: never relabel every NATIONAL_ID as NL_BSN or ACCOUNT_ID as IBAN_CODE.
Version entity filters, operator defaults and model thresholds; 0.8 is not automatically calibrated for kp.
The kp branch must bypass NL fallback/eviction and support all eight languages, preserving shared
chunk/whitelist/overlap/operator/map logic. Fingerprint weights, tokenizer, mapping, thresholds and engine.

**Proposed request seam:** tenant/key flag `GATEWAY_PRIVACY_ENABLED`, default off, invokes one shared
`screen_gateway_payload` after authorization/validation and before spend-plan/body freezing and either
streaming or non-streaming dispatch. Wire both entry paths above, including Compute's routed policy body;
each primary/fallback must consume the screened immutable body. Screening failure means no provider call;
release any reservation already created. Cover message roles, text parts, tool results/arguments and
text-bearing tool descriptions; explicitly refuse unsupported image/audio content when screening is required.
Use request-scoped secret salts/maps, no conversation persistence, and no raw content in logs/telemetry.
Never send maps upstream. Preserve authorized restoration across SSE token boundaries if enabled.

Run kp in the same private service boundary on CPU/Outpost or a blade, not in ASGI workers. Support
explicit rollback to Presidio and deadline-bounded fallback only to an approved language/configuration;
if no qualifying screener is ready, refuse. Record the engine actually used. A failover result cannot
inherit the kp quality claim. Reuse canary/readiness, circuit and attestation machinery with kp identities.

## 2. Acceptance ledger: evidence, gap, closure

Quality figures are existing harness results; 95% CIs are Wilson unless stated. Baselines have one run.
Local detailed rows: `klusai-models/runs/res104-reid-local.json` from `scripts/score_reid_local.py`.

| Acceptance/config | Current evidence (n, seeds, 95% CI) | Gap and closure |
|---|---|---|
| TAB DIRECT leak < Presidio, disjoint CIs | 127 docs / 264 DIRECT subjects: 560M legal seed0 **0.0947 [0.0650,0.1361]**; stock Presidio **0.5000 [0.4401,0.5599]**. Legal seeds 0/1/2: 0.0947/0.0985/0.0985; per-seed CIs beyond s0 not archived here. | Scoped proxy win only. Re-score all seeds against the exact Shield service, frozen operators/thresholds and complete residual metric. [E1], [M2] |
| Shipped 280M on TAB | Same 264 subjects: **0.6742 [0.6156,0.7279]**, one shipped checkpoint; generic 560M s0 **0.5795 [0.5193,0.6375]**. | Neither is the legal winner. Pin model hashes; do not transfer the legal claim to generic kp. [E1], [M2] |
| RO real-skeleton leak < Presidio | 1,500 docs / 1,123 national-ID subjects: kp and Presidio **0 [0,0.003409]**; kp v2 seeds 0/1 also zero. | Strict improvement impossible on this saturated metric against zero. Keep v1 as regression anchor; pre-register an all-PII residual metric and a distinct harder held-out v2. [E2], [M3] |
| PL real-skeleton leak < Presidio | 1,500 docs / 1,096 subjects: kp and Presidio **0 [0,0.003493]**; kp v2 seeds 0/1 also zero. | Same floor; native review and full-PII subject coverage needed. Legacy metric parity must not be renamed a win. [E2], [M3] |
| NEW gateway config, eight languages | No scored rows, n=0, seeds=0, CI unavailable. | Build §3; compare pinned stock Presidio, deployed Shield, kp and kp+rules on identical held-out conversations, ≥3 kp seeds. |
| Over-redaction ≤ Presidio | TAB legal board records 0.043 false-positive token redaction, 127 docs, seed0; no CI or matched Presidio utility result. | Point estimate cannot establish non-inferiority. Score fraction of gold non-PII characters/tokens removed on all configs; paired conversation/document-bootstrap Δ upper bound ≤0 (literal no-worse margin). [E1] |
| Gateway p95/throughput | New CPU timings in §4; no HTTP/queue/Outpost/blade result. | Agree gateway SLO, replay same length/language/concurrency mix on both services; publish CIs, tail/error/timeout and saturation curves. |
| Flagged Shield + pseudonymization integration | Remote contract/maps already exist; no kp engine or gateway privacy hook found. | Implement §1 plus replay/rollback, Unicode offsets, cross-tenant map isolation, streaming, cancellation and fallback tests. |

Additional TAB constraints:
- The paired-bootstrap artifact is **kp − spaCy**, not kp − Presidio: −0.4015 [−0.4773,−0.3220],
  264 subjects, 2,000 resamples, seed0 checkpoint. Recompute paired differences against Shield. [E3]
- `tab_reid_leakage` actually uses `any(predicted token != O)` per occurrence, then requires every
  occurrence for an entity. Its docstring says every token, but implementation is weaker: masking only
  part of a name can count as success. Also QUASI is worse for the legal checkpoint (see source rows;
  3,701 QUASI subjects, no local-checkpoint CI archived). Version a strict residual/full-span metric
  alongside the historical score; no claim of empirical attacker re-identification probability. [E4], [M2]

Stock EuroPriv Presidio runs English on every language with threshold 0.0 and a restricted type crosswalk;
Shield uses NL/EN, threshold 0.8, custom rules and actual operators. Add a service adapter and keep both
baselines named. RO/PL configs remain `dev`; template-disjoint synthetic structure is not reviewed real
traffic. Existing native-speaker/IAA gates still apply. Increasing n alone will not break a zero/zero tie.

## 3. `gateway-traffic-v1` specification (proposed, not generated/scored)

Unit: a conversation, 1–8 turns, one primary language in **RO/EN/DE/FR/ES/IT/NL/PL**, plus code-switch
strata. Start with **1,000 held-out conversations/language (8,000)**, 200 dev/language, and a separate
training pool. Subject/source-document/scenario-family/translation siblings cannot cross splits.
Publish per-language and worst-language results; pooled results cannot hide a regression. Three training
seeds and 2,000+ paired conversation-cluster bootstrap resamples; Wilson counts report actual subjects.

Source manifest for the future dataset builder (`conf/gateway_traffic_manifest.yaml` in klusai-datasets):

| Source / URL | Licence and allowed role |
|---|---|
| Original KlusAI/Codex-authored scenarios, provenance at `https://github.com/klusai/klusai-datasets` + future pinned artifact path | Declare CC-BY-4.0 for KlusAI-authored release; record author/tool/session, prompt and content hashes. Synthetic train/dev/test in disjoint families. No copied customer conversations. |
| [Ai4Privacy openpii-1m](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1m/blob/main/README.md) | CC-BY-4.0 card grant permits commercial modification/redistribution with Ai4Privacy / Ai Suisse SA attribution; verified 2026-10-11. Use a separately reported external synthetic stratum, not invented real-traffic gold. Pin revision and audit overlap with all training manifests before inclusion. |
| [Existing TAB](https://github.com/NorskRegnesentral/text-anonymization-benchmark) | Existing `conf/tab_echr_manifest.yaml` records MIT. Keep frozen TAB as an independent anchor; never turn its held-out cases into gateway training examples. |

Manifest fields: source URL/revision, licence text/hash, attribution, permitted train/redistribute flags,
source split/id, generator/version/prompt hashes, seed, language, subject IDs, scenario family, exclusions,
file SHA-256, annotation/reviewer provenance. Stop ingestion if rights are unclear or restricted.
No Piiranha training/base; no Llama-bound/NC-ND sources, unlicensed web chats, customer logs or direct
HUDOC copying. For any single-file parquet: `hf_hub_download` + pyarrow, never streaming `load_dataset`.

Generation: author original workflows (support, billing, HR, clinical appointment, legal intake, internal
IT, code/log debugging, travel), then inject fictional entities with gold offsets and coreference IDs.
Codex writes RO prose directly, including diacritics, inflections and informal text, rather than translating
one English template. Example: “Te rog să-i răspunzi Anei Ionescu despre factura duplicată; adresa ei este
ana.ionescu@example.org. În mesajul anterior apare drept doamna Ionescu.” Record both mentions explicitly.
Use deterministic checksum-valid synthetic identifiers, reserved-domain emails and inert dummy secrets;
retain identity provenance, never claim generated numbers belong to nobody. No paid generation API.

Target mix per language: 20% PII-free hard negatives; 40% 1–2 subjects; 30% 3–5 subjects; 10% dense
pasted documents/tables. Across positive conversations require coverage of names/aliases, emails, phones,
addresses, national IDs (CNP/PESEL/BSN and local formats), IBAN/cards/accounts, DOB, case/company IDs,
medical identifiers/conditions and secrets. Balance label support explicitly; DATE/ORG/public statute
references are context-sensitive utility negatives, not automatic PII. Include split/obfuscated IDs,
quoted history, Unicode, punctuation, JSON/code, tool payloads and repeated entities across turns.
Length strata: 50% 32–128 tokens, 35% 129–512, 15% 513–2,048; test boundaries and reject silent truncation.

Schema: messages/roles/text parts, language, scenario/source/family IDs, per-message character spans,
KP label, entity/subject ID, DIRECT/QUASI, expected protection policy, split and provenance. Native
double-annotation on ≥100 conversations/language plus adjudication; expand until span agreement and
miss analysis support a frozen annotation guide. Status stays `dev` until review; synthetic never becomes
“real traffic” by relabelling. Score full residual identifiers across every turn and partial-character
leaks, over-redaction, task utility and pseudonym consistency/restoration; retain legacy metrics separately.
The 200 timing inputs below **are not this acceptance dataset**.

## 4. Serving and measured local CPU latency

2026-10-11, Apple M3 Ultra (28 CPU cores, 96 GiB), Python 3.13.12, Torch 2.12.0, Transformers 4.57.6.
One process, CPU fp32, eight intra-op threads / one inter-op, batch=1, ten warmups/model, one timed
pass/model, seed0 checkpoint per model. Same 200 fictional chat prompts (25/language); 280M lengths
73/140.5/207 tokens min/median/max; 560M 72/128/198. No truncation, model loading excluded.
Timed harness `KpModelAdapter.predict_spans`: tokenization, forward, aggregation and character alignment.

| Model / local checkpoint | n | p50 ms | p95 ms [95% bootstrap CI] | Sequential inputs/s [95% CI] |
|---|---:|---:|---:|---:|
| kp-280m / `runs/kp-deid-mdeberta-280m` | 200 | 106.46 | **147.97 [142.55,157.88]** | **9.04 [8.81,9.28]** |
| kp-560m-legal / `runs/res72-cjeu-tab-seed0` | 200 | 153.72 | **284.41 [206.66,347.89]** | **6.07 [5.76,6.37]** |

5,000 input-bootstrap resamples, RNG seed 120; intervals are conditional on this corpus/single pass,
not repeated-run/seed uncertainty. Shared host: training/data preparation and, during 560M, repo checks
were concurrent. These are observed timings, not idle-host capacity.
Tokenizer compatibility warnings were retained in the log; no tokenizer mutation or quality claim made.

Evidence directory: `/Users/mihai/codespace/rb-gtm-2026-10/estate/scratch/res120-gap` (`$S` below).
`bench_cpu.py`, `chat-inputs.jsonl`, `latency-manifest.json`, `*-timings.jsonl`, `latency-summary.json`,
`cpu-latency.log` retain resumable per-input times, environment, source/licence and every checkpoint file hash.
Corpus SHA-256: `7e9e2d2a8b975803b32fb0d01daec594745cc349d441b0daa670ffb338219b17`.
Reproduce here (CPU forced in script; `$S` is the evidence directory):
`PYTHONDONTWRITEBYTECODE=1 /Users/mihai/codespace/klusai-models/.venv/bin/python -u "$S/bench_cpu.py"`.

**Performance interpretation:** neither fp32 single lane reaches the task's ~10.5 req/s Presidio cap.
That cap has no matching run receipt in the inspected mirror, so it is a planning input, not a measured
head-to-head here. Service README reports 363→75 ms CPU→MPS/chunk, without n/CIs; batch schema comments
report ~610 ms round trip versus ~164 ms analysis, also without n/CIs. The newer routing doc reports
485–513 chars/s CPU/lane on a synthetic deed and 963–972 with two lanes. These different workloads do
not establish a chat p95. Pin topology and replay the same corpus before comparing. [W4], [W7], [W8]

CPU plan: export both checkpoints to ONNX fp32 with dynamic sequence lengths, then dynamic int8;
benchmark 1/2/4/8 threads and batches 1/2/4/8 on the actual Outpost CPU. ONNX Runtime recommends
dynamic quantization for transformers but warns accuracy can regress; 560M fp32 also needs care around
the optimizer's >2 GB limit. Retain fp32 fallback and validate spans plus harness leak/utility after every
export/quantization change. [ONNX guidance](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
Start at batch=1; experiment with bounded 2–5 ms microbatch waits. Separate chat/document queues,
bound depth, report overload, isolate tenants/maps and prewarm languages. HTTP batching is not tensor batching.

Baseline is ~148/284 ms p95 **before** transport/queue/operators; no measured ONNX/int8 or blade speedup.
Proposed design targets (requiring product ratification): ≤250 ms added
end-to-end p95 at ≥15 chat req/s for ≤256-token requests, zero silent bypasses; report long-input lane
separately. For a single lane to exceed 10.5 req/s its mean service time must be <95.2 ms. Multi-lane
capacity cannot be inferred by multiplying these results on a shared memory-bandwidth-bound host.
If CPU cannot pass, benchmark the pinned graph through the GPU bridge: RTT, fp16 parity, batching,
cold starts and CPU fallback. No hardware was provisioned here.

## 5. Ordered atomic build issues (estimates: engineer-days, not commitments)

| Order / title | Repo | Acceptance | Est. | GPU bridge |
|---|---|---|---:|---|
| 1. Freeze RES-120 metric/SLO contract | europriv-bench | Version full-span/residual metric; negative control catches partially masked names; retain v1 scores; explicitly resolve RO/PL zero-floor acceptance and ratify latency/load target. | 2 | No |
| 2. Add exact Shield service baseline | europriv-bench | HTTP adapter pins service/recognizers/operators/threshold; full/failed batch and offset tests; stock and Shield rows remain distinct. | 2 | No |
| 3. Manifest gateway sources and splits | klusai-datasets | Licence+URL+revision hashes, immutable family/subject splits, train-overlap audit, schema and eight-language coverage checks. | 1 | No |
| 4. Author gateway traffic and gold | klusai-datasets | 8k test + 1.6k dev; Codex RO originals, reviewed gold/IAA report, hard negatives and length/PII quotas; no unlicensed source ingestion. | 4–6 | No |
| 5. Register gateway/full-PII evaluations | europriv-bench | Per-language subject counts, Wilson and paired cluster CIs, utility residual metrics, corruption controls; TAB/RO/PL anchors retained. | 2 | No |
| 6. Freeze current candidate scorecard | klusai-models | 280M, legal/generic 560M, stock/Shield on identical configs; ≥3 kp seeds, full residual/utility CIs; explicitly report failures. | 2 | No |
| 7. Train targeted multilingual candidate | klusai-models | Train-disjoint gateway/domain mix, three seeds, only harness-supported gains; serial batch-16 fp32 MPS after checking other training jobs. | 3–5 | No |
| 8. Export and quantify CPU graph | klusai-models | ONNX fp32/int8 span/quality parity, hashes and 200+ input timings; actual CPU load curve incl. memory/errors and p95 CIs. | 2–3 | No |
| 9. Add flagged kp detector in Shield | klusai-webos | Default Presidio unchanged; kp + separately evaluated rules variant; eight languages, offsets/maps/filters/readiness/fingerprints and rollback verified. | 3 | No |
| 10. Screen generic gateway chat | klusai-webos | Tenant/key flag; both streaming/non-streaming and provider fallback use screened body; no-egress failure/cancellation tests; maps never upstream. | 2–3 | No |
| 11. Screen Compute gate execution | klusai-webos | Same hook before routed body freeze; bearer/token/browser producers, authorization/accounting and SSE restoration/tenant isolation tests. | 2–3 | No |
| 12. Measure blade + bridge (conditional) | klusai-models | Coordinator-provided blade; same quality corpus, three warm runs + cold, RTT/queue/batch curves and CPU fallback; no inferred GPU advantage. | 2 | **Yes** |
| 13. Admit kp blade identity (conditional) | klusai-webos | Model/graph/threshold attestation; gateway deadline and language admission; circuit/fallback refusal tests on supplied bridge. | 2 | **Yes** |
| 14. Publish internal acceptance/rollout decision | klusai-models | Frozen configs, exact Shield comparator, disjoint leak CIs and no-worse utility for every required slice; deployed CPU or blade SLO receipt. Remain off if any line fails. | 1 | Only if blade chosen |

Not verified: production state, strict residual/Shield comparisons, reviewed gateway gold, ONNX/int8,
Outpost/blade capacity or live integration. Timing JSONL is resumable; use fresh scratch for another trial.

[W1]: https://github.com/klusai/klusai-webos/blob/8f4c04cd29f04d3bfe826c5396aa6c5343159058/klusai_webos/privacy/service.py
[W2]: https://github.com/klusai/klusai-webos/blob/8f4c04cd29f04d3bfe826c5396aa6c5343159058/klusai_webos/privacy/presidio/remote.py
[W3]: https://github.com/klusai/klusai-webos/blob/8f4c04cd29f04d3bfe826c5396aa6c5343159058/services/presidio/presidio_service/engine.py
[W4]: https://github.com/klusai/klusai-webos/blob/8f4c04cd29f04d3bfe826c5396aa6c5343159058/services/presidio/presidio_service/schemas.py
[W5]: https://github.com/klusai/klusai-webos/blob/8f4c04cd29f04d3bfe826c5396aa6c5343159058/services/presidio/presidio_service/analyzer.py
[W6]: https://github.com/klusai/klusai-webos/blob/8f4c04cd29f04d3bfe826c5396aa6c5343159058/docs/architecture/upload-privacy.md#pseudonymization-stable-entity-tokens
[W7]: https://github.com/klusai/klusai-webos/blob/8f4c04cd29f04d3bfe826c5396aa6c5343159058/docs/architecture/privacy-screening-routing.md
[W8]: https://github.com/klusai/klusai-webos/blob/8f4c04cd29f04d3bfe826c5396aa6c5343159058/services/presidio/README.md
[W9]: https://github.com/klusai/klusai-webos/blob/8f4c04cd29f04d3bfe826c5396aa6c5343159058/klusai_webos/gateway/views.py
[W10]: https://github.com/klusai/klusai-webos/blob/8f4c04cd29f04d3bfe826c5396aa6c5343159058/klusai_webos/verticals/compute/gate_views.py
[W11]: https://github.com/klusai/klusai-webos/blob/8f4c04cd29f04d3bfe826c5396aa6c5343159058/klusai_webos/verticals/compute/inference_execution.py
[W12]: https://github.com/klusai/klusai-webos/blob/8f4c04cd29f04d3bfe826c5396aa6c5343159058/klusai_webos/gateway/_dispatch.py
[W13]: https://github.com/klusai/klusai-webos/blob/8f4c04cd29f04d3bfe826c5396aa6c5343159058/klusai_webos/verticals/compute/api_urls.py
[M1]: ../klusai/privacy/sdk/__init__.py
[M2]: res104-reid-win-scorecard.md
[M3]: klu-106-scorecard.md
[E1]: https://github.com/klusai/europriv-bench/blob/43fafaa24cc635c115f1e1cb210e8a2bcb869020/analysis/tab_reid_leaderboard.json
[E2]: https://github.com/klusai/europriv-bench/blob/43fafaa24cc635c115f1e1cb210e8a2bcb869020/baselines/leaderboard.json
[E3]: https://github.com/klusai/europriv-bench/blob/43fafaa24cc635c115f1e1cb210e8a2bcb869020/analysis/reid_paired_bootstrap.json
[E4]: https://github.com/klusai/europriv-bench/blob/43fafaa24cc635c115f1e1cb210e8a2bcb869020/src/europriv_bench/metrics.py#L792

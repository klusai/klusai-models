# RES-125 — frozen candidate scorecard

**Gateway acceptance: unverified.** Exact experimental operator receipts are scored on identical ordered gold for every completed row. Observed failures remain failures; no gateway promotion follows.

Machine-readable [scorecard](res125-candidate-scorecard.json) and [pre-scoring protocol](res125-protocol.json). Rates are fractions; confidence intervals are 95%. Training-seed ranges are descriptive.

- Best F1 and lowest all-PII residual differ on TAB: residual v2 is `560m-tab-s2` 0.6227 [0.6075, 0.6377] versus `560m-tab-s1` 0.5450 [0.5295, 0.5605]. These are post-hoc descriptions, with 127 documents / 3965 subjects per seed and Wilson CIs; all three seeds remain in acceptance.
- TAB `kp-280m` vs `presidio`: all-PII v2 by checkpoint = loss; utility = unverified; full gate **loss**.
- TAB `kp-280m` vs `shield`: all-PII v2 by checkpoint = win; utility = win; full gate **unverified**.
- TAB `560m-legal` vs `presidio`: all-PII v2 by checkpoint = loss, loss, loss; utility = loss, loss, loss; full gate **loss**.
- TAB `560m-legal` vs `shield`: all-PII v2 by checkpoint = win, win, win; utility = win, win, win; full gate **unverified**.
- TAB `560m-tab` vs `presidio`: all-PII v2 by checkpoint = loss, loss, loss; utility = win, win, win; full gate **loss**.
- TAB `560m-tab` vs `shield`: all-PII v2 by checkpoint = win, win, win; utility = win, win, win; full gate **unverified**.
- **Legacy anchor failure:** `560m-legal-s2` / `ro-realskeleton` leaks 3/1123 national-ID subjects, 0.0027 [0.0009, 0.0078] (Wilson).
- **Legacy anchor failure:** `560m-legal-s2` / `pl-realskeleton` leaks 3/1096 national-ID subjects, 0.0027 [0.0009, 0.0080] (Wilson).

## Frozen scope and interpretation

Harness `4e1d64097ff761db0256ad4f3f523c42a071ccc4`; core dataset `58faa4fa572cd8064818a6e1f8ca6e361b959c00`. OpenPII uses the merged RES-101 pinned validation loader: ten languages, 3,985 retained rows after 15 token-collision exclusions. Each JSON row records the ordered gold hash, dataset provenance, checkpoint/tokenizer hashes and runtime.

KP and stock Presidio use a newly frozen local literal block-mask operator on native detector offsets. It executes before scoring and never expands partial tokens. Shield uses its pinned native DEFAULT replace service, including native types outside the detection crosswalk. These are experimental operator results, not a deployed KP gateway. Legacy F1/leakage retains the harness label mask; v2 and utility use all actual operator intervals without that mask. V1 and v2 are therefore not nested: stricter character coverage can still yield a lower rate when an out-of-gold predicted type was discarded by the legacy label filter but actually removed by the operator.

Before any KP receipt or score, an unbounded shipped-tokenizer run was stopped and a 512-token maximum (stride 0) was set for every KP candidate. The amendment is recorded in JSON. Full documents remain in gold; this is a frozen capped-inference experiment, including all truncation failures. Stock and Shield retain their native full-document processing.

Encoder inference uses CPU or MPS fp32, recorded per result. Shared-host CPU contention led to a handoff after training: completed CPU configs were retained; incomplete configs were restarted in full on MPS with the same checkpoints and tokenizers, without mixing devices inside a receipt. Device equivalence over all documents is not asserted. The result is conditional on these frozen runtime receipts; no load/latency inference follows from wall time.

Wilson intervals describe gold subjects. Comparative intervals use 2,000 paired whole-document bootstrap draws, RNG 125, clustering all subjects within each document. Utility is the fraction of gold non-PII non-whitespace characters removed; retention is its complement. Empty denominators are unavailable. Without explicit entity IDs, v2 counts exact-value identifier subjects, not unique people; missing sensitivity remains UNSPECIFIED. This measures residual identifiers, not attacker re-identification probability. Bootstrap CIs condition on each checkpoint; training seeds are never pooled into subject n.

Stock EuroPriv Presidio runs English at threshold 0 with the benchmark adapter's entity whitelist/crosswalk. Shield uses EN/NL at 0.8, with Dutch fallback for other languages; its service pin is `763956e6da84eb755c168acf13c5d10b9239ca5c5415a8a30b4c806c8c55dd1f`. They remain separate comparators, with one inference run per config; baseline repeat variance is unverified. Small stock-Presidio F1 differences from the historical board are recorded in JSON; only fresh matched receipts enter comparisons. All source data is existing MIT/CC-BY-4.0 material; source URLs and licences are in JSON. No Piiranha training, paid APIs or cloud resources.

The shipped 280M is one checkpoint with unknown independent training seed; KLU-106 v2 seeds cannot be relabeled as replicas of it. Its old full-data run took under two hours, but original base/data revisions and initialization state are not pinned, and the current trainer changes its historical split. A faithful additional shipped-recipe seed was not established; this requirement remains unverified rather than treating a changed recipe as replication. Legal seeds 0/1/2 are existing CJEU checkpoints. TAB seeds 0/1/2 were rebuilt sequentially from the registered RES-104 train/dev recipe after the old worktree checkpoints were removed (prior runtime <2h/seed). TAB train/dev and test documents are disjoint; subject and near-duplicate passage disjointness is not established.

Descriptive best TAB-trained seed: **560m-tab-s2** (highest TAB test F1; post-hoc, not an acceptance selector). Lowest TAB all-PII residual v2: **560m-tab-s1**, also descriptive only. The representative table uses the F1-best seed across configs, plus legal seed0. Every seed's point estimate, Wilson/bootstrap CI, DIRECT/QUASI/UNSPECIFIED support and paired delta is in JSON.

## Representative rows

Legacy = TAB DIRECT any-overlap on TAB, decode-bearing national-ID on RO/PL real-skeleton, and name-in-context elsewhere (none is all-PII). V2 = all protected identifier subjects. All applicable legacy metrics and full TAB slices are also in JSON.

| Config | Model | Docs / v2 subjects | F1 [CI] | Legacy [Wilson CI]; n | V2 all [Wilson CI] | Over-redaction [CI] |
|---|---|---:|---|---|---|---|
| tab-echr-legal-en | kp-280m | 127 / 3965 | 0.2617 [0.2406, 0.2865] | 0.5909 [0.5307, 0.6485]; 264 | 0.6633 [0.6484, 0.6778] | 0.0252 [0.0227, 0.0280] |
| tab-echr-legal-en | 560m-legal-s0 | 127 / 3965 | 0.3399 [0.3148, 0.3676] | 0.0947 [0.0650, 0.1361]; 264 | 0.6583 [0.6434, 0.6729] | 0.0540 [0.0484, 0.0603] |
| tab-echr-legal-en | 560m-tab-s2 | 127 / 3965 | 0.5303 [0.4915, 0.5730] | 0.0947 [0.0650, 0.1361]; 264 | 0.6227 [0.6075, 0.6377] | 0.0061 [0.0044, 0.0080] |
| tab-echr-legal-en | presidio | 127 / 3965 | 0.5888 [0.5620, 0.6153] | 0.5000 [0.4401, 0.5599]; 264 | 0.3092 [0.2950, 0.3238] | 0.0244 [0.0222, 0.0269] |
| tab-echr-legal-en | shield | 127 / 3965 | 0.1442 [0.1266, 0.1637] | 0.5038 [0.4439, 0.5636]; 264 | 0.8348 [0.8229, 0.8460] | 0.0917 [0.0850, 0.0988] |
| ro-realskeleton | kp-280m | 1500 / 13751 | 0.7411 [0.7342, 0.7475] | 0.0000 [0.0000, 0.0034]; 1123 | 0.0814 [0.0769, 0.0861] | 0.0379 [0.0362, 0.0398] |
| ro-realskeleton | 560m-legal-s0 | 1500 / 13751 | 0.4809 [0.4768, 0.4851] | 0.0000 [0.0000, 0.0034]; 1123 | 0.6464 [0.6383, 0.6543] | 0.0101 [0.0091, 0.0111] |
| ro-realskeleton | 560m-tab-s2 | 1500 / 13751 | 0.5481 [0.5436, 0.5528] | 0.1211 [0.1033, 0.1415]; 1123 | 0.7246 [0.7171, 0.7320] | 0.0044 [0.0041, 0.0047] |
| ro-realskeleton | presidio | 1500 / 13751 | 0.4723 [0.4668, 0.4779] | 0.0000 [0.0000, 0.0034]; 1123 | 0.3916 [0.3835, 0.3998] | 0.0728 [0.0693, 0.0763] |
| ro-realskeleton | shield | 1500 / 13751 | 0.4129 [0.4078, 0.4178] | 1.0000 [0.9966, 1.0000]; 1123 | 0.4765 [0.4681, 0.4848] | 0.2216 [0.2174, 0.2259] |
| pl-realskeleton | kp-280m | 1500 / 13885 | 0.7627 [0.7554, 0.7707] | 0.0000 [0.0000, 0.0035]; 1096 | 0.0890 [0.0844, 0.0939] | 0.0350 [0.0332, 0.0367] |
| pl-realskeleton | 560m-legal-s0 | 1500 / 13885 | 0.4806 [0.4734, 0.4877] | 0.0000 [0.0000, 0.0035]; 1096 | 0.7171 [0.7096, 0.7245] | 0.0178 [0.0162, 0.0193] |
| pl-realskeleton | 560m-tab-s2 | 1500 / 13885 | 0.4706 [0.4653, 0.4760] | 0.0000 [0.0000, 0.0035]; 1096 | 0.7382 [0.7308, 0.7455] | 0.0039 [0.0036, 0.0042] |
| pl-realskeleton | presidio | 1500 / 13885 | 0.4732 [0.4681, 0.4785] | 0.0000 [0.0000, 0.0035]; 1096 | 0.3160 [0.3083, 0.3237] | 0.2370 [0.2323, 0.2418] |
| pl-realskeleton | shield | 1500 / 13885 | 0.4237 [0.4150, 0.4325] | 1.0000 [0.9965, 1.0000]; 1096 | 0.4943 [0.4860, 0.5027] | 0.2925 [0.2839, 0.3011] |
| en | kp-280m | 1500 / 3619 | 0.4635 [0.4445, 0.4827] | 0.0934 [0.0807, 0.1078]; 1778 | 0.1061 [0.0965, 0.1166] | 0.0591 [0.0514, 0.0697] |
| en | 560m-legal-s0 | 1500 / 3619 | 0.4042 [0.3851, 0.4237] | 0.1069 [0.0933, 0.1221]; 1778 | 0.3987 [0.3829, 0.4148] | 0.0388 [0.0345, 0.0432] |
| en | 560m-tab-s2 | 1500 / 3619 | 0.3611 [0.3425, 0.3802] | 0.1895 [0.1720, 0.2084]; 1778 | 0.6488 [0.6331, 0.6642] | 0.0085 [0.0071, 0.0101] |
| en | presidio | 1500 / 3619 | 0.4363 [0.4179, 0.4542] | 0.2537 [0.2340, 0.2744]; 1778 | 0.3169 [0.3020, 0.3323] | 0.0184 [0.0155, 0.0213] |
| en | shield | 1500 / 3619 | 0.3396 [0.3212, 0.3590] | 0.2677 [0.2477, 0.2888]; 1778 | 0.4435 [0.4274, 0.4597] | 0.0303 [0.0256, 0.0355] |
| de | kp-280m | 1500 / 3161 | 0.5067 [0.4879, 0.5267] | 0.0928 [0.0795, 0.1081]; 1584 | 0.0898 [0.0804, 0.1003] | 0.0394 [0.0358, 0.0432] |
| de | 560m-legal-s0 | 1500 / 3161 | 0.4517 [0.4325, 0.4720] | 0.1029 [0.0889, 0.1188]; 1584 | 0.4369 [0.4197, 0.4542] | 0.0293 [0.0259, 0.0330] |
| de | 560m-tab-s2 | 1500 / 3161 | 0.4161 [0.3959, 0.4369] | 0.2115 [0.1921, 0.2323]; 1584 | 0.6903 [0.6739, 0.7062] | 0.0030 [0.0023, 0.0038] |
| de | presidio | 1500 / 3161 | 0.2264 [0.2133, 0.2396] | 0.2468 [0.2262, 0.2687]; 1584 | 0.3407 [0.3244, 0.3574] | 0.3240 [0.3146, 0.3337] |
| de | shield | 1500 / 3161 | 0.4077 [0.3916, 0.4246] | 0.0827 [0.0701, 0.0973]; 1584 | 0.2218 [0.2076, 0.2366] | 0.1735 [0.1655, 0.1817] |
| fr | kp-280m | 1500 / 3285 | 0.5095 [0.4898, 0.5286] | 0.0811 [0.0685, 0.0957]; 1554 | 0.0935 [0.0840, 0.1039] | 0.0482 [0.0438, 0.0526] |
| fr | 560m-legal-s0 | 1500 / 3285 | 0.4324 [0.4116, 0.4531] | 0.0772 [0.0650, 0.0916]; 1554 | 0.4286 [0.4118, 0.4456] | 0.0253 [0.0223, 0.0285] |
| fr | 560m-tab-s2 | 1500 / 3285 | 0.4083 [0.3882, 0.4289] | 0.1847 [0.1662, 0.2047]; 1554 | 0.6670 [0.6507, 0.6829] | 0.0033 [0.0025, 0.0041] |
| fr | presidio | 1500 / 3285 | 0.2874 [0.2723, 0.3037] | 0.3681 [0.3445, 0.3924]; 1554 | 0.4630 [0.4460, 0.4801] | 0.1321 [0.1224, 0.1416] |
| fr | shield | 1500 / 3285 | 0.3809 [0.3632, 0.3979] | 0.0676 [0.0561, 0.0811]; 1554 | 0.2703 [0.2554, 0.2858] | 0.0950 [0.0894, 0.1010] |
| es | kp-280m | 1500 / 3197 | 0.4717 [0.4516, 0.4915] | 0.1167 [0.1014, 0.1340]; 1491 | 0.1051 [0.0949, 0.1162] | 0.0532 [0.0488, 0.0576] |
| es | 560m-legal-s0 | 1500 / 3197 | 0.3597 [0.3399, 0.3795] | 0.1502 [0.1330, 0.1693]; 1491 | 0.4898 [0.4725, 0.5072] | 0.0212 [0.0183, 0.0241] |
| es | 560m-tab-s2 | 1500 / 3197 | 0.3830 [0.3629, 0.4038] | 0.1831 [0.1643, 0.2035]; 1491 | 0.6819 [0.6655, 0.6978] | 0.0036 [0.0027, 0.0047] |
| es | presidio | 1500 / 3197 | 0.2377 [0.2223, 0.2527] | 0.4071 [0.3824, 0.4323]; 1491 | 0.4923 [0.4750, 0.5097] | 0.1830 [0.1712, 0.1942] |
| es | shield | 1500 / 3197 | 0.3666 [0.3490, 0.3834] | 0.0791 [0.0665, 0.0940]; 1491 | 0.2599 [0.2450, 0.2754] | 0.1682 [0.1610, 0.1760] |
| it | kp-280m | 1500 / 3244 | 0.4610 [0.4421, 0.4801] | 0.0914 [0.0780, 0.1068]; 1543 | 0.0845 [0.0754, 0.0945] | 0.0500 [0.0460, 0.0542] |
| it | 560m-legal-s0 | 1500 / 3244 | 0.4093 [0.3898, 0.4280] | 0.1063 [0.0919, 0.1227]; 1543 | 0.4300 [0.4131, 0.4471] | 0.0203 [0.0172, 0.0236] |
| it | 560m-tab-s2 | 1500 / 3244 | 0.3952 [0.3746, 0.4152] | 0.1828 [0.1643, 0.2028]; 1543 | 0.6159 [0.5990, 0.6325] | 0.0033 [0.0026, 0.0042] |
| it | presidio | 1500 / 3244 | 0.2354 [0.2211, 0.2498] | 0.3785 [0.3546, 0.4030]; 1543 | 0.4427 [0.4256, 0.4598] | 0.2421 [0.2310, 0.2533] |
| it | shield | 1500 / 3244 | 0.3184 [0.3020, 0.3342] | 0.0791 [0.0666, 0.0936]; 1543 | 0.2512 [0.2366, 0.2664] | 0.1911 [0.1828, 0.1990] |
| nl | kp-280m | 1500 / 3125 | 0.5243 [0.5049, 0.5438] | 0.1131 [0.0984, 0.1296]; 1592 | 0.1008 [0.0907, 0.1119] | 0.0585 [0.0530, 0.0648] |
| nl | 560m-legal-s0 | 1500 / 3125 | 0.4541 [0.4355, 0.4736] | 0.1143 [0.0996, 0.1309]; 1592 | 0.4422 [0.4249, 0.4597] | 0.0329 [0.0295, 0.0370] |
| nl | 560m-tab-s2 | 1500 / 3125 | 0.4364 [0.4163, 0.4582] | 0.1903 [0.1718, 0.2103]; 1592 | 0.7136 [0.6975, 0.7292] | 0.0081 [0.0066, 0.0098] |
| nl | presidio | 1500 / 3125 | 0.2305 [0.2170, 0.2440] | 0.2770 [0.2556, 0.2995]; 1592 | 0.3411 [0.3247, 0.3579] | 0.2801 [0.2681, 0.2920] |
| nl | shield | 1500 / 3125 | 0.5081 [0.4895, 0.5279] | 0.0867 [0.0738, 0.1015]; 1592 | 0.2291 [0.2147, 0.2442] | 0.0286 [0.0246, 0.0334] |
| ai4privacy-openpii-cs | kp-280m | 399 / 2526 | 0.4311 [0.4131, 0.4490] | 0.0494 [0.0350, 0.0693]; 627 | 0.0736 [0.0641, 0.0845] | 0.0544 [0.0492, 0.0601] |
| ai4privacy-openpii-cs | 560m-legal-s0 | 399 / 2526 | 0.3445 [0.3245, 0.3640] | 0.0574 [0.0418, 0.0785]; 627 | 0.4363 [0.4170, 0.4557] | 0.0273 [0.0237, 0.0311] |
| ai4privacy-openpii-cs | 560m-tab-s2 | 399 / 2526 | 0.3129 [0.2939, 0.3299] | 0.0590 [0.0431, 0.0803]; 627 | 0.5823 [0.5630, 0.6014] | 0.0069 [0.0059, 0.0080] |
| ai4privacy-openpii-cs | presidio | 399 / 2526 | 0.2813 [0.2672, 0.2957] | 0.3525 [0.3161, 0.3907]; 627 | 0.4371 [0.4178, 0.4565] | 0.1825 [0.1717, 0.1931] |
| ai4privacy-openpii-cs | shield | 399 / 2526 | 0.2538 [0.2394, 0.2683] | 0.0606 [0.0445, 0.0821]; 627 | 0.3797 [0.3609, 0.3987] | 0.2503 [0.2402, 0.2605] |
| ai4privacy-openpii-de | kp-280m | 399 / 2555 | 0.3072 [0.2896, 0.3241] | 0.0311 [0.0202, 0.0476]; 643 | 0.0524 [0.0445, 0.0618] | 0.0558 [0.0503, 0.0614] |
| ai4privacy-openpii-de | 560m-legal-s0 | 399 / 2555 | 0.3094 [0.2931, 0.3269] | 0.0327 [0.0215, 0.0494]; 643 | 0.3464 [0.3282, 0.3650] | 0.0372 [0.0328, 0.0415] |
| ai4privacy-openpii-de | 560m-tab-s2 | 399 / 2555 | 0.2889 [0.2721, 0.3069] | 0.0575 [0.0420, 0.0783]; 643 | 0.5342 [0.5149, 0.5535] | 0.0096 [0.0081, 0.0113] |
| ai4privacy-openpii-de | presidio | 399 / 2555 | 0.2412 [0.2276, 0.2545] | 0.2970 [0.2630, 0.3335]; 643 | 0.3980 [0.3792, 0.4172] | 0.2568 [0.2465, 0.2676] |
| ai4privacy-openpii-de | shield | 399 / 2555 | 0.2691 [0.2533, 0.2860] | 0.0684 [0.0514, 0.0906]; 643 | 0.3742 [0.3556, 0.3931] | 0.2042 [0.1952, 0.2132] |
| ai4privacy-openpii-el | kp-280m | 397 / 2434 | 0.3443 [0.3256, 0.3621] | 0.0169 [0.0095, 0.0301]; 649 | 0.0575 [0.0489, 0.0675] | 0.0558 [0.0510, 0.0608] |
| ai4privacy-openpii-el | 560m-legal-s0 | 397 / 2434 | 0.2910 [0.2719, 0.3110] | 0.0216 [0.0129, 0.0359]; 649 | 0.3878 [0.3687, 0.4074] | 0.0309 [0.0266, 0.0356] |
| ai4privacy-openpii-el | 560m-tab-s2 | 397 / 2434 | 0.3081 [0.2902, 0.3273] | 0.0354 [0.0237, 0.0526]; 649 | 0.5127 [0.4929, 0.5326] | 0.0081 [0.0067, 0.0097] |
| ai4privacy-openpii-el | presidio | 397 / 2434 | 0.3371 [0.3222, 0.3524] | 0.3945 [0.3576, 0.4326]; 649 | 0.3874 [0.3683, 0.4069] | 0.1245 [0.1150, 0.1335] |
| ai4privacy-openpii-el | shield | 397 / 2434 | 0.2553 [0.2404, 0.2703] | 0.0647 [0.0482, 0.0863]; 649 | 0.3135 [0.2954, 0.3322] | 0.2637 [0.2527, 0.2749] |
| ai4privacy-openpii-fi | kp-280m | 399 / 2639 | 0.3671 [0.3506, 0.3835] | 0.0474 [0.0343, 0.0652]; 738 | 0.0769 [0.0674, 0.0877] | 0.0523 [0.0478, 0.0574] |
| ai4privacy-openpii-fi | 560m-legal-s0 | 399 / 2639 | 0.2843 [0.2672, 0.3031] | 0.0271 [0.0176, 0.0415]; 738 | 0.4452 [0.4264, 0.4643] | 0.0282 [0.0245, 0.0321] |
| ai4privacy-openpii-fi | 560m-tab-s2 | 399 / 2639 | 0.2742 [0.2571, 0.2939] | 0.0610 [0.0459, 0.0806]; 738 | 0.5695 [0.5506, 0.5883] | 0.0091 [0.0077, 0.0106] |
| ai4privacy-openpii-fi | presidio | 399 / 2639 | 0.2852 [0.2715, 0.2999] | 0.3103 [0.2780, 0.3446]; 738 | 0.3528 [0.3348, 0.3712] | 0.3016 [0.2816, 0.3217] |
| ai4privacy-openpii-fi | shield | 399 / 2639 | 0.2327 [0.2180, 0.2473] | 0.0501 [0.0366, 0.0683]; 738 | 0.3429 [0.3251, 0.3613] | 0.2977 [0.2819, 0.3141] |
| ai4privacy-openpii-fr | kp-280m | 398 / 2679 | 0.3547 [0.3366, 0.3724] | 0.0276 [0.0177, 0.0427]; 689 | 0.0605 [0.0521, 0.0701] | 0.0630 [0.0556, 0.0716] |
| ai4privacy-openpii-fr | 560m-legal-s0 | 398 / 2679 | 0.3342 [0.3132, 0.3540] | 0.0189 [0.0111, 0.0320]; 689 | 0.3893 [0.3710, 0.4079] | 0.0388 [0.0343, 0.0434] |
| ai4privacy-openpii-fr | 560m-tab-s2 | 398 / 2679 | 0.3471 [0.3252, 0.3696] | 0.0493 [0.0355, 0.0682]; 689 | 0.5002 [0.4813, 0.5191] | 0.0120 [0.0098, 0.0145] |
| ai4privacy-openpii-fr | presidio | 398 / 2679 | 0.3439 [0.3266, 0.3608] | 0.2946 [0.2618, 0.3297]; 689 | 0.4524 [0.4336, 0.4713] | 0.0601 [0.0538, 0.0666] |
| ai4privacy-openpii-fr | shield | 398 / 2679 | 0.3254 [0.3051, 0.3454] | 0.0305 [0.0200, 0.0461]; 689 | 0.3800 [0.3618, 0.3985] | 0.1185 [0.1114, 0.1259] |
| ai4privacy-openpii-hu | kp-280m | 400 / 2752 | 0.2971 [0.2817, 0.3126] | 0.0494 [0.0359, 0.0676]; 729 | 0.0894 [0.0793, 0.1006] | 0.0592 [0.0544, 0.0639] |
| ai4privacy-openpii-hu | 560m-legal-s0 | 400 / 2752 | 0.3049 [0.2887, 0.3211] | 0.0480 [0.0347, 0.0660]; 729 | 0.4124 [0.3942, 0.4309] | 0.0337 [0.0297, 0.0379] |
| ai4privacy-openpii-hu | 560m-tab-s2 | 400 / 2752 | 0.2945 [0.2770, 0.3118] | 0.1015 [0.0816, 0.1256]; 729 | 0.5574 [0.5388, 0.5759] | 0.0110 [0.0091, 0.0132] |
| ai4privacy-openpii-hu | presidio | 400 / 2752 | 0.2962 [0.2826, 0.3096] | 0.3196 [0.2868, 0.3543]; 729 | 0.4437 [0.4252, 0.4623] | 0.1758 [0.1648, 0.1872] |
| ai4privacy-openpii-hu | shield | 400 / 2752 | 0.2414 [0.2270, 0.2569] | 0.1139 [0.0928, 0.1390]; 729 | 0.3674 [0.3496, 0.3856] | 0.2866 [0.2724, 0.3010] |
| ai4privacy-openpii-nl | kp-280m | 399 / 2623 | 0.3483 [0.3311, 0.3649] | 0.0358 [0.0243, 0.0523]; 699 | 0.0797 [0.0699, 0.0907] | 0.0673 [0.0614, 0.0738] |
| ai4privacy-openpii-nl | 560m-legal-s0 | 399 / 2623 | 0.3132 [0.2948, 0.3310] | 0.0358 [0.0243, 0.0523]; 699 | 0.3919 [0.3734, 0.4107] | 0.0370 [0.0328, 0.0417] |
| ai4privacy-openpii-nl | 560m-tab-s2 | 399 / 2623 | 0.3143 [0.2961, 0.3324] | 0.0587 [0.0435, 0.0786]; 699 | 0.5498 [0.5307, 0.5687] | 0.0095 [0.0080, 0.0111] |
| ai4privacy-openpii-nl | presidio | 399 / 2623 | 0.2888 [0.2755, 0.3023] | 0.2904 [0.2580, 0.3251]; 699 | 0.3953 [0.3768, 0.4142] | 0.2118 [0.1992, 0.2252] |
| ai4privacy-openpii-nl | shield | 399 / 2623 | 0.3671 [0.3491, 0.3869] | 0.0401 [0.0279, 0.0573]; 699 | 0.3149 [0.2974, 0.3329] | 0.0368 [0.0324, 0.0415] |
| ai4privacy-openpii-pl | kp-280m | 397 / 2788 | 0.3427 [0.3250, 0.3619] | 0.0407 [0.0286, 0.0574]; 738 | 0.0699 [0.0611, 0.0800] | 0.0582 [0.0523, 0.0651] |
| ai4privacy-openpii-pl | 560m-legal-s0 | 397 / 2788 | 0.3134 [0.2954, 0.3320] | 0.0203 [0.0124, 0.0333]; 738 | 0.4121 [0.3940, 0.4305] | 0.0289 [0.0254, 0.0325] |
| ai4privacy-openpii-pl | 560m-tab-s2 | 397 / 2788 | 0.3044 [0.2853, 0.3244] | 0.0339 [0.0230, 0.0495]; 738 | 0.5169 [0.4983, 0.5354] | 0.0092 [0.0080, 0.0104] |
| ai4privacy-openpii-pl | presidio | 397 / 2788 | 0.2619 [0.2482, 0.2758] | 0.3469 [0.3134, 0.3819]; 738 | 0.4078 [0.3897, 0.4262] | 0.2157 [0.2028, 0.2279] |
| ai4privacy-openpii-pl | shield | 397 / 2788 | 0.2520 [0.2395, 0.2648] | 0.0298 [0.0198, 0.0447]; 738 | 0.3052 [0.2884, 0.3226] | 0.3013 [0.2879, 0.3140] |
| ai4privacy-openpii-ro | kp-280m | 400 / 2732 | 0.3459 [0.3303, 0.3611] | 0.0444 [0.0313, 0.0626]; 676 | 0.0673 [0.0585, 0.0774] | 0.0630 [0.0568, 0.0695] |
| ai4privacy-openpii-ro | 560m-legal-s0 | 400 / 2732 | 0.3238 [0.3074, 0.3405] | 0.0281 [0.0181, 0.0435]; 676 | 0.4158 [0.3975, 0.4344] | 0.0229 [0.0200, 0.0261] |
| ai4privacy-openpii-ro | 560m-tab-s2 | 400 / 2732 | 0.2979 [0.2816, 0.3150] | 0.0666 [0.0501, 0.0879]; 676 | 0.5523 [0.5336, 0.5709] | 0.0096 [0.0083, 0.0109] |
| ai4privacy-openpii-ro | presidio | 400 / 2732 | 0.3030 [0.2887, 0.3177] | 0.3521 [0.3170, 0.3888]; 676 | 0.3587 [0.3409, 0.3769] | 0.1020 [0.0943, 0.1094] |
| ai4privacy-openpii-ro | shield | 400 / 2732 | 0.2729 [0.2587, 0.2869] | 0.0488 [0.0350, 0.0678]; 676 | 0.3920 [0.3739, 0.4105] | 0.2307 [0.2217, 0.2394] |
| ai4privacy-openpii-sv | kp-280m | 397 / 2748 | 0.3748 [0.3574, 0.3917] | 0.0373 [0.0260, 0.0534]; 750 | 0.0706 [0.0616, 0.0808] | 0.0556 [0.0507, 0.0603] |
| ai4privacy-openpii-sv | 560m-legal-s0 | 397 / 2748 | 0.2991 [0.2817, 0.3176] | 0.0227 [0.0142, 0.0360]; 750 | 0.3836 [0.3655, 0.4019] | 0.0317 [0.0281, 0.0356] |
| ai4privacy-openpii-sv | 560m-tab-s2 | 397 / 2748 | 0.3123 [0.2950, 0.3307] | 0.0320 [0.0216, 0.0472]; 750 | 0.5291 [0.5104, 0.5477] | 0.0083 [0.0070, 0.0096] |
| ai4privacy-openpii-sv | presidio | 397 / 2748 | 0.2646 [0.2509, 0.2783] | 0.2760 [0.2452, 0.3091]; 750 | 0.3483 [0.3307, 0.3663] | 0.2143 [0.2020, 0.2263] |
| ai4privacy-openpii-sv | shield | 397 / 2748 | 0.2035 [0.1916, 0.2150] | 0.0387 [0.0271, 0.0550]; 750 | 0.3486 [0.3310, 0.3666] | 0.3650 [0.3518, 0.3773] |

## Training seed spread

Ranges describe seed variability, not confidence intervals.

| Config | Candidate | Checkpoints | F1 mean [min,max] | V2 all mean [min,max] |
|---|---|---:|---|---|
| tab-echr-legal-en | 560m-legal | 3 | 0.3222 [0.3093, 0.3399] | 0.6564 [0.6202, 0.6908] |
| tab-echr-legal-en | 560m-tab | 3 | 0.5292 [0.5283, 0.5303] | 0.5723 [0.5450, 0.6227] |
| ro-realskeleton | 560m-legal | 3 | 0.4738 [0.3880, 0.5524] | 0.6377 [0.5296, 0.7371] |
| ro-realskeleton | 560m-tab | 3 | 0.5495 [0.4857, 0.6148] | 0.6704 [0.5584, 0.7282] |
| pl-realskeleton | 560m-legal | 3 | 0.4528 [0.4087, 0.4806] | 0.7144 [0.6099, 0.8162] |
| pl-realskeleton | 560m-tab | 3 | 0.5254 [0.4706, 0.6047] | 0.7213 [0.6742, 0.7516] |
| en | 560m-legal | 3 | 0.4137 [0.3901, 0.4470] | 0.5228 [0.3987, 0.6480] |
| en | 560m-tab | 3 | 0.3863 [0.3611, 0.4044] | 0.5554 [0.4708, 0.6488] |
| de | 560m-legal | 3 | 0.4455 [0.4137, 0.4710] | 0.5548 [0.4369, 0.7026] |
| de | 560m-tab | 3 | 0.4443 [0.4161, 0.4636] | 0.6085 [0.5201, 0.6903] |
| fr | 560m-legal | 3 | 0.4258 [0.3932, 0.4520] | 0.5525 [0.4286, 0.7011] |
| fr | 560m-tab | 3 | 0.4296 [0.4083, 0.4472] | 0.5832 [0.4959, 0.6670] |
| es | 560m-legal | 3 | 0.3684 [0.3422, 0.4032] | 0.5801 [0.4898, 0.7135] |
| es | 560m-tab | 3 | 0.4031 [0.3830, 0.4180] | 0.5838 [0.4914, 0.6819] |
| it | 560m-legal | 3 | 0.4110 [0.3966, 0.4270] | 0.5261 [0.4300, 0.6800] |
| it | 560m-tab | 3 | 0.4189 [0.3952, 0.4326] | 0.5312 [0.4541, 0.6159] |
| nl | 560m-legal | 3 | 0.4456 [0.4058, 0.4770] | 0.5623 [0.4422, 0.7101] |
| nl | 560m-tab | 3 | 0.4546 [0.4364, 0.4658] | 0.6237 [0.5219, 0.7136] |
| ai4privacy-openpii-cs | 560m-legal | 3 | 0.3492 [0.3295, 0.3737] | 0.4695 [0.3614, 0.6108] |
| ai4privacy-openpii-cs | 560m-tab | 3 | 0.3164 [0.3129, 0.3223] | 0.5104 [0.4557, 0.5823] |
| ai4privacy-openpii-de | 560m-legal | 3 | 0.3066 [0.3049, 0.3094] | 0.3954 [0.2869, 0.5530] |
| ai4privacy-openpii-de | 560m-tab | 3 | 0.3005 [0.2889, 0.3084] | 0.4798 [0.4204, 0.5342] |
| ai4privacy-openpii-el | 560m-legal | 3 | 0.3011 [0.2910, 0.3169] | 0.4344 [0.3225, 0.5929] |
| ai4privacy-openpii-el | 560m-tab | 3 | 0.3129 [0.3081, 0.3172] | 0.4690 [0.4281, 0.5127] |
| ai4privacy-openpii-fi | 560m-legal | 3 | 0.3063 [0.2843, 0.3218] | 0.4721 [0.3812, 0.5900] |
| ai4privacy-openpii-fi | 560m-tab | 3 | 0.2870 [0.2742, 0.2972] | 0.5135 [0.4532, 0.5695] |
| ai4privacy-openpii-fr | 560m-legal | 3 | 0.3445 [0.3321, 0.3671] | 0.4291 [0.3442, 0.5539] |
| ai4privacy-openpii-fr | 560m-tab | 3 | 0.3459 [0.3439, 0.3471] | 0.4645 [0.4274, 0.5002] |
| ai4privacy-openpii-hu | 560m-legal | 3 | 0.3244 [0.3049, 0.3497] | 0.4410 [0.3372, 0.5734] |
| ai4privacy-openpii-hu | 560m-tab | 3 | 0.3012 [0.2945, 0.3109] | 0.4975 [0.4382, 0.5574] |
| ai4privacy-openpii-nl | 560m-legal | 3 | 0.3227 [0.3132, 0.3326] | 0.4429 [0.3629, 0.5738] |
| ai4privacy-openpii-nl | 560m-tab | 3 | 0.3232 [0.3143, 0.3282] | 0.4952 [0.4449, 0.5498] |
| ai4privacy-openpii-pl | 560m-legal | 3 | 0.3275 [0.3134, 0.3455] | 0.4455 [0.3228, 0.6015] |
| ai4privacy-openpii-pl | 560m-tab | 3 | 0.3168 [0.3044, 0.3293] | 0.4909 [0.4423, 0.5169] |
| ai4privacy-openpii-ro | 560m-legal | 3 | 0.3430 [0.3191, 0.3862] | 0.4538 [0.3448, 0.6007] |
| ai4privacy-openpii-ro | 560m-tab | 3 | 0.3080 [0.2979, 0.3275] | 0.5221 [0.4575, 0.5564] |
| ai4privacy-openpii-sv | 560m-legal | 3 | 0.3193 [0.2991, 0.3345] | 0.4241 [0.3282, 0.5604] |
| ai4privacy-openpii-sv | 560m-tab | 3 | 0.3146 [0.3123, 0.3191] | 0.4813 [0.4330, 0.5291] |

## Contract verdicts

A privacy win requires disjoint v2 Wilson intervals **and** a paired delta upper bound <0 for all three candidate seeds, utility delta upper bound ≤0, preserved RO/PL legacy zero floors, and reviewed disjoint gold. `tie` means no established superiority, never statistical equivalence. Missing evidence means `unverified`; significant privacy/utility harm or a new anchor leak means `loss`.

Acceptance lines below are in seed order (0/1/2; one shipped checkpoint for 280M). V2 columns are per-checkpoint statistical comparisons; the final column also applies utility, anchors, three-seed and gold gates. Reviewed gateway gold is unverified throughout. `—` means that acceptance line does not apply to the config.

| Config | Candidate | Comparator | V2 all | TAB DIRECT v2 | Utility | Legacy zero floor | Overall |
|---|---|---|---|---|---|---|---|
| tab-echr-legal-en | kp-280m | presidio | loss | win | unverified | — | loss |
| tab-echr-legal-en | kp-280m | shield | win | win | win | — | unverified |
| tab-echr-legal-en | 560m-legal | presidio | loss / loss / loss | win / win / win | loss / loss / loss | — | loss |
| tab-echr-legal-en | 560m-legal | shield | win / win / win | win / win / win | win / win / win | — | unverified |
| tab-echr-legal-en | 560m-tab | presidio | loss / loss / loss | win / win / win | win / win / win | — | loss |
| tab-echr-legal-en | 560m-tab | shield | win / win / win | win / win / win | win / win / win | — | unverified |
| ro-realskeleton | kp-280m | presidio | win | — | win | tie | unverified |
| ro-realskeleton | kp-280m | shield | win | — | win | tie | unverified |
| ro-realskeleton | 560m-legal | presidio | loss / loss / loss | — | win / win / win | tie / tie / loss | loss |
| ro-realskeleton | 560m-legal | shield | loss / loss / loss | — | win / win / win | tie / tie / loss | loss |
| ro-realskeleton | 560m-tab | presidio | loss / loss / loss | — | win / win / win | loss / loss / loss | loss |
| ro-realskeleton | 560m-tab | shield | loss / loss / loss | — | win / win / win | loss / loss / loss | loss |
| pl-realskeleton | kp-280m | presidio | win | — | win | tie | unverified |
| pl-realskeleton | kp-280m | shield | win | — | win | tie | unverified |
| pl-realskeleton | 560m-legal | presidio | loss / loss / loss | — | win / win / win | tie / tie / loss | loss |
| pl-realskeleton | 560m-legal | shield | loss / loss / loss | — | win / win / win | tie / tie / loss | loss |
| pl-realskeleton | 560m-tab | presidio | loss / loss / loss | — | win / win / win | loss / loss / tie | loss |
| pl-realskeleton | 560m-tab | shield | loss / loss / loss | — | win / win / win | loss / loss / tie | loss |
| en | kp-280m | presidio | win | — | loss | — | loss |
| en | kp-280m | shield | win | — | loss | — | loss |
| en | 560m-legal | presidio | loss / loss / loss | — | loss / loss / win | — | loss |
| en | 560m-legal | shield | win / loss / loss | — | loss / unverified / win | — | loss |
| en | 560m-tab | presidio | loss / loss / loss | — | win / win / win | — | loss |
| en | 560m-tab | shield | loss / tie / loss | — | win / win / win | — | loss |
| de | kp-280m | presidio | win | — | win | — | unverified |
| de | kp-280m | shield | win | — | win | — | unverified |
| de | 560m-legal | presidio | loss / loss / loss | — | win / win / win | — | loss |
| de | 560m-legal | shield | loss / loss / loss | — | win / win / win | — | loss |
| de | 560m-tab | presidio | loss / loss / loss | — | win / win / win | — | loss |
| de | 560m-tab | shield | loss / loss / loss | — | win / win / win | — | loss |
| fr | kp-280m | presidio | win | — | win | — | unverified |
| fr | kp-280m | shield | win | — | win | — | unverified |
| fr | 560m-legal | presidio | win / loss / loss | — | win / win / win | — | loss |
| fr | 560m-legal | shield | loss / loss / loss | — | win / win / win | — | loss |
| fr | 560m-tab | presidio | loss / tie / loss | — | win / win / win | — | loss |
| fr | 560m-tab | shield | loss / loss / loss | — | win / win / win | — | loss |
| es | kp-280m | presidio | win | — | win | — | unverified |
| es | kp-280m | shield | win | — | win | — | unverified |
| es | 560m-legal | presidio | tie / loss / loss | — | win / win / win | — | loss |
| es | 560m-legal | shield | loss / loss / loss | — | win / win / win | — | loss |
| es | 560m-tab | presidio | loss / tie / loss | — | win / win / win | — | loss |
| es | 560m-tab | shield | loss / loss / loss | — | win / win / win | — | loss |
| it | kp-280m | presidio | win | — | win | — | unverified |
| it | kp-280m | shield | win | — | win | — | unverified |
| it | 560m-legal | presidio | tie / tie / loss | — | win / win / win | — | loss |
| it | 560m-legal | shield | loss / loss / loss | — | win / win / win | — | loss |
| it | 560m-tab | presidio | loss / tie / loss | — | win / win / win | — | loss |
| it | 560m-tab | shield | loss / loss / loss | — | win / win / win | — | loss |
| nl | kp-280m | presidio | win | — | win | — | unverified |
| nl | kp-280m | shield | win | — | loss | — | loss |
| nl | 560m-legal | presidio | loss / loss / loss | — | win / win / win | — | loss |
| nl | 560m-legal | shield | loss / loss / loss | — | unverified / loss / win | — | loss |
| nl | 560m-tab | presidio | loss / loss / loss | — | win / win / win | — | loss |
| nl | 560m-tab | shield | loss / loss / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-cs | kp-280m | presidio | win | — | win | — | unverified |
| ai4privacy-openpii-cs | kp-280m | shield | win | — | win | — | unverified |
| ai4privacy-openpii-cs | 560m-legal | presidio | tie / win / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-cs | 560m-legal | shield | loss / tie / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-cs | 560m-tab | presidio | loss / tie / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-cs | 560m-tab | shield | loss / loss / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-de | kp-280m | presidio | win | — | win | — | unverified |
| ai4privacy-openpii-de | kp-280m | shield | win | — | win | — | unverified |
| ai4privacy-openpii-de | 560m-legal | presidio | win / win / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-de | 560m-legal | shield | tie / win / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-de | 560m-tab | presidio | loss / tie / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-de | 560m-tab | shield | loss / loss / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-el | kp-280m | presidio | win | — | win | — | unverified |
| ai4privacy-openpii-el | kp-280m | shield | win | — | win | — | unverified |
| ai4privacy-openpii-el | 560m-legal | presidio | tie / win / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-el | 560m-legal | shield | loss / tie / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-el | 560m-tab | presidio | loss / loss / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-el | 560m-tab | shield | loss / loss / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-fi | kp-280m | presidio | win | — | win | — | unverified |
| ai4privacy-openpii-fi | kp-280m | shield | win | — | win | — | unverified |
| ai4privacy-openpii-fi | 560m-legal | presidio | loss / tie / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-fi | 560m-legal | shield | loss / loss / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-fi | 560m-tab | presidio | loss / loss / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-fi | 560m-tab | shield | loss / loss / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-fr | kp-280m | presidio | win | — | unverified | — | unverified |
| ai4privacy-openpii-fr | kp-280m | shield | win | — | win | — | unverified |
| ai4privacy-openpii-fr | 560m-legal | presidio | win / win / loss | — | win / unverified / win | — | loss |
| ai4privacy-openpii-fr | 560m-legal | shield | tie / tie / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-fr | 560m-tab | presidio | tie / tie / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-fr | 560m-tab | shield | loss / loss / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-hu | kp-280m | presidio | win | — | win | — | unverified |
| ai4privacy-openpii-hu | kp-280m | shield | win | — | win | — | unverified |
| ai4privacy-openpii-hu | 560m-legal | presidio | tie / win / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-hu | 560m-legal | shield | loss / tie / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-hu | 560m-tab | presidio | loss / tie / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-hu | 560m-tab | shield | loss / loss / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-nl | kp-280m | presidio | win | — | win | — | unverified |
| ai4privacy-openpii-nl | kp-280m | shield | win | — | loss | — | loss |
| ai4privacy-openpii-nl | 560m-legal | presidio | tie / tie / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-nl | 560m-legal | shield | loss / loss / loss | — | unverified / loss / win | — | loss |
| ai4privacy-openpii-nl | 560m-tab | presidio | loss / loss / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-nl | 560m-tab | shield | loss / loss / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-pl | kp-280m | presidio | win | — | win | — | unverified |
| ai4privacy-openpii-pl | kp-280m | shield | win | — | win | — | unverified |
| ai4privacy-openpii-pl | 560m-legal | presidio | tie / win / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-pl | 560m-legal | shield | loss / tie / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-pl | 560m-tab | presidio | loss / tie / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-pl | 560m-tab | shield | loss / loss / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-ro | kp-280m | presidio | win | — | win | — | unverified |
| ai4privacy-openpii-ro | kp-280m | shield | win | — | win | — | unverified |
| ai4privacy-openpii-ro | 560m-legal | presidio | loss / tie / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-ro | 560m-legal | shield | tie / win / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-ro | 560m-tab | presidio | loss / loss / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-ro | 560m-tab | shield | loss / loss / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-sv | kp-280m | presidio | win | — | win | — | unverified |
| ai4privacy-openpii-sv | kp-280m | shield | win | — | win | — | unverified |
| ai4privacy-openpii-sv | 560m-legal | presidio | tie / tie / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-sv | 560m-legal | shield | tie / tie / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-sv | 560m-tab | presidio | loss / loss / loss | — | win / win / win | — | loss |
| ai4privacy-openpii-sv | 560m-tab | shield | loss / loss / loss | — | win / win / win | — | loss |

| Gateway acceptance line | Verdict | Reason |
|---|---|---|
| Eight-language reviewed held-out privacy + utility | unverified | Dev/split gates remain; see failures above |
| Three shipped-280M training seeds | unverified | Only one shipped-recipe checkpoint; v2 is distinct |
| p95 ≤250 ms at ≥15 successful req/s | unverified | No contract HTTP/load trials |
| Gateway integration, rollback, pseudonym maps | unverified | Detector/operator experiment only |

## Worst language on the eight core configs

These are heterogeneous benchmark strata, not eight-language gateway traffic. Worst means highest all-PII residual-v2 point estimate; JSON includes all support slices.

| Checkpoint | Completed languages | Worst language | V2 all [Wilson CI] |
|---|---:|---|---|
| kp-280m | 8 | en | 0.1061 [0.0965, 0.1166] |
| 560m-legal-s0 | 8 | pl | 0.7171 [0.7096, 0.7245] |
| 560m-legal-s1 | 8 | pl | 0.6099 [0.6018, 0.6180] |
| 560m-legal-s2 | 8 | pl | 0.8162 [0.8097, 0.8226] |
| 560m-tab-s0 | 8 | pl | 0.7516 [0.7443, 0.7587] |
| 560m-tab-s1 | 8 | pl | 0.6742 [0.6663, 0.6819] |
| 560m-tab-s2 | 8 | pl | 0.7382 [0.7308, 0.7455] |
| presidio | 8 | es | 0.4923 [0.4750, 0.5097] |
| shield | 8 | pl | 0.4943 [0.4860, 0.5027] |

## Failures and limits

All 171 expected model/config runs completed; no configuration or seed was dropped.

- Gateway conversation gold and all-eight-language reviewed acceptance.
- Subject/scenario/translation disjointness for all candidates and corpora.
- Three independent shipped-280M training seeds (v2 seeds are a different recipe).
- Baseline inference-repeat variability (one fresh run/config; historical F1 drift is recorded).
- All-document CPU/MPS bitwise equivalence; per-config device is recorded, fp32 throughout.
- Deployed KP/stock operator and Shield production asset parity.
- No-truncation long-input service behavior (KP pipeline truncates at 512 tokens).
- End-to-end p95 <=250ms at >=15 req/s; no HTTP load acceptance trials.
- Semantic utility, pseudonym restoration, gateway integration and rollback.

KP uses the explicit 512-token truncation; full input rows and gold remain in every denominator. Shield's native chunking remains part of its frozen service. This may explain failures but does not remove them. The older TAB any-overlap win must not be called a v2 win.

Reproduce with `scripts/scorecard_res125.py freeze`, `predict` per pinned candidate/baseline, then `scripts/report_res125.py --scratch <scratch>`; use the pinned harness on PYTHONPATH. Raw private receipts and logs live in the assigned RES-125 scratch directory; only aggregate scores, provenance and sufficient reproduction code are committed.

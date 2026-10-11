# RES-127 — gateway-trained encoder scorecard

**Experimental; no gateway promotion.** Checkpoints are selected only by gateway-dev token loss.
Quality numbers below come only from the EuroPriv harness. No test-based recipe or seed selection.

**Full gateway entity F1 is unavailable:** distinct entities can share a whitespace token (e.g. CSV).
Gateway F1 below is a conditional diagnostic on representable rows, with its n shown. Every conversation
remains in residual-v2 and utility scoring. This limitation does not alter the primary endpoint.

[Protocol](res127-protocol.json) · [Complete results, paired deltas and provenance](res127-encoder-scorecard.json)

## Gateway numerical checks

Each language has 1,000 conversations / 4,506 identifier subjects. The required checks cover three
independent training seeds and eight languages separately; these counts never pool subjects or seeds.
Privacy requires disjoint Wilson intervals and a paired bootstrap delta upper<0; utility requires
a paired delta upper<=0. Review/disjointness and deployment gates remain separate and unverified.

| Comparator | Completed / required seed-language checks | Privacy passes | Utility passes | Both |
|---|---:|---:|---:|---:|
| kp-280m | 24 / 24 | 24 | 24 | 24 |
| presidio | 24 / 24 | 24 | 24 | 24 |
| shield | 24 / 24 | 24 | 24 | 24 |

## Training and measurement

Training: all 40,000 gateway TRAIN rows plus the original 150,000 RO/EN/PL rows; batch 16, fp32, MPS,
max length 512, LoRA rank 16, three epochs, length-grouped batches, padding rounded to 32 tokens,
non-reentrant activation checkpointing and cache clearing every 25 steps.
[Training memory amendment](res127-training-amendments.json); gateway dev: 1,600 conversations.
Each 1,000-step checkpoint is eligible by dev loss; the best eligible checkpoint per seed is scored below.
The final partial interval is not an eligible selection checkpoint. Dev loss is not a quality claim.

Gateway: 512-token windows, stride 64, full conversations. Anchors: frozen RES-125 capped inference
(512 tokens, stride 0), with unchanged matched kp-280m/Presidio/Shield receipts rescored by this harness.
Gold remains complete for capped anchors. Gateway KP/stock operators mask the union of native offsets;
Shield uses its pinned native replacement service. These are local experiments, not deployed gateway behavior.

95% Wilson intervals describe identifier subjects; F1 and utility CIs use 2,000 whole-document/conversation
bootstrap draws (RNG125). Comparative deltas use paired draws. Subjects and seeds are never pooled.
Translations share families across languages; compare each language separately. No aggregate language win.

| Candidate | Seed | Total elapsed seconds (including queue) | Resume | Selected checkpoint |
|---|---:|---:|---|---|
| mdeberta-280m-seed0 | 0 | 9790.6 | True | checkpoint-13000 |
| mdeberta-280m-seed1 | 1 | 7347.4 | False | checkpoint-2000 |
| mdeberta-280m-seed2 | 2 | 7411.4 | False | checkpoint-2000 |

## Harness results

Each candidate row is one independently trained checkpoint; kp-280m is the single shipped checkpoint
(its independent training seed is unknown). Presidio/Shield are one inference run per configuration.

| Config | Model | Docs | Identifier subjects | F1 [bootstrap CI] | Residual v2 [Wilson CI] | Over-redaction [bootstrap CI] |
|---|---|---:|---:|---|---|---|
| gateway-ro | mdeberta-280m-seed0 | 1000 | 4506 | 0.8917 [0.8642, 0.9166] (subset n=893) | 0.1580 [0.1477, 0.1690] | 0.0604 [0.0537, 0.0669] |
| gateway-ro | mdeberta-280m-seed1 | 1000 | 4506 | 0.8674 [0.8435, 0.8909] (subset n=893) | 0.1937 [0.1825, 0.2055] | 0.0629 [0.0556, 0.0699] |
| gateway-ro | mdeberta-280m-seed2 | 1000 | 4506 | 0.8960 [0.8749, 0.9154] (subset n=893) | 0.1882 [0.1770, 0.1999] | 0.0303 [0.0276, 0.0331] |
| gateway-ro | kp-280m | 1000 | 4506 | 0.4352 [0.4073, 0.4630] (subset n=893) | 0.3338 [0.3202, 0.3477] | 0.2334 [0.2108, 0.2559] |
| gateway-ro | presidio | 1000 | 4506 | 0.2789 [0.2676, 0.2901] (subset n=893) | 0.6116 [0.5973, 0.6258] | 0.1732 [0.1678, 0.1795] |
| gateway-ro | shield | 1000 | 4506 | 0.2197 [0.2111, 0.2285] (subset n=893) | 0.4561 [0.4416, 0.4706] | 0.2703 [0.2631, 0.2777] |
| gateway-en | mdeberta-280m-seed0 | 1000 | 4506 | 0.8925 [0.8667, 0.9160] (subset n=893) | 0.1636 [0.1530, 0.1746] | 0.0560 [0.0499, 0.0623] |
| gateway-en | mdeberta-280m-seed1 | 1000 | 4506 | 0.8828 [0.8603, 0.9032] (subset n=893) | 0.2259 [0.2139, 0.2384] | 0.0425 [0.0375, 0.0472] |
| gateway-en | mdeberta-280m-seed2 | 1000 | 4506 | 0.8780 [0.8574, 0.8972] (subset n=893) | 0.2082 [0.1966, 0.2203] | 0.0308 [0.0279, 0.0337] |
| gateway-en | kp-280m | 1000 | 4506 | 0.4205 [0.3948, 0.4461] (subset n=893) | 0.3693 [0.3553, 0.3835] | 0.2163 [0.1954, 0.2379] |
| gateway-en | presidio | 1000 | 4506 | 0.4909 [0.4767, 0.5046] (subset n=893) | 0.5424 [0.5278, 0.5569] | 0.0825 [0.0756, 0.0895] |
| gateway-en | shield | 1000 | 4506 | 0.4914 [0.4813, 0.5022] (subset n=893) | 0.5419 [0.5274, 0.5564] | 0.0719 [0.0631, 0.0807] |
| gateway-de | mdeberta-280m-seed0 | 1000 | 4506 | 0.9039 [0.8786, 0.9265] (subset n=893) | 0.1451 [0.1352, 0.1557] | 0.0574 [0.0508, 0.0642] |
| gateway-de | mdeberta-280m-seed1 | 1000 | 4506 | 0.8871 [0.8660, 0.9066] (subset n=893) | 0.1815 [0.1706, 0.1931] | 0.0604 [0.0535, 0.0672] |
| gateway-de | mdeberta-280m-seed2 | 1000 | 4506 | 0.9021 [0.8816, 0.9201] (subset n=893) | 0.1897 [0.1786, 0.2015] | 0.0341 [0.0308, 0.0375] |
| gateway-de | kp-280m | 1000 | 4506 | 0.4435 [0.4172, 0.4703] (subset n=893) | 0.3402 [0.3265, 0.3542] | 0.2009 [0.1823, 0.2201] |
| gateway-de | presidio | 1000 | 4506 | 0.2669 [0.2559, 0.2778] (subset n=893) | 0.5333 [0.5187, 0.5478] | 0.2887 [0.2781, 0.2991] |
| gateway-de | shield | 1000 | 4506 | 0.3082 [0.2993, 0.3164] (subset n=893) | 0.4481 [0.4336, 0.4626] | 0.2476 [0.2376, 0.2573] |
| gateway-fr | mdeberta-280m-seed0 | 1000 | 4506 | 0.8952 [0.8667, 0.9201] (subset n=893) | 0.1549 [0.1446, 0.1658] | 0.0537 [0.0480, 0.0595] |
| gateway-fr | mdeberta-280m-seed1 | 1000 | 4506 | 0.8744 [0.8506, 0.8971] (subset n=893) | 0.2002 [0.1888, 0.2121] | 0.0658 [0.0580, 0.0733] |
| gateway-fr | mdeberta-280m-seed2 | 1000 | 4506 | 0.8822 [0.8614, 0.9020] (subset n=893) | 0.1880 [0.1768, 0.1996] | 0.0270 [0.0245, 0.0294] |
| gateway-fr | kp-280m | 1000 | 4506 | 0.4416 [0.4155, 0.4687] (subset n=893) | 0.3462 [0.3325, 0.3602] | 0.2288 [0.2067, 0.2504] |
| gateway-fr | presidio | 1000 | 4506 | 0.3215 [0.3089, 0.3342] (subset n=893) | 0.5965 [0.5821, 0.6108] | 0.1435 [0.1369, 0.1503] |
| gateway-fr | shield | 1000 | 4506 | 0.2609 [0.2472, 0.2746] (subset n=893) | 0.4401 [0.4256, 0.4546] | 0.1750 [0.1656, 0.1842] |
| gateway-es | mdeberta-280m-seed0 | 1000 | 4506 | 0.8975 [0.8713, 0.9221] (subset n=893) | 0.1352 [0.1255, 0.1454] | 0.0836 [0.0740, 0.0930] |
| gateway-es | mdeberta-280m-seed1 | 1000 | 4506 | 0.8681 [0.8419, 0.8925] (subset n=893) | 0.1902 [0.1790, 0.2019] | 0.0731 [0.0648, 0.0811] |
| gateway-es | mdeberta-280m-seed2 | 1000 | 4506 | 0.9041 [0.8841, 0.9221] (subset n=893) | 0.1886 [0.1775, 0.2003] | 0.0313 [0.0284, 0.0342] |
| gateway-es | kp-280m | 1000 | 4506 | 0.4388 [0.4125, 0.4658] (subset n=893) | 0.3597 [0.3459, 0.3739] | 0.2239 [0.2030, 0.2446] |
| gateway-es | presidio | 1000 | 4506 | 0.3247 [0.3121, 0.3374] (subset n=893) | 0.5353 [0.5207, 0.5498] | 0.2220 [0.2094, 0.2331] |
| gateway-es | shield | 1000 | 4506 | 0.2928 [0.2848, 0.3007] (subset n=893) | 0.4483 [0.4338, 0.4628] | 0.2554 [0.2451, 0.2647] |
| gateway-it | mdeberta-280m-seed0 | 1000 | 4506 | 0.8984 [0.8734, 0.9220] (subset n=893) | 0.1589 [0.1485, 0.1699] | 0.0485 [0.0434, 0.0535] |
| gateway-it | mdeberta-280m-seed1 | 1000 | 4506 | 0.8637 [0.8388, 0.8874] (subset n=893) | 0.1693 [0.1587, 0.1806] | 0.0549 [0.0487, 0.0608] |
| gateway-it | mdeberta-280m-seed2 | 1000 | 4506 | 0.8923 [0.8716, 0.9120] (subset n=893) | 0.1906 [0.1794, 0.2024] | 0.0306 [0.0279, 0.0334] |
| gateway-it | kp-280m | 1000 | 4506 | 0.4232 [0.3981, 0.4484] (subset n=893) | 0.3506 [0.3368, 0.3647] | 0.2116 [0.1922, 0.2306] |
| gateway-it | presidio | 1000 | 4506 | 0.3377 [0.3250, 0.3498] (subset n=893) | 0.5311 [0.5165, 0.5456] | 0.2095 [0.2014, 0.2169] |
| gateway-it | shield | 1000 | 4506 | 0.2889 [0.2775, 0.3006] (subset n=893) | 0.4414 [0.4270, 0.4560] | 0.2629 [0.2527, 0.2729] |
| gateway-nl | mdeberta-280m-seed0 | 1000 | 4506 | 0.9038 [0.8792, 0.9264] (subset n=893) | 0.1480 [0.1380, 0.1587] | 0.0498 [0.0444, 0.0551] |
| gateway-nl | mdeberta-280m-seed1 | 1000 | 4506 | 0.8603 [0.8391, 0.8805] (subset n=893) | 0.1835 [0.1725, 0.1951] | 0.0578 [0.0513, 0.0642] |
| gateway-nl | mdeberta-280m-seed2 | 1000 | 4506 | 0.8958 [0.8744, 0.9148] (subset n=893) | 0.1875 [0.1764, 0.1992] | 0.0338 [0.0306, 0.0370] |
| gateway-nl | kp-280m | 1000 | 4506 | 0.4132 [0.3882, 0.4384] (subset n=893) | 0.3502 [0.3364, 0.3643] | 0.2059 [0.1872, 0.2247] |
| gateway-nl | presidio | 1000 | 4506 | 0.3372 [0.3240, 0.3495] (subset n=893) | 0.5628 [0.5483, 0.5772] | 0.1679 [0.1607, 0.1753] |
| gateway-nl | shield | 1000 | 4506 | 0.4653 [0.4498, 0.4812] (subset n=893) | 0.4481 [0.4336, 0.4626] | 0.1498 [0.1348, 0.1643] |
| gateway-pl | mdeberta-280m-seed0 | 1000 | 4506 | 0.9004 [0.8763, 0.9229] (subset n=893) | 0.1536 [0.1433, 0.1644] | 0.0651 [0.0576, 0.0727] |
| gateway-pl | mdeberta-280m-seed1 | 1000 | 4506 | 0.8779 [0.8544, 0.8997] (subset n=893) | 0.1709 [0.1602, 0.1822] | 0.0638 [0.0566, 0.0707] |
| gateway-pl | mdeberta-280m-seed2 | 1000 | 4506 | 0.9005 [0.8793, 0.9200] (subset n=893) | 0.1860 [0.1749, 0.1976] | 0.0335 [0.0304, 0.0366] |
| gateway-pl | kp-280m | 1000 | 4506 | 0.4523 [0.4266, 0.4792] (subset n=893) | 0.3149 [0.3015, 0.3286] | 0.1868 [0.1703, 0.2032] |
| gateway-pl | presidio | 1000 | 4506 | 0.3011 [0.2909, 0.3111] (subset n=893) | 0.5384 [0.5238, 0.5529] | 0.2537 [0.2441, 0.2633] |
| gateway-pl | shield | 1000 | 4506 | 0.2316 [0.2214, 0.2422] (subset n=893) | 0.4450 [0.4305, 0.4595] | 0.3556 [0.3431, 0.3672] |
| tab-echr-legal-en | mdeberta-280m-seed0 | 127 | 3965 | 0.2888 [0.2632, 0.3169] | 0.6691 [0.6543, 0.6836] | 0.0095 [0.0082, 0.0111] |
| tab-echr-legal-en | mdeberta-280m-seed1 | 127 | 3965 | 0.1501 [0.1345, 0.1673] | 0.8318 [0.8198, 0.8431] | 0.0103 [0.0089, 0.0120] |
| tab-echr-legal-en | mdeberta-280m-seed2 | 127 | 3965 | 0.1189 [0.1036, 0.1359] | 0.8588 [0.8476, 0.8693] | 0.0038 [0.0032, 0.0045] |
| tab-echr-legal-en | kp-280m | 127 | 3965 | 0.2617 [0.2406, 0.2865] | 0.6633 [0.6484, 0.6778] | 0.0252 [0.0227, 0.0280] |
| tab-echr-legal-en | presidio | 127 | 3965 | 0.5888 [0.5620, 0.6153] | 0.3092 [0.2950, 0.3238] | 0.0244 [0.0222, 0.0269] |
| tab-echr-legal-en | shield | 127 | 3965 | 0.1442 [0.1266, 0.1637] | 0.8348 [0.8229, 0.8460] | 0.0917 [0.0850, 0.0988] |
| ro-realskeleton | mdeberta-280m-seed0 | 1500 | 13751 | 0.7424 [0.7380, 0.7467] | 0.0095 [0.0080, 0.0113] | 0.0190 [0.0179, 0.0202] |
| ro-realskeleton | mdeberta-280m-seed1 | 1500 | 13751 | 0.7933 [0.7902, 0.7963] | 0.0847 [0.0802, 0.0895] | 0.0028 [0.0024, 0.0033] |
| ro-realskeleton | mdeberta-280m-seed2 | 1500 | 13751 | 0.7880 [0.7806, 0.7950] | 0.0380 [0.0350, 0.0414] | 0.0216 [0.0200, 0.0233] |
| ro-realskeleton | kp-280m | 1500 | 13751 | 0.7411 [0.7342, 0.7475] | 0.0814 [0.0769, 0.0861] | 0.0379 [0.0362, 0.0398] |
| ro-realskeleton | presidio | 1500 | 13751 | 0.4723 [0.4668, 0.4779] | 0.3916 [0.3835, 0.3998] | 0.0728 [0.0693, 0.0763] |
| ro-realskeleton | shield | 1500 | 13751 | 0.4129 [0.4078, 0.4178] | 0.4765 [0.4681, 0.4848] | 0.2216 [0.2174, 0.2259] |
| pl-realskeleton | mdeberta-280m-seed0 | 1500 | 13885 | 0.7965 [0.7902, 0.8030] | 0.0749 [0.0706, 0.0794] | 0.0251 [0.0238, 0.0264] |
| pl-realskeleton | mdeberta-280m-seed1 | 1500 | 13885 | 0.8340 [0.8292, 0.8389] | 0.0833 [0.0788, 0.0880] | 0.0061 [0.0054, 0.0067] |
| pl-realskeleton | mdeberta-280m-seed2 | 1500 | 13885 | 0.7997 [0.7946, 0.8050] | 0.1317 [0.1261, 0.1374] | 0.0233 [0.0213, 0.0251] |
| pl-realskeleton | kp-280m | 1500 | 13885 | 0.7627 [0.7554, 0.7707] | 0.0890 [0.0844, 0.0939] | 0.0350 [0.0332, 0.0367] |
| pl-realskeleton | presidio | 1500 | 13885 | 0.4732 [0.4681, 0.4785] | 0.3160 [0.3083, 0.3237] | 0.2370 [0.2323, 0.2418] |
| pl-realskeleton | shield | 1500 | 13885 | 0.4237 [0.4150, 0.4325] | 0.4943 [0.4860, 0.5027] | 0.2925 [0.2839, 0.3011] |
| en | mdeberta-280m-seed0 | 1500 | 3619 | 0.4529 [0.4343, 0.4716] | 0.0934 [0.0843, 0.1033] | 0.0166 [0.0142, 0.0193] |
| en | mdeberta-280m-seed1 | 1500 | 3619 | 0.4939 [0.4743, 0.5119] | 0.1545 [0.1431, 0.1666] | 0.0148 [0.0127, 0.0168] |
| en | mdeberta-280m-seed2 | 1500 | 3619 | 0.4703 [0.4512, 0.4878] | 0.0788 [0.0704, 0.0880] | 0.0280 [0.0249, 0.0312] |
| en | kp-280m | 1500 | 3619 | 0.4635 [0.4445, 0.4827] | 0.1061 [0.0965, 0.1166] | 0.0591 [0.0514, 0.0697] |
| en | presidio | 1500 | 3619 | 0.4363 [0.4179, 0.4542] | 0.3169 [0.3020, 0.3323] | 0.0184 [0.0155, 0.0213] |
| en | shield | 1500 | 3619 | 0.3396 [0.3212, 0.3590] | 0.4435 [0.4274, 0.4597] | 0.0303 [0.0256, 0.0355] |
| de | mdeberta-280m-seed0 | 1500 | 3161 | 0.5210 [0.5017, 0.5417] | 0.0873 [0.0780, 0.0977] | 0.0112 [0.0094, 0.0133] |
| de | mdeberta-280m-seed1 | 1500 | 3161 | 0.5109 [0.4909, 0.5332] | 0.1253 [0.1142, 0.1373] | 0.0178 [0.0156, 0.0201] |
| de | mdeberta-280m-seed2 | 1500 | 3161 | 0.4776 [0.4584, 0.4978] | 0.0743 [0.0657, 0.0840] | 0.0223 [0.0196, 0.0251] |
| de | kp-280m | 1500 | 3161 | 0.5067 [0.4879, 0.5267] | 0.0898 [0.0804, 0.1003] | 0.0394 [0.0358, 0.0432] |
| de | presidio | 1500 | 3161 | 0.2264 [0.2133, 0.2396] | 0.3407 [0.3244, 0.3574] | 0.3240 [0.3146, 0.3337] |
| de | shield | 1500 | 3161 | 0.4077 [0.3916, 0.4246] | 0.2218 [0.2076, 0.2366] | 0.1735 [0.1655, 0.1817] |
| fr | mdeberta-280m-seed0 | 1500 | 3285 | 0.4997 [0.4794, 0.5198] | 0.1056 [0.0956, 0.1166] | 0.0112 [0.0092, 0.0136] |
| fr | mdeberta-280m-seed1 | 1500 | 3285 | 0.5202 [0.4998, 0.5404] | 0.1443 [0.1327, 0.1567] | 0.0107 [0.0089, 0.0126] |
| fr | mdeberta-280m-seed2 | 1500 | 3285 | 0.4864 [0.4667, 0.5063] | 0.0925 [0.0831, 0.1029] | 0.0192 [0.0168, 0.0218] |
| fr | kp-280m | 1500 | 3285 | 0.5095 [0.4898, 0.5286] | 0.0935 [0.0840, 0.1039] | 0.0482 [0.0438, 0.0526] |
| fr | presidio | 1500 | 3285 | 0.2874 [0.2723, 0.3037] | 0.4630 [0.4460, 0.4801] | 0.1321 [0.1224, 0.1416] |
| fr | shield | 1500 | 3285 | 0.3809 [0.3632, 0.3979] | 0.2703 [0.2554, 0.2858] | 0.0950 [0.0894, 0.1010] |
| es | mdeberta-280m-seed0 | 1500 | 3197 | 0.4565 [0.4367, 0.4763] | 0.1070 [0.0967, 0.1182] | 0.0098 [0.0083, 0.0113] |
| es | mdeberta-280m-seed1 | 1500 | 3197 | 0.4920 [0.4713, 0.5116] | 0.1486 [0.1367, 0.1613] | 0.0090 [0.0076, 0.0104] |
| es | mdeberta-280m-seed2 | 1500 | 3197 | 0.4569 [0.4382, 0.4755] | 0.0782 [0.0694, 0.0880] | 0.0205 [0.0180, 0.0230] |
| es | kp-280m | 1500 | 3197 | 0.4717 [0.4516, 0.4915] | 0.1051 [0.0949, 0.1162] | 0.0532 [0.0488, 0.0576] |
| es | presidio | 1500 | 3197 | 0.2377 [0.2223, 0.2527] | 0.4923 [0.4750, 0.5097] | 0.1830 [0.1712, 0.1942] |
| es | shield | 1500 | 3197 | 0.3666 [0.3490, 0.3834] | 0.2599 [0.2450, 0.2754] | 0.1682 [0.1610, 0.1760] |
| it | mdeberta-280m-seed0 | 1500 | 3244 | 0.4456 [0.4251, 0.4658] | 0.0906 [0.0812, 0.1010] | 0.0111 [0.0095, 0.0128] |
| it | mdeberta-280m-seed1 | 1500 | 3244 | 0.4822 [0.4615, 0.5016] | 0.1289 [0.1178, 0.1408] | 0.0107 [0.0092, 0.0124] |
| it | mdeberta-280m-seed2 | 1500 | 3244 | 0.4305 [0.4112, 0.4497] | 0.0808 [0.0719, 0.0906] | 0.0197 [0.0174, 0.0222] |
| it | kp-280m | 1500 | 3244 | 0.4610 [0.4421, 0.4801] | 0.0845 [0.0754, 0.0945] | 0.0500 [0.0460, 0.0542] |
| it | presidio | 1500 | 3244 | 0.2354 [0.2211, 0.2498] | 0.4427 [0.4256, 0.4598] | 0.2421 [0.2310, 0.2533] |
| it | shield | 1500 | 3244 | 0.3184 [0.3020, 0.3342] | 0.2512 [0.2366, 0.2664] | 0.1911 [0.1828, 0.1990] |
| nl | mdeberta-280m-seed0 | 1500 | 3125 | 0.5133 [0.4935, 0.5331] | 0.1037 [0.0935, 0.1149] | 0.0150 [0.0127, 0.0174] |
| nl | mdeberta-280m-seed1 | 1500 | 3125 | 0.5271 [0.5078, 0.5474] | 0.1603 [0.1479, 0.1736] | 0.0135 [0.0116, 0.0157] |
| nl | mdeberta-280m-seed2 | 1500 | 3125 | 0.4967 [0.4774, 0.5169] | 0.0752 [0.0665, 0.0850] | 0.0252 [0.0223, 0.0285] |
| nl | kp-280m | 1500 | 3125 | 0.5243 [0.5049, 0.5438] | 0.1008 [0.0907, 0.1119] | 0.0585 [0.0530, 0.0648] |
| nl | presidio | 1500 | 3125 | 0.2305 [0.2170, 0.2440] | 0.3411 [0.3247, 0.3579] | 0.2801 [0.2681, 0.2920] |
| nl | shield | 1500 | 3125 | 0.5081 [0.4895, 0.5279] | 0.2291 [0.2147, 0.2442] | 0.0286 [0.0246, 0.0334] |

## Worst gateway language

| Model | Completed languages | Worst language | All-PII residual v2 [Wilson CI] |
|---|---:|---|---|
| mdeberta-280m-seed0 | 8 | en | 0.1636 [0.1530, 0.1746] |
| mdeberta-280m-seed1 | 8 | en | 0.2259 [0.2139, 0.2384] |
| mdeberta-280m-seed2 | 8 | en | 0.2082 [0.1966, 0.2203] |
| kp-280m | 8 | en | 0.3693 [0.3553, 0.3835] |
| presidio | 8 | ro | 0.6116 [0.5973, 0.6258] |
| shield | 8 | en | 0.5419 [0.5274, 0.5564] |

## Legacy national-ID anchors

| Candidate | Config | Subjects | Leak rate [Wilson CI] | Verdict |
|---|---|---:|---|---|
| mdeberta-280m-seed0 | ro-realskeleton | 1123.0 | 0.0000 [0.0000, 0.0034] | tie |
| mdeberta-280m-seed0 | pl-realskeleton | 1096.0 | 0.0000 [0.0000, 0.0035] | tie |
| mdeberta-280m-seed1 | ro-realskeleton | 1123.0 | 0.0000 [0.0000, 0.0034] | tie |
| mdeberta-280m-seed1 | pl-realskeleton | 1096.0 | 0.0000 [0.0000, 0.0035] | tie |
| mdeberta-280m-seed2 | ro-realskeleton | 1123.0 | 0.0000 [0.0000, 0.0034] | tie |
| mdeberta-280m-seed2 | pl-realskeleton | 1096.0 | 0.0000 [0.0000, 0.0035] | tie |

## Post-hoc exact-character national-ID diagnostic

This supplementary check was added after scoring to distinguish legacy any-overlap detection
from complete coverage. It does not change model selection, primary gold or acceptance endpoints.
Both scopes retain all 1,500 documents and the same native intervals; rates use 95% Wilson CIs.
The validator-valid scope uses the pinned legacy CNP/PESEL validators across all gold labels.
The broader NATIONAL_ID scope includes every span with that gold label, including other values.

| Config / scope | Subjects | seed0 | seed1 | seed2 | kp-280m | presidio | shield |
|---|---|---|---|---|---|---|---|
| pl-realskeleton / all_national_id_gold | 1443 | 0.1635 [0.1454, 0.1835] | 0.0471 [0.0373, 0.0593] | 0.1074 [0.0925, 0.1244] | 0.1220 [0.1061, 0.1399] | 0.2148 [0.1944, 0.2368] | 0.9958 [0.9910, 0.9981] |
| pl-realskeleton / validator_valid_decode_bearing | 1096 | 0.0000 [0.0000, 0.0035] | 0.0000 [0.0000, 0.0035] | 0.0000 [0.0000, 0.0035] | 0.0201 [0.0133, 0.0302] | 0.0000 [0.0000, 0.0035] | 1.0000 [0.9965, 1.0000] |
| ro-realskeleton / all_national_id_gold | 1907 | 0.0000 [0.0000, 0.0020] | 0.0000 [0.0000, 0.0020] | 0.0000 [0.0000, 0.0020] | 0.0000 [0.0000, 0.0020] | 0.3728 [0.3514, 0.3948] | 0.9990 [0.9962, 0.9997] |
| ro-realskeleton / validator_valid_decode_bearing | 1123 | 0.0000 [0.0000, 0.0034] | 0.0000 [0.0000, 0.0034] | 0.0000 [0.0000, 0.0034] | 0.0000 [0.0000, 0.0034] | 0.0000 [0.0000, 0.0034] | 1.0000 [0.9966, 1.0000] |

## Non-gateway diagnostic losses

These regressions limit general-domain conclusions despite any gateway improvements.
Per-seed estimates, n and CIs appear above; all paired deltas are in JSON.

| Config | Comparator | Seeds with privacy loss | Seeds with utility loss |
|---|---|---|---|
| de | kp-280m | 1 | — |
| en | kp-280m | 1 | — |
| en | presidio | — | 2 |
| es | kp-280m | 1 | — |
| fr | kp-280m | 1 | — |
| it | kp-280m | 1 | — |
| nl | kp-280m | 1 | — |
| pl-realskeleton | kp-280m | 2 | — |
| tab-echr-legal-en | kp-280m | 1, 2 | — |
| tab-echr-legal-en | presidio | 0, 1, 2 | — |

## Contract verdicts

Source: [frozen gateway contract v1](https://github.com/klusai/europriv-bench/blob/4ce4a3290731c5e8b0f27185339680fd5ca9433a/docs/gateway-contract.md).

| Contract line | Verdict | Evidence / limit |
|---|---|---|
| §1 exact residual v2 | measured | Harness exact-character coverage; identifier subjects, not empirical attacker re-identification |
| §2 operator and gold pins | recorded | Raw native offsets, operator receipts and ordered gold hashes in scratch |
| §2 deployed operator parity and replacement-content attestation | unverified | Pinned local operator experiment; replacement fragment-copy audit and deployed parity not established |
| §3 legacy RO/PL zero floors | tie | Every observed new decode-bearing national-ID leak fails; see anchor table |
| §3.1 three training seeds + disjoint Wilson | pass | 3 completed candidate seeds; each gateway seed/language/comparator must pass separately |
| §3.2 paired cluster delta | pass | 2000 paired resamples, RNG125; gateway delta upper<0 required; all deltas in JSON |
| §3.3 no-worse utility + anchors | pass | Gateway non-PII non-whitespace character delta upper<=0, plus preserved legacy RO/PL zero floors |
| §3.4 all eight languages | pass | Numeric gateway gates only; reviewed/disjoint gold is the separate gate below; no pooled acceptance |
| §3.4 reviewed/disjoint gold | unverified | Native review and legacy semantic disjointness remain open |
| §4 target-CPU short-lane load protocol | unverified | No open-loop three-by-ten-minute trials, frozen <=256-token replay, or 16 offered req/s load point |
| §4 p95 point and bootstrap upper <=250 ms | unverified | No end-to-end HTTP latency trials; training/inference runtime is not serving latency |
| §4 successful completed throughput >=15 req/s in every trial | unverified | No registered load trials |
| §4 errors + timeouts + overload refusals <=0.1% | unverified | No offered-request failure accounting |
| §4 1000 ms deadline and fail-closed dispatch | unverified | No gateway deadline or cancellation trials |
| §4 zero silent bypass/truncation | unverified | Gateway evaluation uses full-input overflow; anchors retain 512-token cap; deployment untested |
| §4 long-input lane, admission and mixed-traffic interference | unverified | No separate 257-2048-token SLO or unsupported-payload refusal/load trials |
| Integration, rollback, pseudonym maps and semantic utility | unverified | Local detector/operator experiment only |

The new-configuration numeric gates above concern gateway traffic and the legacy national-ID floors.
TAB/general/anchor diagnostics additionally contain 13 seed/config/comparator
comparisons with a privacy or utility loss; every such row and its paired CI is retained in JSON.
These diagnostics remain limitations, not gateway-wide or general-domain acceptance claims.

A numeric per-checkpoint privacy win requires disjoint Wilson intervals AND paired delta upper<0.
A privacy tie means superiority was not established; it is not a statistical equivalence finding.
A scoped contract win additionally requires all three seeds, no-worse utility, preserved anchors and
reviewed/disjoint gold. Missing review alone prevents a scoped win. Every DIRECT/QUASI/UNSPECIFIED
slice and comparator delta is retained in JSON; empty support is unavailable, never a measured zero.

## Limitations and reproduction

Missing model/config rows: 0. Synthetic gold has no independent native review. The gateway
TRAIN audit covers new train versus dev/test; legacy-corpus semantic disjointness is unverified.
All original KP training rows remain; the general configurations are not certified clean held-out data.
Training truncates at 512; gateway test inference screens overflow. The train export uses two newlines
between parts; the harness uses its frozen single-newline coordinate policy on dev/test.
No baseline repeat-variance, all-document CPU/MPS equivalence, serving latency, rollback or semantic-utility verification.

Seed 0 resumed after execution-profile changes; equality to an uninterrupted training trajectory was not tested.
The exact executed trainer is archived in scratch; the committed copy has an AST-identical line wrap.
The JSON source-formatting audit records both hashes and the archived source needed for the original resume pins.

Optional XLM-R-560M was deferred before candidate test scoring to complete the mDeBERTa replication
and evaluation matrix. No XLM-R result or cross-architecture conclusion is claimed.

Run `scripts/train_gateway_res127.py prepare` then `train` with pinned local base, data and scratch paths.
Use the RES-126 harness on PYTHONPATH; `scripts/score_gateway_res127.py freeze` imports frozen RES-125
anchors and loads local gateway test through the RES-126 override. `predict` scores each selected checkpoint
and each gateway comparator; `scripts/report_gateway_res127.py` assembles this report without selection.
`scripts/audit_national_ids_res127.py` replays the optional post-hoc national-ID coverage diagnostic.
Exact commands, logs, checkpoints and native receipts are retained in the assigned RES-127 scratch directory.

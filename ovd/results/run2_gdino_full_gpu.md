# Owner's Colab L4 output, pasted 23 Nov 2026 (code 0f8a9af / 668d666). Verbatim tables from full_summary.md.
## coco / gdino: 1000 images, 5013 pairs, 120312 runs

| group | mAP50 dep | mAP50 indep | ΔmAP vs k1 95% CI | mean |Δs| | mean Δs | det flips | loc flips | FA dep | chunk |
|---|---|---|---|---|---|---|---|---|---|
| k1 | 0.707 | 0.707 |  | 0.0000 | +0.0000 | 0.000 | 0.000 | 0.179 | 1.0 |
| rand5 | 0.795 | 0.796 | [+0.078, +0.095] | 0.0865 | +0.0465 | 0.166 | 0.157 | 0.007 | 5.0 |
| rand20 | 0.795 | 0.799 | [+0.076, +0.095] | 0.1111 | +0.0644 | 0.192 | 0.184 | 0.003 | 20.0 |
| far20 | 0.796 | 0.796 | [+0.077, +0.097] | 0.1048 | +0.0517 | 0.188 | 0.178 | 0.002 | 20.0 |
| far5 | 0.796 | 0.796 | [+0.077, +0.096] | 0.0820 | +0.0385 | 0.160 | 0.152 | 0.004 | 5.0 |
| near20 | 0.790 | 0.803 | [+0.070, +0.091] | 0.1250 | +0.0872 | 0.199 | 0.192 | 0.004 | 20.0 |
| near5 | 0.784 | 0.797 | [+0.064, +0.084] | 0.0983 | +0.0602 | 0.179 | 0.173 | 0.008 | 5.0 |
| vis20 | 0.793 | 0.804 | [+0.072, +0.094] | 0.1294 | +0.0989 | 0.198 | 0.196 | 0.007 | 20.0 |
| vis5 | 0.787 | 0.794 | [+0.068, +0.087] | 0.1071 | +0.0802 | 0.179 | 0.169 | 0.028 | 5.0 |
| all | 0.792 | 0.805 | [+0.071, +0.094] | 0.1401 | +0.1092 | 0.205 | 0.204 | 0.004 | 80.0 |
| prune | 0.689 | 0.697 | [-0.041, -0.008] | 0.1854 | +0.0144 | 0.250 | 0.374 | 0.006 | 10.0 |
| FIX | 0.805 | 0.805 | [+0.085, +0.106] | 0.0000 | +0.0000 | 0.000 | 0.000 | 0.004 | 80.0 |

order-only flips (5 orders of the full list): 0.000, mean score range 0.0002
instances whose detection flips under some vocabulary: 0.278, mean score range 0.1948
FIX - k1 mAP50 (deployed): +0.0975, 95% CI [+0.0849, +0.1055]
FIX - all mAP50 (deployed): +0.0129, 95% CI [+0.0093, +0.0173]
FIX - prune mAP50 (deployed): +0.1157, 95% CI [+0.1058, +0.1329]
interference orig: score 0.1094, flips 0.185; FIX: score 0.0000, flips 0.000
probe (n=7342): rand20_0 mean_abs_change_full 0.11038 vs qs_target 0.11070; all_0 0.14012 vs 0.14188 (signed 0.10924 vs 0.11199)

## lvis / gdino: 600 images, 2689 pairs, 86048 runs

| group | mAP50 dep | mAP50 indep | ΔmAP vs k1 95% CI | mean |Δs| | mean Δs | det flips | loc flips | FA dep | chunk |
|---|---|---|---|---|---|---|---|---|---|
| k1 | 0.496 | 0.496 |  | 0.0000 | +0.0000 | 0.000 | 0.000 | 0.559 | 1.0 |
| rand5 | 0.571 | 0.580 | [+0.056, +0.084] | 0.0695 | -0.0427 | 0.101 | 0.306 | 0.022 | 5.0 |
| rand20 | 0.543 | 0.568 | [+0.024, +0.058] | 0.0972 | -0.0779 | 0.136 | 0.378 | 0.002 | 20.0 |
| rand80 | 0.506 | 0.558 | [-0.014, +0.021] | 0.0919 | -0.0620 | 0.125 | 0.401 | 0.001 | 80.0 |
| rand200 | 0.508 | 0.561 | [-0.012, +0.023] | 0.0924 | -0.0626 | 0.127 | 0.399 | 0.001 | 81.2 |
| rand500 | 0.500 | 0.560 | [-0.020, +0.015] | 0.0917 | -0.0616 | 0.126 | 0.398 | 0.001 | 85.1 |
| far20 | 0.556 | 0.562 | [+0.033, +0.067] | 0.0981 | -0.0802 | 0.139 | 0.380 | 0.002 | 20.0 |
| far5 | 0.576 | 0.581 | [+0.060, +0.091] | 0.0714 | -0.0459 | 0.098 | 0.306 | 0.026 | 5.0 |
| far80 | 0.519 | 0.550 | [-0.002, +0.036] | 0.0931 | -0.0648 | 0.126 | 0.384 | 0.004 | 79.8 |
| near20 | 0.446 | 0.578 | [-0.077, -0.038] | 0.0924 | -0.0629 | 0.129 | 0.416 | 0.001 | 20.0 |
| near5 | 0.497 | 0.599 | [-0.026, +0.011] | 0.0737 | -0.0291 | 0.113 | 0.361 | 0.026 | 5.0 |
| near80 | 0.424 | 0.576 | [-0.099, -0.058] | 0.0920 | -0.0542 | 0.130 | 0.428 | 0.003 | 80.0 |
| vis20 | 0.478 | 0.558 | [-0.042, -0.006] | 0.0935 | -0.0653 | 0.133 | 0.394 | 0.003 | 20.0 |
| vis5 | 0.517 | 0.577 | [+0.002, +0.036] | 0.0736 | -0.0314 | 0.114 | 0.354 | 0.052 | 5.0 |
| vis80 | 0.434 | 0.556 | [-0.089, -0.047] | 0.0947 | -0.0620 | 0.132 | 0.424 | 0.001 | 80.0 |
| all | 0.508 | 0.561 | [-0.012, +0.023] | 0.0920 | -0.0611 | 0.126 | 0.402 | 0.001 | 87.0 |
| prune | 0.256 | 0.316 | [-0.279, -0.235] | 0.1300 | -0.1091 | 0.138 | 0.642 | 0.001 | 50.0 |
| FIX | 0.558 | 0.558 | [+0.040, +0.076] | 0.0000 | +0.0000 | 0.000 | 0.000 | 0.001 | 87.3 |

order-only flips (5 orders of the full list): 0.036, mean score range 0.0292
instances whose detection flips under some vocabulary: 0.218, mean score range 0.1528
FIX - k1 mAP50 (deployed): +0.0624, 95% CI [+0.0402, +0.0760]
FIX - all mAP50 (deployed): +0.0502, 95% CI [+0.0428, +0.0665]
FIX - prune mAP50 (deployed): +0.3023, 95% CI [+0.2924, +0.3395]
interference orig: score 0.0878, flips 0.124; FIX: score 0.0000, flips 0.000
probe (n=4798): rand20_0 0.09752 vs 0.09742; all_0 0.09124 vs 0.09115

## lvis latency (s / image, NVIDIA L4)
- gdino: orig/full 4.284, orig/rand5 0.363, orig/rand20 0.369, orig/rand80 0.410, orig/rand200 0.962, fixctx/full 4.278, fixctx/rand5 1.613, fixctx/rand20 3.380, fixctx/rand80 4.279, fixctx/rand200 4.275, single/rand5 1.286, single/rand20 4.801, single/rand80 18.831
- yoloworld: orig/full 0.045, orig/rand5 0.014, orig/rand20 0.015, orig/rand80 0.015, orig/rand200 0.016, fixguide/full 0.045, fixguide/rand5 0.016, fixguide/rand20 0.015, fixguide/rand80 0.016, fixguide/rand200 0.017, single/rand5 0.045, single/rand20 0.155, single/rand80 0.605
- owlv2: orig/full 0.457, orig/rand5 0.170, orig/rand20 0.171, orig/rand80 0.181, orig/rand200 0.205, single/rand5 0.851, single/rand20 3.435, single/rand80 13.829

## independent_check.py
coco/gdino k1 0.7072 / all 0.7919 / FIX 0.8048; lvis/gdino k1 0.4957 / all 0.5078 / FIX 0.5581 — 6 values checked; largest absolute difference 1.11e-16
## plan md5: plan_coco f92528394033c5b07545914e745113fd, plan_lvis 22114501f7a6a38314b3a9033cac7e02 (unchanged since the runs started)

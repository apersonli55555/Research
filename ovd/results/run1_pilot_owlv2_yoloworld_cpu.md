
VOCABULARY-INTERFERENCE PILOT (OVD run 1)
GO if, for at least one model: (a) |mAP50(k=1) - mAP50(all 80 classes)| >= 0.03 with a 95% CI (by image) excluding zero; (b) >= 10% of target instances change detected / missed between k=1 and all 80 classes (either way; mean over orders); or (c) >= 10% of target instances change detected / missed between orders of the same 80 classes

owlv2 (threshold 0.1; 2471 image-target pairs; 3722 present-target instances)
  mAP50:                         k1 0.746, rand5 0.744, near5 0.729, rand20 0.742, near20 0.727, all80 0.726
  mAP50 k=1 minus all 80:        +0.020 [+0.013, +0.027]
  detected rate (threshold):     k1 0.805, rand5 0.802, near5 0.782, rand20 0.797, near20 0.780, all80 0.777
  mean raw target score:         k1 0.284, rand5 0.284, near5 0.284, rand20 0.284, near20 0.284, all80 0.284
  mean |score change| vs k=1:    k1 0.000, rand5 0.000, near5 0.000, rand20 0.000, near20 0.000, all80 0.000
  detection flips vs k=1:        k1 0.000, rand5 0.002, near5 0.022, rand20 0.007, near20 0.025, all80 0.028
  best-box localisation flips:   k1 0.000, rand5 0.000, near5 0.000, rand20 0.000, near20 0.000, all80 0.000
  false alarms, absent targets:  k1 0.045, rand5 0.044, near5 0.036, rand20 0.041, near20 0.033, all80 0.033
  seconds per run:               k1 1.392, rand5 0.000, near5 0.000, rand20 0.000, near20 0.000, all80 0.000
  detected with k=1 but missed with all 80: 0.035; order-only flips among the 5 orders of all 80: 0.000 (mean score range 0.0000)
  go criteria: a False, b False, c False

yoloworld (threshold 0.25; 2471 image-target pairs; 3717 present-target instances)
  mAP50:                         k1 0.598, rand5 0.609, near5 0.596, rand20 0.608, near20 0.600, all80 0.600
  mAP50 k=1 minus all 80:        -0.003 [-0.012, +0.010]
  detected rate (threshold):     k1 0.531, rand5 0.553, near5 0.541, rand20 0.560, near20 0.553, all80 0.557
  mean raw target score:         k1 0.386, rand5 0.395, near5 0.383, rand20 0.397, near20 0.388, all80 0.391
  mean |score change| vs k=1:    k1 0.000, rand5 0.032, near5 0.034, rand20 0.042, near20 0.048, all80 0.052
  detection flips vs k=1:        k1 0.000, rand5 0.040, near5 0.041, rand20 0.051, near20 0.059, all80 0.064
  best-box localisation flips:   k1 0.000, rand5 0.032, near5 0.040, rand20 0.049, near20 0.058, all80 0.068
  false alarms, absent targets:  k1 0.024, rand5 0.018, near5 0.013, rand20 0.015, near20 0.012, all80 0.011
  seconds per run:               k1 0.287, rand5 0.282, near5 0.283, rand20 0.419, near20 0.291, all80 0.288
  detected with k=1 but missed with all 80: 0.035; order-only flips among the 5 orders of all 80: 0.000 (mean score range 0.0000)
  go criteria: a False, b False, c False

PILOT RESULT: STOP

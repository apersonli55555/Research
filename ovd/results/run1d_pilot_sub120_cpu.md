
VOCABULARY-INTERFERENCE PILOT (OVD run 1)
GO if, for at least one model: (a) |mAP50(k=1) - mAP50(all 80 classes)| >= 0.03 with a 95% CI (by image) excluding zero; (b) >= 10% of target instances change detected / missed between k=1 and all 80 classes (either way; mean over orders); or (c) >= 10% of target instances change detected / missed between orders of the same 80 classes

gdino (threshold 0.35; 120 image-target pairs; 156 present-target instances)
  mAP50:                         k1 0.836, rand5 0.886, near5 0.868, rand20 0.874, near20 0.843, all80 0.835
  mAP50 k=1 minus all 80:        +0.001 [-0.029, +0.025]
  detected rate (threshold):     k1 0.641, rand5 0.740, near5 0.756, rand20 0.717, near20 0.724, all80 0.718
  mean raw target score:         k1 0.432, rand5 0.476, near5 0.488, rand20 0.484, near20 0.507, all80 0.524
  mean |score change| vs k=1:    k1 0.000, rand5 0.094, near5 0.097, rand20 0.114, near20 0.127, all80 0.142
  detection flips vs k=1:        k1 0.000, rand5 0.137, near5 0.128, rand20 0.165, near20 0.173, all80 0.154
  best-box localisation flips:   k1 0.000, rand5 0.055, near5 0.064, rand20 0.083, near20 0.077, all80 0.071
  false alarms, absent targets:  k1 0.146, rand5 0.000, near5 0.021, rand20 0.000, near20 0.000, all80 0.000
  seconds per run:               k1 5.186, rand5 4.693, near5 4.663, rand20 4.841, near20 4.791, all80 5.827
  detected with k=1 but missed with all 80: 0.060; order-only flips among the 5 orders of all 80: 0.000 (mean score range 0.0001)
  go criteria: a False, b True, c False

owlv2 (threshold 0.1; 120 image-target pairs; 156 present-target instances)
  mAP50:                         k1 0.870, rand5 0.867, near5 0.850, rand20 0.863, near20 0.851, all80 0.852
  mAP50 k=1 minus all 80:        +0.018 [+0.005, +0.058]
  detected rate (threshold):     k1 0.801, rand5 0.799, near5 0.776, rand20 0.795, near20 0.776, all80 0.776
  mean raw target score:         k1 0.301, rand5 0.301, near5 0.301, rand20 0.301, near20 0.301, all80 0.301
  mean |score change| vs k=1:    k1 0.000, rand5 0.000, near5 0.000, rand20 0.000, near20 0.000, all80 0.000
  detection flips vs k=1:        k1 0.000, rand5 0.003, near5 0.026, rand20 0.006, near20 0.026, all80 0.026
  best-box localisation flips:   k1 0.000, rand5 0.000, near5 0.000, rand20 0.000, near20 0.000, all80 0.000
  false alarms, absent targets:  k1 0.062, rand5 0.062, near5 0.042, rand20 0.058, near20 0.042, all80 0.042
  seconds per run:               k1 15.185, rand5 0.000, near5 0.000, rand20 0.000, near20 0.000, all80 0.000
  detected with k=1 but missed with all 80: 0.032; order-only flips among the 5 orders of all 80: 0.000 (mean score range 0.0000)
  go criteria: a False, b False, c False

yoloworld (threshold 0.25; 120 image-target pairs; 156 present-target instances)
  mAP50:                         k1 0.746, rand5 0.724, near5 0.719, rand20 0.713, near20 0.705, all80 0.702
  mAP50 k=1 minus all 80:        +0.044 [-0.011, +0.095]
  detected rate (threshold):     k1 0.628, rand5 0.622, near5 0.609, rand20 0.618, near20 0.622, all80 0.609
  mean raw target score:         k1 0.439, rand5 0.444, near5 0.420, rand20 0.437, near20 0.424, all80 0.426
  mean |score change| vs k=1:    k1 0.000, rand5 0.043, near5 0.045, rand20 0.058, near20 0.063, all80 0.068
  detection flips vs k=1:        k1 0.000, rand5 0.045, near5 0.045, rand20 0.059, near20 0.071, all80 0.071
  best-box localisation flips:   k1 0.000, rand5 0.038, near5 0.058, rand20 0.053, near20 0.064, all80 0.071
  false alarms, absent targets:  k1 0.000, rand5 0.000, near5 0.000, rand20 0.000, near20 0.000, all80 0.000
  seconds per run:               k1 0.295, rand5 0.250, near5 0.264, rand20 0.257, near20 0.267, all80 0.259
  detected with k=1 but missed with all 80: 0.071; order-only flips among the 5 orders of all 80: 0.000 (mean score range 0.0000)
  go criteria: a False, b False, c False

PILOT RESULT: GO

# SIMULATED DATA: pipeline test only, NOT a result

Calibration range for the restricted design: 15.0 to 30.0 degC. Target coverage 90 %.

## Coverage of 90 % intervals

| design     | region   | model            |   n_blocks |   coverage_split |   coverage_weighted |   weighted_refused |   worst_cell_coverage |
|:-----------|:---------|:-----------------|-----------:|-----------------:|--------------------:|-------------------:|----------------------:|
| full       | cold     | Boosted trees    |       6228 |            0.858 |               0.958 |              0     |                 0.173 |
| full       | cold     | Cubic            |       6228 |            0.865 |               0.932 |              0     |                 0.378 |
| full       | cold     | Linear           |       6228 |            0.749 |               0.808 |              0     |                 0.127 |
| full       | cold     | Quadratic        |       6228 |            0.891 |               0.897 |              0     |                 0.236 |
| full       | cold     | Quadratic + rate |       6228 |            0.843 |               0.835 |              0     |                 0.285 |
| full       | inside   | Boosted trees    |      36672 |            0.946 |               0.985 |              0     |                 0.568 |
| full       | inside   | Cubic            |      36672 |            0.891 |               0.955 |              0     |                 0     |
| full       | inside   | Linear           |      36672 |            0.931 |               0.955 |              0     |                 0     |
| full       | inside   | Quadratic        |      36672 |            0.897 |               0.947 |              0     |                 0.015 |
| full       | inside   | Quadratic + rate |      36672 |            0.932 |               0.95  |              0     |                 0.125 |
| full       | warm     | Boosted trees    |       5484 |            0.922 |               1     |              0.002 |                 0.734 |
| full       | warm     | Cubic            |       5484 |            0.863 |               0.957 |              0.002 |                 0.431 |
| full       | warm     | Linear           |       5484 |            0.579 |               0.925 |              0.002 |                 0     |
| full       | warm     | Quadratic        |       5484 |            0.808 |               0.946 |              0.002 |                 0.378 |
| full       | warm     | Quadratic + rate |       5484 |            0.837 |               0.94  |              0.002 |                 0.247 |
| restricted | cold     | Boosted trees    |       6228 |            0.215 |               0.796 |              0.827 |                 0.063 |
| restricted | cold     | Cubic            |       6228 |            0.133 |               0.528 |              0.827 |                 0     |
| restricted | cold     | Linear           |       6228 |            0.309 |               0.525 |              0.827 |                 0     |
| restricted | cold     | Quadratic        |       6228 |            0.442 |               0.809 |              0.827 |                 0     |
| restricted | cold     | Quadratic + rate |       6228 |            0.446 |               0.817 |              0.827 |                 0     |
| restricted | inside   | Boosted trees    |      36672 |            0.9   |               0.922 |              0     |                 0.271 |
| restricted | inside   | Cubic            |      36672 |            0.88  |               0.926 |              0     |                 0.121 |
| restricted | inside   | Linear           |      36672 |            0.88  |               0.886 |              0     |                 0.01  |
| restricted | inside   | Quadratic        |      36672 |            0.895 |               0.877 |              0     |                 0.145 |
| restricted | inside   | Quadratic + rate |      36672 |            0.92  |               0.889 |              0     |                 0.287 |
| restricted | warm     | Boosted trees    |       5484 |            0.293 |               0.81  |              0.856 |                 0     |
| restricted | warm     | Cubic            |       5484 |            0.268 |               0.695 |              0.856 |                 0     |
| restricted | warm     | Linear           |       5484 |            0.538 |               0.716 |              0.856 |                 0     |
| restricted | warm     | Quadratic        |       5484 |            0.194 |               0.623 |              0.856 |                 0     |
| restricted | warm     | Quadratic + rate |       5484 |            0.228 |               0.646 |              0.856 |                 0     |

## Error before and after compensation (median RMS over folds)

| design     | channel   | region   | model            |   raw_rms |   resid_rms |
|:-----------|:----------|:---------|:-----------------|----------:|------------:|
| full       | anorm     | cold     | Boosted trees    |    4.3119 |      0.3409 |
| full       | anorm     | cold     | Cubic            |    4.3119 |      0.3124 |
| full       | anorm     | cold     | Linear           |    4.3119 |      0.4754 |
| full       | anorm     | cold     | Quadratic        |    4.3119 |      0.2419 |
| full       | anorm     | cold     | Quadratic + rate |    4.3119 |      0.1891 |
| full       | anorm     | inside   | Boosted trees    |    0.7442 |      0.3052 |
| full       | anorm     | inside   | Cubic            |    0.7442 |      0.2792 |
| full       | anorm     | inside   | Linear           |    0.7442 |      0.4026 |
| full       | anorm     | inside   | Quadratic        |    0.7442 |      0.273  |
| full       | anorm     | inside   | Quadratic + rate |    0.7442 |      0.2818 |
| full       | anorm     | warm     | Boosted trees    |    2.6676 |      0.3312 |
| full       | anorm     | warm     | Cubic            |    2.6676 |      0.3993 |
| full       | anorm     | warm     | Linear           |    2.6676 |      0.9241 |
| full       | anorm     | warm     | Quadratic        |    2.6676 |      0.416  |
| full       | anorm     | warm     | Quadratic + rate |    2.6676 |      0.4689 |
| full       | gx        | cold     | Boosted trees    |    0.077  |      0.0088 |
| full       | gx        | cold     | Cubic            |    0.077  |      0.0077 |
| full       | gx        | cold     | Linear           |    0.077  |      0.0168 |
| full       | gx        | cold     | Quadratic        |    0.077  |      0.007  |
| full       | gx        | cold     | Quadratic + rate |    0.077  |      0.0076 |
| full       | gx        | inside   | Boosted trees    |    0.0085 |      0.0057 |
| full       | gx        | inside   | Cubic            |    0.0085 |      0.0057 |
| full       | gx        | inside   | Linear           |    0.0085 |      0.0093 |
| full       | gx        | inside   | Quadratic        |    0.0085 |      0.0063 |
| full       | gx        | inside   | Quadratic + rate |    0.0085 |      0.0067 |
| full       | gx        | warm     | Boosted trees    |    0.0239 |      0.0059 |
| full       | gx        | warm     | Cubic            |    0.0239 |      0.0043 |
| full       | gx        | warm     | Linear           |    0.0239 |      0.0468 |
| full       | gx        | warm     | Quadratic        |    0.0239 |      0.0085 |
| full       | gx        | warm     | Quadratic + rate |    0.0239 |      0.0084 |
| full       | gy        | cold     | Boosted trees    |    0.1954 |      0.0125 |
| full       | gy        | cold     | Cubic            |    0.1954 |      0.0207 |
| full       | gy        | cold     | Linear           |    0.1954 |      0.015  |
| full       | gy        | cold     | Quadratic        |    0.1954 |      0.0174 |
| full       | gy        | cold     | Quadratic + rate |    0.1954 |      0.0194 |
| full       | gy        | inside   | Boosted trees    |    0.0342 |      0.0124 |
| full       | gy        | inside   | Cubic            |    0.0342 |      0.0115 |
| full       | gy        | inside   | Linear           |    0.0342 |      0.0119 |
| full       | gy        | inside   | Quadratic        |    0.0342 |      0.0117 |
| full       | gy        | inside   | Quadratic + rate |    0.0342 |      0.0118 |
| full       | gy        | warm     | Boosted trees    |    0.1495 |      0.0166 |
| full       | gy        | warm     | Cubic            |    0.1495 |      0.0197 |
| full       | gy        | warm     | Linear           |    0.1495 |      0.0215 |
| full       | gy        | warm     | Quadratic        |    0.1495 |      0.0224 |
| full       | gy        | warm     | Quadratic + rate |    0.1495 |      0.0188 |
| full       | gz        | cold     | Boosted trees    |    0.1355 |      0.007  |
| full       | gz        | cold     | Cubic            |    0.1355 |      0.0085 |
| full       | gz        | cold     | Linear           |    0.1355 |      0.0232 |
| full       | gz        | cold     | Quadratic        |    0.1355 |      0.0081 |
| full       | gz        | cold     | Quadratic + rate |    0.1355 |      0.0086 |
| full       | gz        | inside   | Boosted trees    |    0.0302 |      0.0089 |
| full       | gz        | inside   | Cubic            |    0.0302 |      0.012  |
| full       | gz        | inside   | Linear           |    0.0302 |      0.0106 |
| full       | gz        | inside   | Quadratic        |    0.0302 |      0.0124 |
| full       | gz        | inside   | Quadratic + rate |    0.0302 |      0.0125 |
| full       | gz        | warm     | Boosted trees    |    0.1488 |      0.0156 |
| full       | gz        | warm     | Cubic            |    0.1488 |      0.0131 |
| full       | gz        | warm     | Linear           |    0.1488 |      0.0477 |
| full       | gz        | warm     | Quadratic        |    0.1488 |      0.0185 |
| full       | gz        | warm     | Quadratic + rate |    0.1488 |      0.0184 |
| restricted | anorm     | cold     | Boosted trees    |    4.3119 |      1.8994 |
| restricted | anorm     | cold     | Cubic            |    4.3119 |      4.6014 |
| restricted | anorm     | cold     | Linear           |    4.3119 |      1.1859 |
| restricted | anorm     | cold     | Quadratic        |    4.3119 |      0.9073 |
| restricted | anorm     | cold     | Quadratic + rate |    4.3119 |      0.9319 |
| restricted | anorm     | inside   | Boosted trees    |    0.7442 |      0.3065 |
| restricted | anorm     | inside   | Cubic            |    0.7442 |      0.3125 |
| restricted | anorm     | inside   | Linear           |    0.7442 |      0.323  |
| restricted | anorm     | inside   | Quadratic        |    0.7442 |      0.2984 |
| restricted | anorm     | inside   | Quadratic + rate |    0.7442 |      0.2909 |
| restricted | anorm     | warm     | Boosted trees    |    2.6676 |      1.7833 |
| restricted | anorm     | warm     | Cubic            |    2.6676 |      2.5501 |
| restricted | anorm     | warm     | Linear           |    2.6676 |      0.4227 |
| restricted | anorm     | warm     | Quadratic        |    2.6676 |      2.9872 |
| restricted | anorm     | warm     | Quadratic + rate |    2.6676 |      3.0283 |
| restricted | gx        | cold     | Boosted trees    |    0.077  |      0.0415 |
| restricted | gx        | cold     | Cubic            |    0.077  |      0.0222 |
| restricted | gx        | cold     | Linear           |    0.077  |      0.037  |
| restricted | gx        | cold     | Quadratic        |    0.077  |      0.0113 |
| restricted | gx        | cold     | Quadratic + rate |    0.077  |      0.0112 |
| restricted | gx        | inside   | Boosted trees    |    0.0085 |      0.0056 |
| restricted | gx        | inside   | Cubic            |    0.0085 |      0.0055 |
| restricted | gx        | inside   | Linear           |    0.0085 |      0.0069 |
| restricted | gx        | inside   | Quadratic        |    0.0085 |      0.0055 |
| restricted | gx        | inside   | Quadratic + rate |    0.0085 |      0.0057 |
| restricted | gx        | warm     | Boosted trees    |    0.0239 |      0.0172 |
| restricted | gx        | warm     | Cubic            |    0.0239 |      0.0632 |
| restricted | gx        | warm     | Linear           |    0.0239 |      0.0339 |
| restricted | gx        | warm     | Quadratic        |    0.0239 |      0.0399 |
| restricted | gx        | warm     | Quadratic + rate |    0.0239 |      0.0387 |
| restricted | gy        | cold     | Boosted trees    |    0.1954 |      0.0727 |
| restricted | gy        | cold     | Cubic            |    0.1954 |      0.2306 |
| restricted | gy        | cold     | Linear           |    0.1954 |      0.0368 |
| restricted | gy        | cold     | Quadratic        |    0.1954 |      0.0449 |
| restricted | gy        | cold     | Quadratic + rate |    0.1954 |      0.0419 |
| restricted | gy        | inside   | Boosted trees    |    0.0342 |      0.0124 |
| restricted | gy        | inside   | Cubic            |    0.0342 |      0.0124 |
| restricted | gy        | inside   | Linear           |    0.0342 |      0.0132 |
| restricted | gy        | inside   | Quadratic        |    0.0342 |      0.0104 |
| restricted | gy        | inside   | Quadratic + rate |    0.0342 |      0.0111 |
| restricted | gy        | warm     | Boosted trees    |    0.1495 |      0.1003 |
| restricted | gy        | warm     | Cubic            |    0.1495 |      0.0882 |
| restricted | gy        | warm     | Linear           |    0.1495 |      0.0243 |
| restricted | gy        | warm     | Quadratic        |    0.1495 |      0.115  |
| restricted | gy        | warm     | Quadratic + rate |    0.1495 |      0.1114 |
| restricted | gz        | cold     | Boosted trees    |    0.1355 |      0.048  |
| restricted | gz        | cold     | Cubic            |    0.1355 |      0.2194 |
| restricted | gz        | cold     | Linear           |    0.1355 |      0.0261 |
| restricted | gz        | cold     | Quadratic        |    0.1355 |      0.0299 |
| restricted | gz        | cold     | Quadratic + rate |    0.1355 |      0.0311 |
| restricted | gz        | inside   | Boosted trees    |    0.0302 |      0.0088 |
| restricted | gz        | inside   | Cubic            |    0.0302 |      0.0108 |
| restricted | gz        | inside   | Linear           |    0.0302 |      0.0113 |
| restricted | gz        | inside   | Quadratic        |    0.0302 |      0.0105 |
| restricted | gz        | inside   | Quadratic + rate |    0.0302 |      0.0118 |
| restricted | gz        | warm     | Boosted trees    |    0.1488 |      0.1153 |
| restricted | gz        | warm     | Cubic            |    0.1488 |      0.1299 |
| restricted | gz        | warm     | Linear           |    0.1488 |      0.0483 |
| restricted | gz        | warm     | Quadratic        |    0.1488 |      0.1492 |
| restricted | gz        | warm     | Quadratic + rate |    0.1488 |      0.1418 |

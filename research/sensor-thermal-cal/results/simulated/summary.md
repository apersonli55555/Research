# SIMULATED DATA: pipeline test only, NOT a result

Calibration range for the restricted design: 15.0 to 30.0 degC. Target coverage 90 %.

## Coverage of 90 % intervals

| design     | region   | model            |   n_blocks |   coverage_split |   coverage_weighted |   weighted_refused |   worst_cell_coverage |   n_recordings |   ci_lo |   ci_hi |
|:-----------|:---------|:-----------------|-----------:|-----------------:|--------------------:|-------------------:|----------------------:|---------------:|--------:|--------:|
| full       | cold     | Boosted trees    |       6076 |            0.923 |               0.989 |              0     |                 0.643 |              8 |   0.882 |   0.968 |
| full       | cold     | Cubic            |       6076 |            0.886 |               0.935 |              0     |                 0.337 |              8 |   0.816 |   0.95  |
| full       | cold     | Linear           |       6076 |            0.722 |               0.84  |              0     |                 0.431 |              8 |   0.657 |   0.848 |
| full       | cold     | Quadratic        |       6076 |            0.883 |               0.919 |              0     |                 0.624 |              8 |   0.842 |   0.93  |
| full       | cold     | Quadratic + rate |       6076 |            0.873 |               0.914 |              0     |                 0.176 |              8 |   0.82  |   0.932 |
| full       | inside   | Boosted trees    |      37300 |            0.938 |               0.958 |              0     |                 0.712 |             12 |   0.903 |   0.967 |
| full       | inside   | Cubic            |      37300 |            0.884 |               0.962 |              0     |                 0.156 |             12 |   0.786 |   0.964 |
| full       | inside   | Linear           |      37300 |            0.939 |               0.942 |              0     |                 0.069 |             12 |   0.883 |   0.984 |
| full       | inside   | Quadratic        |      37300 |            0.892 |               0.949 |              0     |                 0.067 |             12 |   0.785 |   0.978 |
| full       | inside   | Quadratic + rate |      37300 |            0.925 |               0.971 |              0     |                 0.083 |             12 |   0.865 |   0.976 |
| full       | warm     | Boosted trees    |       5008 |            0.977 |               1     |              0.001 |                 0.911 |              4 |   0.968 |   0.987 |
| full       | warm     | Cubic            |       5008 |            0.92  |               0.975 |              0.001 |                 0.719 |              4 |   0.888 |   0.957 |
| full       | warm     | Linear           |       5008 |            0.633 |               0.943 |              0.001 |                 0.015 |              4 |   0.436 |   0.854 |
| full       | warm     | Quadratic        |       5008 |            0.85  |               0.957 |              0.001 |                 0.601 |              4 |   0.804 |   0.896 |
| full       | warm     | Quadratic + rate |       5008 |            0.9   |               0.967 |              0.001 |                 0.689 |              4 |   0.87  |   0.935 |
| restricted | cold     | Boosted trees    |       6076 |            0.187 |               0.796 |              0.835 |                 0     |              8 |   0.144 |   0.259 |
| restricted | cold     | Cubic            |       6076 |            0.131 |               0.626 |              0.835 |                 0     |              8 |   0.072 |   0.227 |
| restricted | cold     | Linear           |       6076 |            0.331 |               0.614 |              0.835 |                 0     |              8 |   0.186 |   0.435 |
| restricted | cold     | Quadratic        |       6076 |            0.438 |               0.821 |              0.835 |                 0.003 |              8 |   0.329 |   0.613 |
| restricted | cold     | Quadratic + rate |       6076 |            0.444 |               0.825 |              0.835 |                 0.003 |              8 |   0.332 |   0.624 |
| restricted | inside   | Boosted trees    |      37300 |            0.909 |               0.936 |              0     |                 0.634 |             12 |   0.877 |   0.94  |
| restricted | inside   | Cubic            |      37300 |            0.892 |               0.933 |              0     |                 0.398 |             12 |   0.84  |   0.942 |
| restricted | inside   | Linear           |      37300 |            0.878 |               0.891 |              0     |                 0.173 |             12 |   0.784 |   0.952 |
| restricted | inside   | Quadratic        |      37300 |            0.891 |               0.891 |              0     |                 0.391 |             12 |   0.821 |   0.945 |
| restricted | inside   | Quadratic + rate |      37300 |            0.925 |               0.942 |              0     |                 0.394 |             12 |   0.884 |   0.961 |
| restricted | warm     | Boosted trees    |       5008 |            0.238 |               0.867 |              0.86  |                 0     |              4 |   0.17  |   0.305 |
| restricted | warm     | Cubic            |       5008 |            0.313 |               0.926 |              0.86  |                 0     |              4 |   0.126 |   0.532 |
| restricted | warm     | Linear           |       5008 |            0.556 |               0.867 |              0.86  |                 0     |              4 |   0.419 |   0.711 |
| restricted | warm     | Quadratic        |       5008 |            0.14  |               0.52  |              0.86  |                 0     |              4 |   0.029 |   0.239 |
| restricted | warm     | Quadratic + rate |       5008 |            0.163 |               0.711 |              0.86  |                 0     |              4 |   0.039 |   0.275 |

## Error before and after compensation (median RMS over folds)

| design     | channel   | region   | model            |   raw_rms |   resid_rms |
|:-----------|:----------|:---------|:-----------------|----------:|------------:|
| full       | anorm     | cold     | Boosted trees    |    4.4325 |      0.2382 |
| full       | anorm     | cold     | Cubic            |    4.4325 |      0.2187 |
| full       | anorm     | cold     | Linear           |    4.4325 |      0.4807 |
| full       | anorm     | cold     | Quadratic        |    4.4325 |      0.2244 |
| full       | anorm     | cold     | Quadratic + rate |    4.4325 |      0.217  |
| full       | anorm     | inside   | Boosted trees    |    0.6993 |      0.2771 |
| full       | anorm     | inside   | Cubic            |    0.6993 |      0.2707 |
| full       | anorm     | inside   | Linear           |    0.6993 |      0.3306 |
| full       | anorm     | inside   | Quadratic        |    0.6993 |      0.2772 |
| full       | anorm     | inside   | Quadratic + rate |    0.6993 |      0.3098 |
| full       | anorm     | warm     | Boosted trees    |    2.6253 |      0.3319 |
| full       | anorm     | warm     | Cubic            |    2.6253 |      0.3816 |
| full       | anorm     | warm     | Linear           |    2.6253 |      0.7288 |
| full       | anorm     | warm     | Quadratic        |    2.6253 |      0.402  |
| full       | anorm     | warm     | Quadratic + rate |    2.6253 |      0.3226 |
| full       | gx        | cold     | Boosted trees    |    0.0741 |      0.0056 |
| full       | gx        | cold     | Cubic            |    0.0741 |      0.0056 |
| full       | gx        | cold     | Linear           |    0.0741 |      0.0143 |
| full       | gx        | cold     | Quadratic        |    0.0741 |      0.0071 |
| full       | gx        | cold     | Quadratic + rate |    0.0741 |      0.0074 |
| full       | gx        | inside   | Boosted trees    |    0.0077 |      0.0048 |
| full       | gx        | inside   | Cubic            |    0.0077 |      0.0052 |
| full       | gx        | inside   | Linear           |    0.0077 |      0.0089 |
| full       | gx        | inside   | Quadratic        |    0.0077 |      0.0064 |
| full       | gx        | inside   | Quadratic + rate |    0.0077 |      0.0063 |
| full       | gx        | warm     | Boosted trees    |    0.0203 |      0.0045 |
| full       | gx        | warm     | Cubic            |    0.0203 |      0.0052 |
| full       | gx        | warm     | Linear           |    0.0203 |      0.0458 |
| full       | gx        | warm     | Quadratic        |    0.0203 |      0.0085 |
| full       | gx        | warm     | Quadratic + rate |    0.0203 |      0.0086 |
| full       | gy        | cold     | Boosted trees    |    0.1896 |      0.0117 |
| full       | gy        | cold     | Cubic            |    0.1896 |      0.0141 |
| full       | gy        | cold     | Linear           |    0.1896 |      0.0196 |
| full       | gy        | cold     | Quadratic        |    0.1896 |      0.0178 |
| full       | gy        | cold     | Quadratic + rate |    0.1896 |      0.0205 |
| full       | gy        | inside   | Boosted trees    |    0.034  |      0.0142 |
| full       | gy        | inside   | Cubic            |    0.034  |      0.0124 |
| full       | gy        | inside   | Linear           |    0.034  |      0.0126 |
| full       | gy        | inside   | Quadratic        |    0.034  |      0.0123 |
| full       | gy        | inside   | Quadratic + rate |    0.034  |      0.0129 |
| full       | gy        | warm     | Boosted trees    |    0.1508 |      0.0114 |
| full       | gy        | warm     | Cubic            |    0.1508 |      0.0096 |
| full       | gy        | warm     | Linear           |    0.1508 |      0.0127 |
| full       | gy        | warm     | Quadratic        |    0.1508 |      0.0095 |
| full       | gy        | warm     | Quadratic + rate |    0.1508 |      0.008  |
| full       | gz        | cold     | Boosted trees    |    0.1409 |      0.0092 |
| full       | gz        | cold     | Cubic            |    0.1409 |      0.0097 |
| full       | gz        | cold     | Linear           |    0.1409 |      0.0224 |
| full       | gz        | cold     | Quadratic        |    0.1409 |      0.0105 |
| full       | gz        | cold     | Quadratic + rate |    0.1409 |      0.0123 |
| full       | gz        | inside   | Boosted trees    |    0.0289 |      0.0091 |
| full       | gz        | inside   | Cubic            |    0.0289 |      0.0121 |
| full       | gz        | inside   | Linear           |    0.0289 |      0.0124 |
| full       | gz        | inside   | Quadratic        |    0.0289 |      0.0124 |
| full       | gz        | inside   | Quadratic + rate |    0.0289 |      0.0146 |
| full       | gz        | warm     | Boosted trees    |    0.1511 |      0.011  |
| full       | gz        | warm     | Cubic            |    0.1511 |      0.0104 |
| full       | gz        | warm     | Linear           |    0.1511 |      0.0483 |
| full       | gz        | warm     | Quadratic        |    0.1511 |      0.0153 |
| full       | gz        | warm     | Quadratic + rate |    0.1511 |      0.0148 |
| restricted | anorm     | cold     | Boosted trees    |    4.4325 |      2.0991 |
| restricted | anorm     | cold     | Cubic            |    4.4325 |      3.5548 |
| restricted | anorm     | cold     | Linear           |    4.4325 |      1.1164 |
| restricted | anorm     | cold     | Quadratic        |    4.4325 |      0.867  |
| restricted | anorm     | cold     | Quadratic + rate |    4.4325 |      0.8283 |
| restricted | anorm     | inside   | Boosted trees    |    0.6993 |      0.2804 |
| restricted | anorm     | inside   | Cubic            |    0.6993 |      0.3023 |
| restricted | anorm     | inside   | Linear           |    0.6993 |      0.3234 |
| restricted | anorm     | inside   | Quadratic        |    0.6993 |      0.2808 |
| restricted | anorm     | inside   | Quadratic + rate |    0.6993 |      0.2842 |
| restricted | anorm     | warm     | Boosted trees    |    2.6253 |      1.649  |
| restricted | anorm     | warm     | Cubic            |    2.6253 |      3.1619 |
| restricted | anorm     | warm     | Linear           |    2.6253 |      0.357  |
| restricted | anorm     | warm     | Quadratic        |    2.6253 |      2.1059 |
| restricted | anorm     | warm     | Quadratic + rate |    2.6253 |      2.1404 |
| restricted | gx        | cold     | Boosted trees    |    0.0741 |      0.0406 |
| restricted | gx        | cold     | Cubic            |    0.0741 |      0.0749 |
| restricted | gx        | cold     | Linear           |    0.0741 |      0.0383 |
| restricted | gx        | cold     | Quadratic        |    0.0741 |      0.0111 |
| restricted | gx        | cold     | Quadratic + rate |    0.0741 |      0.0117 |
| restricted | gx        | inside   | Boosted trees    |    0.0077 |      0.0047 |
| restricted | gx        | inside   | Cubic            |    0.0077 |      0.0047 |
| restricted | gx        | inside   | Linear           |    0.0077 |      0.006  |
| restricted | gx        | inside   | Quadratic        |    0.0077 |      0.0051 |
| restricted | gx        | inside   | Quadratic + rate |    0.0077 |      0.0051 |
| restricted | gx        | warm     | Boosted trees    |    0.0203 |      0.0176 |
| restricted | gx        | warm     | Cubic            |    0.0203 |      0.0332 |
| restricted | gx        | warm     | Linear           |    0.0203 |      0.0272 |
| restricted | gx        | warm     | Quadratic        |    0.0203 |      0.0365 |
| restricted | gx        | warm     | Quadratic + rate |    0.0203 |      0.037  |
| restricted | gy        | cold     | Boosted trees    |    0.1896 |      0.0777 |
| restricted | gy        | cold     | Cubic            |    0.1896 |      0.2172 |
| restricted | gy        | cold     | Linear           |    0.1896 |      0.0257 |
| restricted | gy        | cold     | Quadratic        |    0.1896 |      0.0468 |
| restricted | gy        | cold     | Quadratic + rate |    0.1896 |      0.0497 |
| restricted | gy        | inside   | Boosted trees    |    0.034  |      0.0145 |
| restricted | gy        | inside   | Cubic            |    0.034  |      0.0118 |
| restricted | gy        | inside   | Linear           |    0.034  |      0.0122 |
| restricted | gy        | inside   | Quadratic        |    0.034  |      0.0119 |
| restricted | gy        | inside   | Quadratic + rate |    0.034  |      0.0123 |
| restricted | gy        | warm     | Boosted trees    |    0.1508 |      0.1019 |
| restricted | gy        | warm     | Cubic            |    0.1508 |      0.0498 |
| restricted | gy        | warm     | Linear           |    0.1508 |      0.0222 |
| restricted | gy        | warm     | Quadratic        |    0.1508 |      0.1085 |
| restricted | gy        | warm     | Quadratic + rate |    0.1508 |      0.1202 |
| restricted | gz        | cold     | Boosted trees    |    0.1409 |      0.053  |
| restricted | gz        | cold     | Cubic            |    0.1409 |      0.1834 |
| restricted | gz        | cold     | Linear           |    0.1409 |      0.0274 |
| restricted | gz        | cold     | Quadratic        |    0.1409 |      0.0215 |
| restricted | gz        | cold     | Quadratic + rate |    0.1409 |      0.0213 |
| restricted | gz        | inside   | Boosted trees    |    0.0289 |      0.0092 |
| restricted | gz        | inside   | Cubic            |    0.0289 |      0.0115 |
| restricted | gz        | inside   | Linear           |    0.0289 |      0.0114 |
| restricted | gz        | inside   | Quadratic        |    0.0289 |      0.011  |
| restricted | gz        | inside   | Quadratic + rate |    0.0289 |      0.0122 |
| restricted | gz        | warm     | Boosted trees    |    0.1511 |      0.1113 |
| restricted | gz        | warm     | Cubic            |    0.1511 |      0.0739 |
| restricted | gz        | warm     | Linear           |    0.1511 |      0.0434 |
| restricted | gz        | warm     | Quadratic        |    0.1511 |      0.1203 |
| restricted | gz        | warm     | Quadratic + rate |    0.1511 |      0.1198 |

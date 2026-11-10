# Why does mode 3 stiffen as it warms? A half-page note

*Written by the AI assistant from the results in `results/mode_type_slopes.csv` (run `python scripts/run_shared_threshold.py`). The data facts are verified. The explanations are hypotheses, not findings. Rewrite this in your own words before using it.*

## What the data show
The dataset authors classify the 14 tracked modes in their own plotting code (`matlab-functions/plot_trackedmodes.m`, Zenodo 3745914): arch modes 1, 2, 4, 7, 8, 14; **lateral deck modes 3 and 11**; global vertical modes 5, 6, 10, 12, 13; torsional mode 9.

Fitting a straight line to each mode's frequency against steel temperature, using only pre-retrofit hours above 5 °C so that freezing plays no part:

| Mode type | Modes | Frequency change per °C (fraction), 95 % range |
|---|---|---|
| Arch | 1, 2, 4, 7, 8, 14 | all negative: −1.2e-4 to −2.6e-4, every range below zero |
| Vertical | 5, 6, 10, 12, 13 | all negative: −1.5e-4 to −5.0e-4, every range below zero |
| Torsional | 9 | −1.6e-4 [−1.8e-4, −1.4e-4] |
| **Lateral** | **3** | **+1.2e-4 [+0.6e-4, +1.7e-4]** |
| **Lateral** | **11** | **−0.6e-4 [−2.0e-4, +0.6e-4]** (not distinguishable from zero) |

So all 12 non-lateral modes soften as the bridge warms, as steel softening predicts. The two lateral modes are the only exceptions: mode 3 clearly stiffens, and mode 11 shows no clear trend. Mode 11 was not part of the original five-mode study, so it is an independent check, and it only half-confirms the pattern: lateral modes do behave differently, but "lateral modes stiffen when warm" holds for mode 3 only.

## Why this could happen (hypotheses)
A natural frequency depends on stiffness, and for different modes the stiffness comes from different parts of the structure.

- **Vertical, arch and torsional modes** are governed mainly by the bending and axial stiffness of the steel members. Steel's elastic modulus falls slightly as temperature rises, so these frequencies fall.
- **Lateral deck modes** depend much more on how the deck is held sideways: bearings, lateral guides, and the track. Three mechanisms could make that restraint *stiffer* when warm, working against steel softening:
  1. **Thermal expansion closing gaps.** Bearings and lateral guides have small clearances. As the deck expands, clearances can close or contact pressure and friction can rise, so the support acts more rigidly sideways.
  2. **Forces from restrained expansion.** If the deck cannot expand freely, warming changes the internal forces in the arch-and-tie system. Internal force changes stiffness (the "geometric stiffness" effect, like tightening a guitar string), and it can affect sideways motion differently from vertical motion.
  3. **The track.** Continuous rails and ballast run across the bridge and resist sideways motion. Their behaviour changes with temperature in a different way from the steel structure.

If such a stiffening effect and steel softening roughly cancel, the net trend would be near zero, which is what mode 11 shows. If the stiffening dominates, the trend is positive, as in mode 3. That the lower lateral mode is the more affected one would fit a support effect, since supports matter most for the lowest, most global mode of a given type. This last step is speculation.

## How it could be tested
- **Bearing displacements.** The full dataset includes displacement measurements at the bearings. If mode 3's frequency tracks bearing position better than it tracks temperature, that points to the supports.
- **Hysteresis.** A friction mechanism would make the frequency differ between warming and cooling at the same temperature.
- **After the retrofit.** The retrofit strengthened connections of the diagonals. If mode 3's temperature trend changed afterwards, the mechanism involves those members.
- **Ask the people who know the bridge.** This is a specific, answerable question for the dataset authors.

## What not to claim
The data show an association between mode type and the sign of the temperature trend. They do not show the cause. Two lateral modes are a very small sample.

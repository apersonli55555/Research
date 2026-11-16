# Hypotheses and analysis choices, fixed before any real data

**Why this file exists.** If you choose the analysis after seeing the results, it is easy to fool yourself. Writing the plan down first, with a date, protects you and makes the paper more credible. Commit this file to Git before your first real recording. If you later change anything, add a dated note below explaining what and why. Do not edit the original text.

*Drafted by the AI assistant on 10 November 2026. Read it, change what you disagree with, then sign and date it.*

## Primary question
For a low-cost inertial sensor calibrated against temperature over 15–30 °C, do nominal 90 % split-conformal intervals keep 90 % coverage (a) inside 15–30 °C on separate recordings, and (b) at colder and warmer temperatures?

## Hypotheses
- **H1.** Inside the calibration range, on held-out recordings, coverage is within 0.85–0.95 for every model.
- **H2.** Outside the calibration range, coverage falls below 0.80 for at least the cubic polynomial and the boosted trees.
- **H3.** Weighted conformal prediction declines to give a finite interval for more than half of the out-of-range test points, and for less than 10 % of in-range points.
- **H4.** With calibration over the full temperature range, out-of-range failures largely disappear (coverage above 0.85 in all regions for at least one model).

A hypothesis that turns out false is still a result. Report it.

## Fixed analysis choices
- Targets: gyroscope x, y, z and accelerometer magnitude minus 1 g, per sensor.
- Preprocessing: first 10 minutes of each run discarded; seconds with gyroscope standard deviation above 0.5 deg/s discarded; 10-second block averages; per-run offset removed using the run's median within 20–30 °C.
- Models: linear, quadratic, cubic, quadratic + heating rate, boosted trees (settings as in `src/thermcal/models.py`).
- Intervals: split conformal, α = 0.1, calibration errors pooled by cross-validation over whole runs.
- Evaluation: leave one run out; regions cold (< 15 °C), inside, warm (> 30 °C).
- Primary metric: pooled coverage per region and model, with uncertainty from resampling whole runs.
- Minimum data before looking at results: 6 runs per sensor, 2 sensors.

## Things deliberately left for later (exploratory, to be labelled as such)
Other calibration ranges, other α, per-axis accelerometer analysis with a six-position test, more sensors or sensor models, adaptive-width intervals.

## Signed
Name: ____________________  Date: ____________

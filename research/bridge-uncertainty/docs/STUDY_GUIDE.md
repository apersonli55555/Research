# Study guide: understand and explain your own project

Read this until you can say every part out loud without looking. Judges and professors will test that.

---

## 1. The project in one sentence
I tested whether the error bars used to monitor a real bridge's vibrations stay honest when the weather changes. A model that treats freezing weather as a separate regime kept them honest when it got warmer, but no method worked when it got colder than anything the model had seen.

## 2. The 30-second version
- Bridges vibrate at natural frequencies, like a guitar string's note.
- Damage makes a bridge less stiff, so its frequencies drop. Engineers watch for that drop.
- Temperature also changes the frequencies, so engineers predict the "normal" frequency for today's temperature and draw an error band around it. A reading outside the band raises an alarm.
- I checked whether those bands really contain 90 % of normal readings, as they claim, when the season changes.
- Most models broke on warm days they had never seen. My physics-guided model didn't.
- On cold days nobody's model worked, including mine.

## 3. The key ideas, explained simply

**Natural frequency.** f = (1/2π)·√(k/m). Stiffer (bigger k) means higher frequency. Heavier (bigger m) means lower.

**Why temperature matters.** Warm steel is very slightly softer, so the frequency drops a tiny bit. Near freezing the bridge suddenly gets stiffer, maybe because the gravel under the tracks or the bearings freeze. (That cause is a guess; the data don't prove it.)

**Prediction interval (error band).** "I predict 2.92 Hz, and I'm 90 % sure the real value is between 2.91 and 2.93."

**Coverage.** The fraction of real readings that actually land inside the band. A 90 % band should have about 90 % coverage. Below that, the band is overconfident and causes false alarms.

**Conformal prediction.**
1. Train a model.
2. Check how wrong it is on some separate "calibration" data.
3. Make the band just wide enough to cover 90 % of those errors.

It *guarantees* 90 % coverage only if the future looks like the calibration data ("exchangeable"). Spring doesn't look like winter, which is why I tested it.

**Extrapolation.** Predicting outside the range you trained on, for example training on temperatures up to 14 °C and predicting at 25 °C. This is where most models fail.

**My model ("Physics + GBM").**
- Step 1: a 4-number physics formula: frequency ∝ √(1 + a·ΔT), plus an extra-stiffness term below about 1 °C.
- Step 2: a small machine-learning model (gradient-boosted trees) fixes whatever the formula misses.

The formula carries the trend into new temperatures; the machine learning handles the details.

**Why the others fail when it's warmer.**
- *Trees:* predict a flat line outside their training range.
- *Standard GP:* drifts back toward the average of its training data.
- *Polynomials:* curve off wildly.
- *A plain straight line:* gets its slope wrong, because the freezing jump in January drags it too steep.

**The ablation (my "why").** I removed the freezing term and coverage dropped. The freezing term isn't there to predict the cold. It's there to stop the cold-weather jump from ruining the warm-weather slope.

**The control that keeps me honest.** √(1 + a·ΔT) is nearly a straight line when *a* is tiny. So I also tested a physics-free straight line with the same kink. It did exactly as well as my physics model (0.896 vs 0.895). So the win comes from the **two-regime shape**, not from the physics formula. The physics is why I looked for that shape, and it makes the fitted numbers meaningful, but I must not claim that "physics beats data-driven models".

**Paired test with day-block bootstrap.**
- I compare two models on exactly the same hours.
- I resample whole *days*, not single hours, because hours within a day are nearly copies of each other.
- A Holm correction is applied because I compared against 8 models; more comparisons means more chances of a fluke.

**Rolling windows.** Instead of choosing train/test months myself, I slid a window across the data one week at a time: 22 windows in total. The result held in all of them, so it isn't luck in how I chose the split.

## 4. My numbers (memorize these five)
1. Random split: every model ≈ 0.90 coverage. Conformal prediction works when its assumption holds.
2. Warmer than training: my model's median is **0.91**; physics-free trees 0.88; GP 0.79; cubic 0.32.
3. Paired tests: I beat all 8 alternatives on the warm side. For example, +0.08 [0.05, 0.12] over boosted trees; adjusted p ≤ 0.002.
4. Colder than training: every model ≈ 0.50–0.62. Nobody wins.
5. Retrofit detection: the 2019 repair raised mode 6 by about +0.06 Hz (+2 %). It was caught on 100 % of later days, with 0–1.3 % false-alarm days before the repair.

## 5. Honest limitations (say these before a judge does)
- **One bridge, one season of data.** I've asked (or will ask) for a second structure's data (LUMO).
- **The cold regime is unsolved.**
- **The retrofit was a strengthening, not damage.** I say "state change".
- **The frequencies were identified by the dataset authors.** I didn't process raw vibrations.
- **I used an AI assistant for code scaffolding and drafts.** I verified and extended the work myself. List exactly what you did yourself.

## 6. Two-minute pitch script (practise out loud)
> "Bridges are monitored by tracking their vibration frequencies, because damage lowers them. But temperature changes them too, so engineers draw an error band around a temperature-based prediction and alarm when a reading falls outside. I asked: do those bands stay honest when the season changes?
> Using two years of public data from a railway bridge in Belgium, I compared nine models. When the test data look like the training data, all of them work. But on days warmer than anything they trained on, most models' bands broke: some contained as little as a third of the readings instead of 90 %.
> My model adds a four-parameter physics formula for how steel stiffness changes with temperature, including extra stiffening near freezing. It kept about 90 % coverage on those warm days, and it beat all eight alternatives in paired statistical tests over 22 test windows. An ablation showed why: the freezing term stops cold-weather stiffening from distorting the warm-weather trend.
> I also found a limit: when tested on colder days than it trained on, every model failed, mine included. Physics helps you extrapolate along a pattern you've seen, not into one you haven't. Next, I'm testing a second structure and a way to carry freezing behaviour from one winter to the next."

## 7. Self-quiz (cover the answers)
1. What does conformal prediction guarantee, and when? → *At least 90 % coverage, if future data are exchangeable with the calibration data.*
2. Why can't trees extrapolate? → *They predict a constant beyond their training range.*
3. Why did "linear + trees" do worse than trees alone? → *The straight line's slope was distorted by the freezing jump.*
4. What is coverage? → *The fraction of real readings inside the band.*
5. Why resample days, not hours? → *Hours in a day are strongly correlated; treating them as independent makes the CIs falsely narrow.*
6. Why not use adaptive conformal inference everywhere? → *It adapts to whatever happens, including real damage, which would hide the alarm.*
7. What does a positive *a* for mode 3 mean? → *The frequency rises with temperature for that mode, which steel softening alone can't explain. Honest answer: unknown; possibly bearings or the track.*
8. What's the biggest weakness? → *One bridge, and the cold regime is unsolved.*
9. What's novel? → *Testing conformal coverage under seasonal shift on KW51, inside vs outside the training range, with paired tests. (Confirm with your own literature search before claiming "first".)*
10. Did you tune your model until it won? → *No. Settings were fixed in advance. I re-ran 9 settings for my model and gave the competitor the same 9. Every version of mine (0.88–0.90) beat every version of theirs (0.77–0.83), and my default wasn't even the best one.*
11. Is your model just drawing wider bands? → *No. It has the narrowest bands of all nine models and still the best coverage.*
12. Is it really the physics that helps? → *No, it's the shape. A physics-free line with a kink at the freezing point matches my model. The physics motivated the shape and makes the parameters interpretable.*
13. What did you do yourself? → *Answer this truthfully and specifically.*

## 8. Glossary
| Term | Meaning |
|---|---|
| Mode | One of the bridge's natural vibration patterns, each with its own frequency |
| Hz | Vibrations per second |
| Calibration set | Data held back to size the error band |
| Out-of-range | Test temperatures outside the training temperatures |
| Ablation | Removing one part of a model to see if it mattered |
| Interval score | A score that rewards bands that are narrow *and* contain the truth |
| GBM | Gradient-boosted machine: many small decision trees added together |
| GP | Gaussian process: a flexible model that also estimates its own uncertainty |
| p-value | How surprising a result would be if there were really no difference |
| 95 % CI | The range the true value plausibly lies in |

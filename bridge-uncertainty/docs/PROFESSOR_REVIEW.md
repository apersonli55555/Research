# A professor-style review of this project

**What this is.** A critical review written by the AI assistant, playing the role of a faculty reviewer in a first meeting. It is **not** from Professors Bui-Thanh, Fuhg, Bennighof or Sirohi, and they might disagree with it. Treat it as practice for the real conversation, and check anything marked *unverified* yourself.

Two small checks were run on the data for this review (10 Nov 2026). Their numbers are quoted below.

---

## Follow-up results (10 Nov 2026): two of this review's suggestions, tested
Both were run in `scripts/run_shared_threshold.py`. One of this review's predictions turned out wrong.

1. **Shared freezing threshold: it did not help.** Fitting all five modes with one common threshold gives 1.22 °C with a 95 % range of −0.40 to 2.25 (width 2.65 °C). The separate fits have widths of 2.36–3.01 °C. This review predicted the uncertainty would "shrink a lot". It didn't, and the reason is instructive: all five modes are measured on the *same few freezing days*, so they are five views of the same evidence, not five independent pieces of evidence. More modes cannot substitute for more cold days.
2. **Mode types confirmed from the dataset authors' own code.** Modes 3 and 11 are lateral deck modes; 5, 6, 10, 12 and 13 are global vertical; **mode 9 is torsional**, not vertical as the table below loosely suggests; the rest are arch modes. All 12 non-lateral modes soften as they warm. Mode 3 stiffens, and mode 11, the other lateral mode, shows no clear trend. See `docs/mode3_lateral_note.md`.
3. **Retrofit dates confirmed** as 15 May to 27 September 2019 in the same code.

4. **Prior work found (important).** The dataset authors' 2022 paper already reports the two-regime ("bilinear") temperature relation with a knee near 0 °C, attributes the cold stiffening to frozen ballast, and excludes temperatures below 2 °C because they bias the regression slope. A professor who knows this literature would say so in the first minute. Your contribution is the coverage analysis, not the regime structure. See README, section v7.

---

## Overall verdict
The question is good, and the evaluation is more careful than most student work: rolling windows, paired tests, an ablation, and a control that overturned your own first claim. The weak points are that the modelling contribution is small (a kinked line), there is one bridge and one season, and the hardest case (freezing) is unsolved. As it stands this is a solid science-fair project and a reasonable NHSJS paper. It is not yet a professional journal paper.

The strongest thing you have is not the model. It is the **evaluation idea**: average coverage hides failures, so report coverage separately for conditions inside and outside the training range.

---

## The four questions you emailed, answered

### 1. "Is inside-vs-outside-range coverage a sound way to study this shift?" (asked of Prof. Bui-Thanh)
Yes, and there is established theory to anchor it to. Read these before you talk to anyone in uncertainty quantification.

- **This is "covariate shift".** The inputs (temperature) are distributed differently at test time, while the relation between input and output is assumed unchanged. Tibshirani, Barber, Candès & Ramdas (2019), *Conformal Prediction Under Covariate Shift* (arXiv:1904.06019), show how to restore the guarantee by re-weighting calibration points by how likely each is under the test distribution.
- **Why "outside the range" is special.** That re-weighting needs test conditions to have *some* presence in the calibration data. A temperature never seen has none, so no weighting can fix it. Your inside/outside split is exactly the line between "fixable by weighting" and "pure extrapolation". Say this in the paper; it turns an ad-hoc split into a principled one.
- **A known impossibility result supports you.** Barber, Candès, Ramdas & Tibshirani (2021), *The limits of distribution-free conditional predictive inference* (arXiv:1903.04684): no method can guarantee coverage at every individual input value without assumptions. So checking coverage by temperature bin is the right diagnostic, and failures there are expected, not a bug.
- **For time-ordered data,** see Barber et al. (2023), *Conformal prediction beyond exchangeability* (arXiv:2202.13415), which weights recent calibration points more.

**Criticisms a statistician would raise.**
1. **Missing baseline:** you never tried weighted conformal prediction (the first paper above). It should help inside the range and fail outside it. Showing that would be a clean result.
2. **Constant-width bands:** your band has the same width at every temperature. In the check run for this review, the typical error of your model on mode 6 was 3.2 mHz at 5–10 °C but 4.7 mHz above 20 °C, and larger by day (4.1) than by night (3.2). Mode 13 showed no such pattern. So for some modes the band should widen in warm, sunny hours. The standard fix is a band whose width adapts, e.g. Romano, Patterson & Candès (2019), *Conformalized Quantile Regression* (arXiv:1905.03222).
3. **"Outside" uses one variable only.** A test hour can have a familiar temperature but an unfamiliar *combination* (e.g. warm steel and cold air in spring sunshine). You only test the range of steel temperature.
4. **Few independent days.** The warm-side result rests on 28 distinct days from one spring. Your confidence intervals account for this, but it is still one season.

### 2. "How do you build in physics the data never exercise?" (asked of Prof. Fuhg)
- **Share information across modes.** Your own fits put the freezing threshold at 1.1–1.4 °C in all five modes, with wide, overlapping uncertainty (roughly −1 to +2.4 °C each). If freezing is one physical event, there is **one threshold for the whole bridge**. Fit all modes together with a single shared threshold. That should shrink its uncertainty a lot, and it is a small, well-defined piece of work you could do yourself.
- **Carry parameters across time as a prior.** Fit the threshold and size of the cold effect on one winter, then hold them fixed (or allow small changes) when training on warm data. This is your "Task 5", and it is the right idea. A Bayesian version puts a probability distribution on those parameters instead of fixing them.
- **Constrain the shape by construction.** Your control showed the exact formula doesn't matter, but the *shape* does: decreasing with temperature when warm, extra stiffness when cold. Models can be built so that they cannot violate such a shape (monotone or convex networks). That is the spirit of Prof. Fuhg's polyconvexity work, and it is the honest way to connect your project to his.
- **Be careful with the word "physics".** After your kink control, the defensible claim is "a physically motivated two-regime structure", not "a physics model". Say it that way first, before someone else does.

### 3. "Why would mode 3's frequency rise with temperature?" (asked of Prof. Bennighof)
The check run for this review found something useful. Using the sensor names in the dataset (those ending in `y` versus `z`, on the bridge deck):

| Mode | Frequency | Relative motion, `z` sensors | Relative motion, `y` sensors |
|---|---|---|---|
| **3** | 1.89 Hz | 0.17 | **0.88** |
| 5 | 2.58 Hz | 0.79 | 0.08 |
| 6 | 2.92 Hz | 0.50 | 0.02 |
| 9 | 4.11 Hz | 0.92 | 0.28 |
| 13 | 6.32 Hz | 0.70 | 0.13 |

**Mode 3 moves in a different direction from the other four.** If `z` is vertical and `y` is sideways (the usual convention; *confirm it in Maes & Lombaert 2021*), then mode 3 is a sideways (lateral) mode and the others are mainly vertical.

That makes the odd sign much less mysterious. Vertical bending stiffness comes mostly from the steel itself, which softens when warm. Sideways stiffness depends far more on how the bridge is held: its bearings and lateral restraints, the track, and how the arches are braced. Hypotheses to put to a structural engineer:
- **Supports that tighten when warm.** Thermal expansion can close gaps or increase friction at bearings and joints, stiffening the bridge sideways.
- **Forces from blocked expansion.** If expansion is partly restrained, warming changes the internal forces in the arch-and-tie system, and those forces change stiffness (a "geometric stiffness" effect).
- **The track.** Continuous rails and ballast respond to temperature differently from the steel structure.

None of these is established by your data. But "the one mode with the opposite sign is the one lateral mode" is a real observation, and it is a far better question to bring to a professor than "why is mode 3 weird?".

**A sanity check you should finish (your Task 2).** For the vertical modes you fitted a frequency change of about −1.3e-4 per °C (modes 6 and 9). Since frequency goes as the square root of stiffness, that implies stiffness falling by about 2.6e-4 per °C. *From memory, and unverified,* published values for how steel's elastic modulus falls with temperature near room temperature are of a similar size, a few parts in 10,000 per °C. If a citable source confirms that, it is strong evidence that your fitted slope really is steel softening. Find the source before you claim it.

### 4. "Does this matter for aerospace structures?" (asked of Prof. Sirohi)
In general terms, yes: aircraft and spacecraft see far larger temperature swings than a bridge, and heating changes stiffness and therefore vibration behaviour. But this was your weakest email. Your data is a bridge, and you had no aerospace result to show. Don't expect a reply here, and don't claim aerospace relevance in the paper beyond one sentence of motivation.

---

## The questions a professor would ask you in the first ten minutes
1. *"What exactly did you do, and what did the AI do?"* Have a precise, honest answer. Right now most of the analysis was done by the assistant. Your answer gets better with every task you complete yourself.
2. *"Your model is a line with a kink. What's the contribution?"* Answer: the evaluation, not the model. Coverage inside versus outside the training range, tested with paired statistics, plus the finding that regime separation is what matters.
3. *"Why should I believe one spring on one bridge?"* Answer: you shouldn't yet. That is why Z24 is next.
4. *"What would you do with three more months?"* Have a list of three things, in order.

## What to do next, in a professor's priority order
1. **Shared freezing threshold across modes** (question 2). Small, original, and yours to do.
2. **Weighted conformal prediction as a baseline** (question 1). It connects your work to the standard theory.
3. **Confirm mode 3 is lateral** from the dataset paper, and write half a page on the hypotheses.
4. **Finish Task 2** (the steel modulus value).
5. **Z24** as the second structure. It freezes hard, so it is the right test of your unsolved case.

## If the professors don't reply
That is normal for cold emails from high-school students; it says nothing about the project. More promising routes:
- **The people who work on this exact dataset.** The KW51 dataset authors (Maes and Lombaert, KU Leuven) and the group that has published several KW51 papers (Wittenberg, Neumann, Gertheiss and co-authors) know this bridge. A short, specific question, such as the mode 3 observation above, is much more likely to get an answer from them than a general request for feedback.
- **Graduate students and postdocs** in the UT groups, who are often more reachable than the professor.
- **Structured programs** that assign you a mentor, such as the summer research programs already on your list.
- **Your science fair.** Regional fairs often have a mentor-matching scheme or can point you to one; ask your teacher or the GARSEF organisers.
- **One polite follow-up** after about ten days is fine. The drafts are already in your Gmail.

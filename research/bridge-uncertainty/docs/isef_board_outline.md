# ISEF-format project board outline

**Before printing anything, check the current-year rules.** Get the ISEF Display & Safety rules and the category list from Society for Science, plus the GARSEF and Texas state rules. They change; I'm not quoting them from memory as fact. In particular, check:
- the board size limits;
- the image-credit rules (every figure from someone else's data or paper needs a credit line);
- what may and may not be displayed;
- whether a computational project on public data needs any forms beyond the standard ones (usually just the student checklist, research plan, and adult sponsor checklist);
- **the current policy on disclosing AI assistance**.

**Likely category:** Engineering Technology: Statics & Dynamics, or a computational/software category. Confirm the exact current name and pick by where the *question* lives. Here the question is structural and measurement-focused.

---

## Layout (three-panel tri-fold, read left → center → right)

```
┌──────────────┬────────────────────────────────┬──────────────┐
│ LEFT         │ CENTER                         │ RIGHT        │
│              │  TITLE (one line, big)         │              │
│ 1 Problem    │  Question in one sentence      │ 6 Results    │
│ 2 Background │                                │   (Table 1   │
│   (bridge    │  KEY FIGURE (large)            │    + Fig 3)  │
│   photo CR,  │  coverage vs temperature       │ 7 Detection  │
│   f ∝ √k)    │                                │   (Fig 5)    │
│ 3 Goal &     │  4 Method flow diagram         │ 8 Conclusions│
│   hypothesis │  data → models → conformal →   │ 9 Limitations│
│              │  metrics                       │   & future   │
│              │  5 Physics model (Fig 2)       │ 10 Refs, AI  │
│              │                                │   disclosure │
└──────────────┴────────────────────────────────┴──────────────┘
```

## Panel contents (keep each block short enough to read in 30 s)

**Title (≤ 12 words).** "Do Bridge-Monitoring Error Bars Survive an Unseen Season?" Subtitle: "Physics-guided, conformally calibrated temperature compensation on a real railway bridge."

**1. Problem (3 bullets).**
- Bridges are monitored by tracking their vibration frequencies; damage lowers them.
- Temperature changes them too, so the alarm threshold has to come from an uncertainty band.
- Do those bands stay honest when the season changes?

**2. Background.**
- f = (1/2π)√(k/m), and steel stiffness changes with temperature.
- Conformal prediction in one sentence and one picture: calibration errors → band.
- The exchangeability assumption, in plain words: "the future looks like a shuffled version of the past."
- The KW51 bridge: photo with a credit line, location, monitored 2018–2020, retrofitted in 2019. Cite Maes & Lombaert.

**3. Engineering goal and hypothesis.**
- **Goal:** prediction intervals for temperature-compensated frequencies that keep 90 % coverage in an unseen season, with few false alarms.
- **Hypothesis (state it before the results):** models that cannot extrapolate a temperature trend will lose coverage at unseen temperatures, and a physics prior will reduce that loss.
- **[TODO: write the hypothesis in your notebook with a date. Judges ask whether you stated it before you saw the results. Be honest: here, the hypothesis came after the first run. Saying so is fine.]**

**4. Methods (a flow diagram, not paragraphs).**
1. Public data (Zenodo, MD5-checked).
2. Six models.
3. Same split-conformal wrapper.
4. Random split vs chronological split (train Nov–Jan / calibrate Feb / test Mar–May).
5. Metrics: coverage, width, interval score, coverage by temperature bin with day-block bootstrap CIs.
6. Detection: frozen bands with a daily alarm rule.

**5. Physics model.** Show the equation with each term labelled in words, and Fig. 2 (frequency vs temperature with the fitted curve and a "highest training temperature" line).

**6. Results.**
- Table 1 (coverage, chronological vs random).
- Fig. 3 (calibration curves).
- **One sentence in large type:** "Average coverage hid failures: above 20 °C, the GP covered 1–7 % on three modes; the physics-guided model covered 87–96 % on all five."

**7. Detection.** Fig. 5, plus the false-alarm numbers. Say plainly that the retrofit was a large stiffening, so every method saw it. **[TODO: add your minimum-detectable-shift curve here once done. It is the strongest addition you can make.]**

**8. Conclusions (3 bullets).**
- Conformal prediction is reliable when its assumption holds and can fail badly in an unseen season.
- Failures happen where models extrapolate badly, so always report coverage by condition.
- A four-parameter physics prior fixed most of it, and the fitted parameters are physically interpretable.

**9. Limitations and future work.**
- Limitations: 8 hot days; one bridge; no sub-zero test data; the retrofit is not damage; one split.
- Future work: rolling-origin evaluation, a second structure (LUMO), a PINN residual, minimum detectable shift.

**10. References and disclosure.** 6–8 key references, a data credit, and a code link with QR code. Describe the AI-assistance and mentor help accurately; see the "AI-use disclosure" section in the manuscript.

## Lab notebook and judging prep
Judges probe whether *you* understand and did the work. Be ready to answer these without notes:
1. What does conformal prediction guarantee, and under what assumption? Why does bridge data break it?
2. Why is "random split" coverage misleading for time series? (Neighbouring hours are near-duplicates, so information leaks.)
3. Derive f ∝ √(1 + aΔT) from f ∝ √(k/m). What is *a* physically? How did your fitted *a* compare with handbook steel values? (Task 2.)
4. Why does mode 3 give a positive *a*? (An honest answer: "I don't know; here are two hypotheses.")
5. Why does a GP fail above the training temperature? (An RBF kernel reverts to the training mean.) Why do trees fail? (They predict a constant outside the training range.)
6. Why use a day-block bootstrap instead of an ordinary bootstrap?
7. Why not just use adaptive conformal inference? (It would adapt away real damage.)
8. What would change your conclusion? (Rolling-origin results, a second structure.)
9. What exactly did you do versus the AI assistant or a mentor? Keep a dated log.

## Timeline to GARSEF (adjust to the actual fair date)
- **Week 1:** tasks 1–3 (verify dates, find dE/dT, literature check); rewrite the README in your own words.
- **Weeks 2–3:** minimum-detectable-shift study (task 4) and rolling-origin evaluation (task 5).
- **Week 4:** ablations (task 7); optional PINN residual (task 6).
- **Week 5:** update the manuscript and figures; email professors with the one-pager.
- **Week 6:** print the board, practise a 2-minute and a 10-minute pitch, and do mock judging.

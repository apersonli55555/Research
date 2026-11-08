# What to do next, in order

Tick each box. Keep a dated lab notebook (paper or a Google Doc) and write every step in your own words. That notebook is your proof that the work is yours.

## Week 1: own the basics (about 5–6 hours)
- [ ] **Read `docs/STUDY_GUIDE.md` twice.** Then do the self-quiz with the answers covered.
- [ ] **Run the project on your laptop.** Download it (see the README), install it, and run `python -m pytest -q`. If something breaks, write down the error and how you fixed it.
- [ ] **Read** Angelopoulos & Bates (arXiv 2107.07511), sections 1–2. Write half a page: what does conformal prediction guarantee, and when does it break?
- [ ] **Task 1:** find the retrofit dates in Maes & Lombaert (2021) or Maes et al. (2022). If they aren't 15 May–27 Sep 2019, fix `RETROFIT_START`/`RETROFIT_END` in `src/kw51uq/data.py` and tell me.
- [ ] **Send the LUMO email** (`docs/lumo_data_request_email.md`) from your own email address.
- [ ] **Find your fair's dates** (GARSEF registration and deadlines) and write them at the top of your notebook.

## Week 2: your own experiment (about 6–8 hours)
- [ ] **Open `scripts/student_min_detectable_shift.py`** and complete TODO 1 and TODO 2. They're short. Run it.
- [ ] Complete TODO 3 (more models and modes) and TODO 4 (the figure).
- [ ] Complete TODO 5: write your conclusions in your notebook.
- [ ] **Bring me your results.** I'll review them like a mentor would, check them for bugs, and help you fit them into the paper. I won't redo them.

## Week 3: physics and literature (about 5 hours)
- [ ] **Task 2:** find a citable value for how steel's Young's modulus changes with temperature near 0–30 °C (try the NIST technical note on structural steel, or Eurocode 3 tables). Compare it with your fitted *a* values in `results/physics_parameters.csv`. Do they agree in sign and size?
- [ ] **Task 3:** search Google Scholar for "conformal prediction structural health monitoring", "conformal natural frequency" and "KW51". Read the abstracts of anything close. Write one honest sentence on what's new in your project.
- [ ] Read Jones, Pitchforth & Cross (EWSHM 2024) on physics-informed GP kernels.

## Week 4: writing (about 6 hours)
- [ ] Rewrite the manuscript abstract and introduction (`docs/manuscript_draft.md`) **in your own words**. Then resolve every **[TODO]**.
- [ ] Write the AI-use disclosure honestly: what the assistant did and what you did.
- [ ] Fill in your name, school and links in `docs/one_page_summary.html`, then regenerate the PDF (or ask me to).

## Week 5: professors and board
- [ ] Create the `bridge-uncertainty` GitHub repo and tell me. I'll move the project there so you have a clean link.
- [ ] **Email 2–4 UT Austin professors.** Pick people whose recent papers are on structural health monitoring, uncertainty quantification or physics-informed ML; read one paper of theirs first. Attach the one-pager. Keep it to 5–6 sentences, mention their paper specifically, and ask one concrete question.
- [ ] Build the board from `docs/isef_board_outline.md`. Practise the 2-minute pitch from the study guide 10 times out loud, then do a mock judging session with a teacher.

## Optional stretch goals (only after everything above)
- **Task 5, cross-season cold prior:** fit the freezing parameters on one winter and reuse them on warm-trained models. Does that fix the cold-side failure?
- **PINN residual model:** replace the trees with a small neural network trained with a physics penalty. This builds on your earlier paper.
- **LUMO second structure:** if they send the data.

## When to come back to me
- When a TODO is done and you want a review.
- When something breaks.
- When you get a reply from LUMO or a professor.
- When you want the one-pager or manuscript regenerated with your new results.

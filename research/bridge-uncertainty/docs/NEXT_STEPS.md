# What to do next, in order

Tick each box. Keep a dated lab notebook (paper or a Google Doc) and write every step in your own words. That notebook is your proof that the work is yours.

## New: professor-style review (read this first)
`docs/PROFESSOR_REVIEW.md` answers the four questions you emailed, lists what a professor would criticise, and ranks what to do next. Two new tasks come out of it, both small and both yours to do:
- [ ] **Shared freezing threshold:** fit all five modes together with one common T_c. Does its uncertainty shrink?
- [ ] **Mode 3:** confirm from Maes & Lombaert (2021) that the `y` sensors are lateral, then write half a page on why a lateral mode could stiffen as it warms.

## Week 1: own the basics (about 5–6 hours)
- [ ] **Read `docs/STUDY_GUIDE.md` twice.** Then do the self-quiz with the answers covered.
- [ ] **Run the project on your laptop.** Download it (see the README), install it, and run `python -m pytest -q`. If something breaks, write down the error and how you fixed it.
- [ ] **Read** Angelopoulos & Bates (arXiv 2107.07511), sections 1–2. Write half a page: what does conformal prediction guarantee, and when does it break?
- [ ] **Task 1:** find the retrofit dates in Maes & Lombaert (2021) or Maes et al. (2022). If they aren't 15 May–27 Sep 2019, fix `RETROFIT_START`/`RETROFIT_END` in `src/kw51uq/data.py` and tell me.
- [ ] **Register for the Z24 data** at bwk.kuleuven.be/bwm/z24 with your school email (see "Get a second structure yourself" below).
- [ ] **Send the LUMO email** (`docs/lumo_data_request_email.md`) from your own email address. This is a backup second structure.
- [ ] **Find your fair's dates** (GARSEF registration and deadlines) and write them at the top of your notebook.

## Week 1½: the learning lab (about 5–6 hours, spread over the week)
- [ ] Work through `learning/oma_lab.ipynb` (5 short labs; see `learning/README.md`). It teaches you how natural frequencies are extracted from raw vibrations, which you'll need for Z24, and lets you reproduce your core finding on a simulated bridge.
- [ ] Bring me your answers to questions Q6–Q8.

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


## The goal: RSI (and why it changes the priorities)
RSI looks for evidence that **you** can do research: the project, your own decisions, and recommendation letters from people who watched you work. **Check the official requirements and deadline at cee.org/research-science-institute.** Third-party sites disagree, so don't trust them. The items below matter more for RSI than any further polishing of the KW51 analysis.

### 1. Get a second structure yourself: the Z24 bridge (highest value)
- **What it is:** the most-cited benchmark in vibration-based monitoring (Switzerland, 1997–98).
  - It has a year of hourly vibration recordings and detailed temperatures (deck, pavement, soil).
  - It has the famous **freezing-asphalt stiffening below 0 °C**, which is exactly where your cold-side result says every model fails.
  - It was also progressively damaged before demolition. That is *real damage*, unlike KW51's retrofit.
- **How to get it:**
  - Fill in the registration form at **bwk.kuleuven.be/bwm/z24** ("Registration") using your **school email address**. In the Comments box, say in 2–3 sentences what you'll use it for.
  - KU Leuven reviews the request and, if accepted, emails you a temporary account within a few working days.
- **Terms (follow them exactly):**
  - non-commercial research only;
  - acknowledge the KU Leuven Structural Mechanics Section;
  - cite their Z24 papers;
  - the data **may not be passed to anyone else**. So the Z24 files must **never** go into the public GitHub repo. Only your code and results go there. Add the data folder to `.gitignore`.
- **What you'll probably need to learn:** the Z24 download may contain raw accelerations rather than ready-made frequencies. If so, you'd identify the natural frequencies yourself (operational modal analysis, e.g. stochastic subspace identification). That's harder, but it's precisely the kind of real measurement-science skill RSI and a mentor will notice. A professor's help here would be ideal.
- **Not yet verified:** the exact file contents behind the login. Tell me what you receive and we'll plan from there.

### 2. Do your own experiments
The minimum-detectable-shift study (Week 2) and the cold-prior idea (Task 5) are *your* contributions. Z24 gives the cold-prior idea a real test.

### 3. Turn a professor contact into a mentor
- Email when you have **your own** result: your MDS figure, or Z24 access.
- In the email, ask one specific technical question, for example about the cold-stiffening mechanism or how to identify Z24's modes.
- A mentor who has *worked with you for months* writes the letter RSI values. One who only read your PDF doesn't.

### 4. Keep the evidence trail
- Keep a dated lab notebook.
- Make your own Git commits from your own GitHub account.
- Write an honest AI-use disclosure.

If anyone asks "what did you do?", you should be able to point to dates, code and notes.

## When to come back to me
- When a TODO is done and you want a review.
- When something breaks.
- When you get a reply from LUMO or a professor.
- When you want the one-pager or manuscript regenerated with your new results.

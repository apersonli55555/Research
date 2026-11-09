# Learning lab: from raw vibrations to honest error bars

`oma_lab.ipynb` is a hands-on lab (about 5–6 hours in total) that teaches the skills you need for the Z24 bridge and for defending your KW51 project:

1. **Natural frequencies** from mass and stiffness (eigenvalue problem)
2. **Spectrum and peak picking** on noisy ambient vibration
3. **Stochastic subspace identification (SSI)** and stabilization diagrams: how the KW51 frequencies were made
4. **Temperature and freezing:** fit the physics law, then see what happens when training never sees freezing
5. **Conformal prediction:** random vs seasonal split, physics vs polynomial

Everything runs on a simulated 3-mass "bridge" with known true answers, and each lab has a `check_labN()` that prints PASS or a hint.

**Run it:** `pip install jupyter`, then `jupyter notebook learning/oma_lab.ipynb` from the project folder (or open it in VS Code or Google Colab, uploading `labkit.py` alongside it).

**Rules for yourself:**
- Write the `# YOUR CODE` cells yourself.
- Answer the 💬 questions in your lab notebook in your own words.
- Don't read the `check_*` functions in `labkit.py` until you've passed or given up on an exercise.

**`oma_lab_worked.ipynb`** is the fully worked version, with every solution explained line by line, answer sketches for the 💬 questions, and all figures. Try each exercise first, use the worked version when stuck, then redo the lab from memory. Being able to rebuild it without looking is what a judge or mentor will test.

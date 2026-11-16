#!/usr/bin/env python3
"""Build the conference-template version (cvpr/main.tex) from paper.tex.

paper.tex stays the single source of the text and numbers; this script only swaps the preamble for
the official style file, adds the LLM Usage Statement required by CVPR 2027, and spaces the
bibliography entries (needed by the style's back-references). For ICCV, replace cvpr.sty and
\\confName when that template is published.

    python vision/paper/make_cvpr.py && cd vision/paper/cvpr && pdflatex main && pdflatex main
"""
import os, re, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "paper.tex")).read()

title = re.search(r"\\title\{\\Large\\bf (.*?)\}\n", src, re.S).group(1)
body = src[src.index(r"\begin{abstract}"):src.index(r"\begin{thebibliography}")]
bib = src[src.index(r"\begin{thebibliography}"):src.index(r"\end{document}")]
bib = re.sub(r"\n(\\bibitem)", r"\n\n\1", bib)
bib = bib.replace("\\end{thebibliography}", "\n\\end{thebibliography}")
body = body.rstrip()
if body.endswith("{\\small"):
    body = body[:-len("{\\small")].rstrip()
    bib = "{\\small\n" + bib

PREAMBLE = r"""% Conference-template version. Generated from ../paper.tex (same text and numbers) by ../make_cvpr.py. Do not edit by hand.
\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage[review]{cvpr}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{adjustbox}
\usepackage{amsmath}
\usepackage{multirow}
\usepackage{rotating}
\usepackage{xcolor}
\definecolor{cvprblue}{rgb}{0.21,0.49,0.74}
\usepackage[pagebackref,breaklinks,colorlinks,allcolors=cvprblue]{hyperref}
\setlength{\tabcolsep}{4pt}
\def\paperID{*****} % TODO: replace with the OpenReview paper ID
\def\confName{CVPR}
\def\confYear{2027}

\title{TITLE}
\author{Anonymous submission}

\begin{document}
\maketitle
"""

LLM = r"""

\section*{LLM Usage Statement}
A large language model assistant (Claude, Anthropic) was used extensively in this work. It wrote the data-processing, feature-extraction, fine-tuning and evaluation code; proposed the experimental design, including the model-independent test sample, the additional test sets, the zero-label adaptation experiment with its controls, the tests specified in advance and the leakage check; produced the figures; searched for and summarised related work; and drafted the text of this paper. A second assistant (ChatGPT, OpenAI) was used to check bibliographic details. The human authors chose the research topic, ran every experiment, labelled the data (no label was produced by a model), recruited the second annotator, decided which experiments to run, and reviewed the results and the manuscript. All numbers in the paper are copied from logged experiment outputs, and all references were checked against their sources. The authors take full responsibility for the content of the paper.

"""

out = PREAMBLE.replace("TITLE", title.replace("\\\\", " ")) + body + LLM + bib + "\\end{document}\n"
os.makedirs(os.path.join(HERE, "cvpr"), exist_ok=True)
open(os.path.join(HERE, "cvpr", "main.tex"), "w").write(out)
for f in os.listdir(HERE):
    if f.startswith("fig_") and f.endswith(".pdf") and f[:-4] in src:   # only figures the paper uses
        shutil.copy(os.path.join(HERE, f), os.path.join(HERE, "cvpr", f))
print("wrote cvpr/main.tex and copied figures")

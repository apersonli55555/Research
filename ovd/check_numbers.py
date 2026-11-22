#!/usr/bin/env python3
"""Traceability check: every decimal number in the manuscript must appear in ovd/RESULTS.md or ovd/results/*.

Numbers are compared as written and with common rescalings (0.083 <-> 8.3 points, 0.167 <-> 16.7%). Prints any
number that cannot be traced; exit code 1 if there are any.

  python ovd/check_numbers.py [ovd/paper/paper.tex]
"""
import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def numbers_in(text):
    return set(re.findall(r"(?<![\w.])[-+]?\d*\.\d+", text))


def variants(x):
    v = float(x); out = set()
    for f in (1, 100, 0.01):
        y = abs(v * f)
        for d in (0, 1, 2, 3, 4):
            out.add(f"{y:.{d}f}")
    return out


def corpus():
    txt = open(os.path.join(HERE, "RESULTS.md")).read()
    for p in glob.glob(os.path.join(HERE, "results", "*")):
        txt += "\n" + open(p).read()
    nums = set()
    for n in numbers_in(txt):
        nums |= variants(n)
    return nums


def main():
    tex = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "paper", "paper.tex")
    body = open(tex).read()
    body = body[body.index(r"\begin{abstract}"):body.index(r"\begin{thebibliography}")]
    body = re.sub(r"(?m)%.*$", "", body)
    body = re.sub(r"\\(includegraphics|setlength|tabcolsep)[^\n]*", "", body)
    known = corpus()
    missing = []
    for n in sorted(numbers_in(body), key=lambda s: float(s)):
        if not (variants(n) & known):
            missing.append(n)
    total = len(numbers_in(body))
    print(f"{total} distinct decimal numbers in the manuscript; {total - len(missing)} traced; {len(missing)} not traced")
    for n in missing:
        ctx = [m.start() for m in re.finditer(re.escape(n), body)][:1]
        print(f"  NOT TRACED {n}: ...{body[max(0, ctx[0] - 60):ctx[0] + 20].replace(chr(10), ' ')}...")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check the proposed replacement texts of the seven Chapter-5 audits (not an audit).

Copies docs/techref to a temporary directory, replaces in the COPY of chapters/05_vof_transport.tex (taken from the audited commit ec20986) each passage named in
the audits by the ```latex block of the corresponding audit.md, and builds chapter 5 with latexmk (the real source is not
touched).  Fails if an anchor passage no longer matches the source (the audited text changed) or if LaTeX reports an error.

    python3 proof_review/_shared/check_vof_replacements.py          (from anywhere inside the repository)
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True,
                                    cwd=Path(__file__).resolve().parent).strip())
audits = root / "proof_review"


def blocks(name):
    text = (audits / name / "audit.md").read_text()
    return re.findall(r"```latex\n(.*?)```", text, re.S)


# (audit folder, index of the latex block, first words of the replaced passage, last words of the replaced passage)
REPLACEMENTS = [
    ("eq-vof-fluctuation", 0, "Without snapping~\\cref{eq:vof:fluctuation} equals~\\cref{eq:vof:scalar};", "in cut cells."),
    ("eq-vof-scalar-balance", 0, "A snap to a positive fraction adds\n", "(\\cref{sec:vof:projected})."),
    ("eq-vof-frozen-flux", 0, "Without a solid and with $\\velo^{\\liq}_f=\\velo^{\\gas}_f$ the two subface",
     "one-velocity rectangle spans the whole face)."),
    ("vof-frozen-dilation", 0, "The dilation of a step,\n", "(\\cref{ch:projection})."),
    ("vof-projected-dilation", 0, "For the projected carrier every cell with\n",
     "$\\dt\\projtol\\sum_{\\indicator{\\liq}_c=1}\\meas{\\opencell{c}}$."),
    ("vof-boundedness", 0, "They are also disjoint in a cut donor whose two sides normal to", "$\\drdepth{f}=1/\\sqrt2$."),
    ("vof-boundedness", 1, "(the donor is mixed; pure donors\nneed no rectangle)", "need no rectangle)"),
]

work = Path(tempfile.mkdtemp(prefix="techref_check_"))
shutil.copytree(root / "docs" / "techref", work, dirs_exist_ok=True)
chapter = work / "chapters" / "05_vof_transport.tex"
# base text: the chapter of the audited commit (the working tree may already contain the replacements)
tex = subprocess.check_output(["git", "show", "ec20986:docs/techref/chapters/05_vof_transport.tex"], text=True, cwd=root)
tex = tex.replace("\\emph{Boundedness.} The run-time", "\\emph{Boundedness.}\\phantomsection\\label{prop:vof:boundedness} The run-time", 1)
for folder, index, start, end in REPLACEMENTS:
    new = blocks(folder)[index].rstrip("\n")
    try:
        i = tex.index(start)
        j = tex.index(end, i) + len(end)
    except ValueError:
        sys.exit(f"anchor of {folder} block {index} not found in the source: the audited text changed")
    tex = tex[:i] + new + tex[j:]
chapter.write_text(tex)

main = (work / "techref.tex").read_text()
preamble = main[: main.index("\\begin{document}")]
(work / "test.tex").write_text(preamble + "\\begin{document}\n\\mainmatter\n\\input{chapters/05_vof_transport}\n\\end{document}\n")
result = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "test.tex"], cwd=work,
                        capture_output=True, text=True)
log = (work / "test.log").read_text() if (work / "test.log").exists() else ""
errors = [line for line in log.splitlines() if line.startswith("!")]
if result.returncode != 0 or errors:
    print("\n".join(errors) or result.stdout[-2000:])
    sys.exit(f"LaTeX failed (work directory {work})")
unresolved = sorted(set(re.findall(r"Reference `([^']*)' on page", log)))
own = [label for label in unresolved if label.startswith(("eq:vof", "sec:vof", "tab:vof", "fig:vof", "dev:vof"))]
if own:
    sys.exit(f"unresolved chapter-5 labels: {own}")
print(f"ok: {len(REPLACEMENTS)} replacement texts applied and chapter 5 builds ({len(unresolved)} cross-chapter labels unresolved, expected)")
shutil.rmtree(work, ignore_errors=True)

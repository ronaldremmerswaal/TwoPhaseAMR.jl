#!/usr/bin/env python3
"""Check the proposed replacement texts of the five Chapter-6 momentum audits (not an audit).

Copies docs/techref to a temporary directory, replaces in the COPY of chapters/06_momentum_transport.tex (taken from the audited commit ad10545)
each passage named in the audits by the ```latex block of the corresponding audit.md, and builds chapter 6 with latexmk (the real source is not
touched).  Fails if an anchor passage no longer matches the source (the audited text changed), if a replaced label is lost or duplicated, or if LaTeX
reports an error.

    python3 proof_review/_shared/check_momentum_replacements.py          (from anywhere inside the repository)
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
    ("eq-momentum-identity", 0, "For every unknown $f$\nwhose cells have unknowns on both sides", "quantifies the effect."),
    ("eq-momentum-skew", 0, "the state is the mean\nof the two unknowns and~\\cref{eq:momentum:operator} gives, on a periodic mesh",
     "is not skew on a composite mesh."),
    ("eq-momentum-mass-identity", 0, "with the dilation of the split update. By~\\cref{eq:momentum:identity} and the",
     "two weightings (\\issue{63})."),
    ("sec-momentum-cfl", 0, "On a uniform mesh\nwithout solid this bounds the dual Courant numbers:", "with $\\stagvol{f}=\\meas{c}$."),
    ("sec-momentum-cfl", 1, "Under the bound\nthe unsplit upwind update is a convex combination of old values when the\nvelocity is solenoidal (test below).",
     "velocity is solenoidal (test below)."),
    ("eq-momentum-ctu", 0, "Keeping the partial interpolated volumes separate lets", "and 6.3}."),
]

work = Path(tempfile.mkdtemp(prefix="techref_check_"))
shutil.copytree(root / "docs" / "techref", work, dirs_exist_ok=True)
chapter = work / "chapters" / "06_momentum_transport.tex"
# base text: the chapter of the audited commit (the working tree may already contain the replacements)
tex = subprocess.check_output(["git", "show", "ad10545:docs/techref/chapters/06_momentum_transport.tex"], text=True, cwd=root)
labels_before = re.findall(r"\\label\{([^}]*)\}", tex)
for folder, index, start, end in REPLACEMENTS:
    new = blocks(folder)[index].rstrip("\n")
    try:
        i = tex.index(start)
        j = tex.index(end, i) + len(end)
    except ValueError:
        sys.exit(f"anchor of {folder} block {index} not found in the source: the audited text changed")
    tex = tex[:i] + new + tex[j:]
chapter.write_text(tex)
labels_after = re.findall(r"\\label\{([^}]*)\}", tex)
lost = sorted(set(labels_before) - set(labels_after))
duplicated = sorted({label for label in labels_after if labels_after.count(label) > 1})
added = sorted(set(labels_after) - set(labels_before))
if lost or duplicated:
    sys.exit(f"labels lost: {lost}; labels duplicated: {duplicated}")

main = (work / "techref.tex").read_text()
preamble = main[: main.index("\\begin{document}")]
(work / "test.tex").write_text(preamble + "\\begin{document}\n\\mainmatter\n\\input{chapters/06_momentum_transport}\n\\end{document}\n")
result = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "test.tex"], cwd=work,
                        capture_output=True, text=True)
log = (work / "test.log").read_text() if (work / "test.log").exists() else ""
errors = [line for line in log.splitlines() if line.startswith("!")]
if result.returncode != 0 or errors:
    print("\n".join(errors) or result.stdout[-2000:])
    sys.exit(f"LaTeX failed (work directory {work})")
unresolved = sorted(set(re.findall(r"Reference `([^']*)' on page", log)))
own = [label for label in unresolved if label.startswith(("eq:momentum", "sec:momentum", "tab:momentum", "fig:momentum", "dev:momentum"))]
if own:
    sys.exit(f"unresolved chapter-6 labels: {own}")
print(f"ok: {len(REPLACEMENTS)} replacement texts applied and chapter 6 builds (new labels: {added}; "
      f"{len(unresolved)} cross-chapter labels unresolved, expected)")
shutil.rmtree(work, ignore_errors=True)

"""Run every check of the audit of notes/mimetic_momentum_transport.tex (NumPy only; no Julia).
Command: python3 proof_review/mimetic-momentum-transport/scripts/run_all.py   (exit code != 0 if any check fails)
Checks marked FINDING assert a defect/limitation of the note and pass while the defect is present.
"""
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
steps = [
    "step01_prop_constant.py",
    "step02_mass_match.py",
    "step03_obstruction_flux.py",
    "step04_theta.py",
    "step05_redistribution.py",
    "step06_kernel_local_Y.py",
    "step07_increment_dimomentum_dp.py",
]
bad = []
for s in steps:
    print(f"\n##### {s}")
    r = subprocess.run([sys.executable, os.path.join(here, s)], cwd=here, capture_output=True, text=True)
    lines = r.stdout.strip().splitlines()
    print("\n".join(l for l in lines if l.startswith("==") or "FAIL" in l or "FINDING" in l or "Traceback" in l))
    if r.returncode != 0:
        print(r.stderr[-1500:])
        bad.append(s)
print("\nALL CHECKS PASSED" if not bad else f"\nFAILED STEPS: {bad}")
sys.exit(1 if bad else 0)

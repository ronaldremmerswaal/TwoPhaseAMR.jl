"""Shared helpers for the audit of notes/mimetic_momentum_transport.tex (no Julia, NumPy only).

Run any step with `python3 proof_review/mimetic-momentum-transport/scripts/<step>.py`,
everything with `python3 proof_review/mimetic-momentum-transport/scripts/run_all.py`.
"""
import sys
import numpy as np


class Checker:
    def __init__(self, name):
        self.name = name
        self.n = 0
        self.fails = []

    def _rec(self, ok, label, info):
        self.n += 1
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {label}  {info}")
        if not ok:
            self.fails.append(label)

    def close(self, label, a, b, tol=1e-10):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        err = float(np.max(np.abs(a - b))) if a.size else 0.0
        scale = max(1.0, float(np.max(np.abs(a))) if a.size else 0.0,
                    float(np.max(np.abs(b))) if b.size else 0.0)
        self._rec(err <= tol * scale, label, f"(max err {err:.2e}, scale {scale:.2e})")

    def true(self, label, cond, info=""):
        self._rec(bool(cond), label, info)

    def finish(self):
        print(f"== {self.name}: {self.n - len(self.fails)}/{self.n} checks passed")
        if self.fails:
            print("   FAILED:", "; ".join(self.fails))
        return 0 if not self.fails else 1


def spd(rng, n, cond=5.0):
    A = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)
    ev = np.linspace(1.0, cond, n)
    return (Q * ev) @ Q.T


def sym(A):
    return 0.5 * (A + A.T)


def nullspace(A, tol=1e-10):
    u, s, vt = np.linalg.svd(A)
    r = int((s > tol * max(1.0, s[0] if s.size else 1.0)).sum())
    return vt[r:].T


def exit_code(chk):
    sys.exit(chk.finish())

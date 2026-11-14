"""eq:match (mass matching) and its side claims  [notes/mimetic_momentum_transport.tex, l.~178-195]
Checks:
  (1) sliding x* by s eta-hat changes G^l by -s|eta| eta-hat eta-hat^T (and G^g by the opposite)  [metric_options prop:gram(c)]
  (2) least-norm slides = Lagrange formula  s_c = |eta_c| <P_c, Lambda>, sum_c |eta_c|^2 <P_c,Lambda> P_c = T  (P_c = eta-hat eta-hat^T)
      against an independent min-norm solve (np.linalg.lstsq) of  sum_c s_c |eta_c| P_c = T.
  (3) after the slides  Z^T M Z = Z^T W Z  for both phases (full assembled M on the random MAC model).
  (4) FINDING: the 3x3 system is singular whenever the normals take <= 2 distinct directions (mod sign),
      not only when all normals are parallel; then the system is solvable only if T lies in span{P_c}.
Run: python3 proof_review/mimetic-momentum-transport/scripts/step02_mass_match.py
"""
import numpy as np
from kit import Checker, exit_code
from meshmodel import Mesh

chk = Checker("step02 eq:match")

# ---------------- (1)+(3) on the random MAC mesh ----------------
rng = np.random.default_rng(11)
m = Mesh(rng, 4, 4, h=0.6)
h, F = m.h, m.F
xs0 = [m.line_point(c, s=rng.uniform(-0.04, 0.04) * h) for c in range(m.ncell)]
for c in range(m.ncell):
    eta, _ = m.eta_tau(c)
    en = np.linalg.norm(eta)
    eh = eta / en
    s = 0.013
    dG = m.G(c, 0, xs0[c] + s * eh) - m.G(c, 0, xs0[c])
    chk.close(f"cell{c}: (1) dG^l = -s|eta| eh eh^T", dG, -s * en * np.outer(eh, eh), 1e-13)
    dGg = m.G(c, 1, xs0[c] + s * eh) - m.G(c, 1, xs0[c])
    chk.close(f"cell{c}: (1) dG^g = +s|eta| eh eh^T", dGg, s * en * np.outer(eh, eh), 1e-13)
    if c >= 3:
        break

alpha = np.array([np.trace(m.G(c, 0, xs0[c])) / (2 * h * h) for c in range(m.ncell)]) + rng.uniform(-0.05, 0.05, m.ncell)
T = sum(m.G(c, 0, xs0[c]) - alpha[c] * h * h * np.eye(2) for c in range(m.ncell))
T = 0.5 * (T + T.T)
etas = [m.eta_tau(c)[0] for c in range(m.ncell)]
en = np.array([np.linalg.norm(e) for e in etas])
Ps = [np.outer(e, e) / (e @ e) for e in etas]
# Lagrange formula
A = sum(en[c] ** 2 * np.einsum("ij,kl->ijkl", Ps[c], Ps[c]) for c in range(m.ncell))   # acts on Lambda (2x2)
# 3x3 symmetric system in orthonormal basis of Sym(2): (xx, sqrt2 xy, yy)
def vec(S):
    return np.array([S[0, 0], np.sqrt(2) * S[0, 1], S[1, 1]])
def unvec(v):
    return np.array([[v[0], v[1] / np.sqrt(2)], [v[1] / np.sqrt(2), v[2]]])
A3 = sum(en[c] ** 2 * np.outer(vec(Ps[c]), vec(Ps[c])) for c in range(m.ncell))
lam = unvec(np.linalg.solve(A3, vec(T)))
s_formula = np.array([en[c] * np.sum(Ps[c] * lam) for c in range(m.ncell)])
# independent min-norm solve
C = np.array([en[c] * vec(Ps[c]) for c in range(m.ncell)]).T   # 3 x ncell
s_lsq = np.linalg.lstsq(C, vec(T), rcond=None)[0]
chk.close("(2) Lagrange formula = min-norm lstsq slides", s_formula, s_lsq, 1e-12)
chk.close("(2) constraint sum s|eta|P = T", C @ s_formula, vec(T), 1e-12)
# minimality: any other feasible s has larger norm
for _ in range(5):
    ns = np.linalg.svd(C)[2][3:].T
    s2 = s_formula + ns @ rng.standard_normal(ns.shape[1])
    chk.true("(2) other feasible slides have larger norm", np.linalg.norm(s2) >= np.linalg.norm(s_formula) - 1e-14)

# (3) slide and reassemble; compare with Z^T W Z
xs1 = [xs0[c] + s_formula[c] * etas[c] / en[c] for c in range(m.ncell)]
M1 = m.assemble(xs1)
M0 = m.assemble(xs0)
Z = m.Zcols()
for ph, V in ((0, alpha.sum() * h * h), (1, (1 - alpha).sum() * h * h)):
    Zp = Z[:, 2 * ph:2 * ph + 2]
    chk.close(f"(3) after matching: Z^T M Z = rho V I, phase {ph}", Zp.T @ M1 @ Zp, m.rho[ph] * V * np.eye(2), 1e-11)
    before = np.max(np.abs(Zp.T @ M0 @ Zp - m.rho[ph] * V * np.eye(2)))
    chk.true(f"(3) before matching the mismatch is nonzero, phase {ph}", before > 1e-4, f"(mismatch {before:.2e})")

# ---------------- (4) singularity claim ----------------
def system_rank(angles):
    P = [np.outer([np.cos(t), np.sin(t)], [np.cos(t), np.sin(t)]) for t in angles]
    Am = sum(np.outer(vec(p), vec(p)) for p in P)
    return int(np.linalg.matrix_rank(Am, tol=1e-10)), P

rk1, _ = system_rank([0.3, 0.3 + np.pi, 0.3])                 # all parallel (also antiparallel)
rk2, P2 = system_rank([0.0, np.pi / 2, 0.0, np.pi / 2 + np.pi, 0.0])  # normals along +-x, +-y only
rk3, _ = system_rank([0.1, 0.9, 2.0])
chk.true("(4) all normals parallel: rank 1", rk1 == 1)
chk.true("(4) FINDING: only two normal directions (x and y): rank 2, singular although not parallel", rk2 == 2)
chk.true("(4) three distinct directions: rank 3", rk3 == 3)
# solvability: T generic is not in the range -> residual of the constraint is nonzero
Cm = np.array([vec(p) for p in P2]).T
Tg = np.array([[0.3, 0.2], [0.2, -0.1]])
sol = np.linalg.lstsq(Cm, vec(Tg), rcond=None)[0]
res = np.linalg.norm(Cm @ sol - vec(Tg))
chk.true("(4) two directions: generic T is NOT matchable (residual > 0)", res > 0.1, f"(residual {res:.3f}; = |T_xy| sqrt2 = {np.sqrt(2)*0.2:.3f})")
# T in the span is matchable
Tin = 0.4 * P2[0] - 0.7 * P2[1]
sol = np.linalg.lstsq(Cm, vec(Tin), rcond=None)[0]
chk.close("(4) two directions: T in span matchable", Cm @ sol, vec(Tin), 1e-12)
exit_code(chk)

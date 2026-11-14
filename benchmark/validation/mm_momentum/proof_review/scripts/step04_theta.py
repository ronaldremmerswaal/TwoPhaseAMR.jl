"""prop:theta (a),(b),(c), orphan remark, and the 'trilemma' sentence  [notes/mimetic_momentum_transport.tex, l.~255-300]
D_theta:  M+u+ = M+ ubar + W+ (u_W - ubar) + (1-theta)(c - E+ ubar),  c = E u mapped.
Checks:
  T0 u_W = Z U for constants; expansion  M+u+ - Mu = W+u_W - Wu + theta (E+-E) u
  Ta M+(u+ - ZU) = -(1-theta)(E+-E) Z U
  Tb Z^T M+ u+ - Z^T M u = theta Z^T (E+-E) u
  Tc semi-discrete energy rate (theta - 1/2) u^T Edot u   (C + C^T = -Wdot, Z^T C = 0; dt -> 0, error O(dt))
  To orphans: Z+^T M+u+ - Z^T M u = theta (Z+^T E+ ubar - Z^T E u) - (1-theta) orphan,   orphan = Z^T E u - Z+^T (mapped E u)
  Tt trilemma: R1 defect = theta Z^T dE u, R2 defect = (1-theta) dE Z U, R3 defect = (theta-1/2) u^T Edot u.
     FINDING: 'no theta satisfies two of R1-R3 unless dE u = 0' is not the right condition (counterexamples below).
Run: python3 proof_review/mimetic-momentum-transport/scripts/step04_theta.py
"""
import numpy as np
from kit import Checker, exit_code, sym, spd
from absmodel import *

chk = Checker("step04 prop:theta")
thetas = (0.0, 0.25, 0.5, 0.8, 1.0)
for seed in range(5):
    rng = np.random.default_rng(400 + seed)
    Z = build_Z((4, 4, 3, 4), (1, 1))
    N = Z.shape[0]
    P, Pi = proj(Z)
    M, W = make_metric(rng, Z)
    dw = dW_conserving(rng, Z, 0.1)
    Wp = W + np.diag(dw)
    Mp = M + sym(rng.standard_normal((N, N))) * 0.05      # generic: Z^T dM Z != 0
    Ct = transport_Ct(rng, Z, dw)
    E, Ep = M - W, Mp - Wp
    dE = Ep - E
    u = rng.standard_normal(N)
    U = rng.standard_normal(4)
    for th in thetas:
        up, uW = theta_update(M, W, Mp, Wp, Ct, u, th)
        # expansion
        chk.close(f"seed{seed} th={th}: T0 M+u+ - Mu = W+uW - Wu + th dE u", Mp @ up - M @ u, Wp @ uW - W @ u + th * dE @ u, 1e-11)
        # (b)
        chk.close(f"seed{seed} th={th}: Tb momentum", Z.T @ Mp @ up - Z.T @ M @ u, th * Z.T @ dE @ u, 1e-11)
        # (a)
        uc = Z @ U
        ucp, ucW = theta_update(M, W, Mp, Wp, Ct, uc, th)
        chk.close(f"seed{seed} th={th}: T0 uW = ZU", ucW, uc, 1e-11)
        chk.close(f"seed{seed} th={th}: Ta constants", Mp @ (ucp - uc), -(1 - th) * dE @ Z @ U, 1e-11)
    # exactness iff
    for th in thetas:
        ucp, _ = theta_update(M, W, Mp, Wp, Ct, Z @ U, th)
        chk.true(f"seed{seed} th={th}: constants exact <=> theta=1", (np.max(np.abs(ucp - Z @ U)) < 1e-12) == (th == 1.0))
        up, _ = theta_update(M, W, Mp, Wp, Ct, u, th)
        mom_ok = np.max(np.abs(Z.T @ Mp @ up - Z.T @ M @ u)) < 1e-12
        chk.true(f"seed{seed} th={th}: momentum exact <=> theta=0", mom_ok == (th == 0.0))

    # ---------- (c) semi-discrete energy ----------
    Mdot = sym(rng.standard_normal((N, N))) * 0.3
    wd = dW_conserving(rng, Z, 0.3)
    Wdot = np.diag(wd)
    Edot = Mdot - Wdot
    D = -0.5 * Wdot @ Z
    A = D @ P.T - P @ D.T
    S = rng.standard_normal((N, N)); S = S - S.T
    I = np.eye(N)
    A = A + 0.3 * (I - Pi) @ S @ (I - Pi)
    C = -0.5 * Wdot + A
    chk.close(f"seed{seed}: Tc C + C^T = -Wdot", C + C.T, -Wdot, 1e-12)
    chk.close(f"seed{seed}: Tc Z^T C = 0", Z.T @ C, np.zeros((4, N)), 1e-12)
    chk.close(f"seed{seed}: Tc C Z = -Wdot Z", C @ Z, -Wdot @ Z, 1e-12)
    errs = {}
    for th in (0.0, 0.5, 1.0):
        prev = None
        for dt in (1e-4, 1e-5):
            Mp_, Wp_ = M + dt * Mdot, W + dt * Wdot
            up, _ = theta_update(M, W, Mp_, Wp_, dt * C, u, th)
            rate = (0.5 * up @ Mp_ @ up - 0.5 * u @ M @ u) / dt
            pred = (th - 0.5) * u @ Edot @ u
            errs[(th, dt)] = abs(rate - pred)
        chk.true(f"seed{seed} th={th}: Tc energy rate -> (th-1/2) u^T Edot u, error O(dt)",
                 errs[(th, 1e-5)] < 0.25 * errs[(th, 1e-4)] + 1e-9 and errs[(th, 1e-5)] < 2e-2,
                 f"(err {errs[(th,1e-4)]:.2e} -> {errs[(th,1e-5)]:.2e}; pred {pred:.3f})")
    # exactness iff theta=1/2 (u^T Edot u != 0)
    chk.true(f"seed{seed}: u^T Edot u != 0 (energy defect nonzero off theta=1/2)", abs(u @ Edot @ u) > 1e-3)

    # ---------- orphans ----------
    N0 = N
    Zo = Z.copy()
    # remove 3 ordinary (non-identified) dofs, add 2 new ones
    orph_idx = [1, 6, N0 - 4]
    keep = [i for i in range(N0) if i not in orph_idx]
    new_rows = np.array([[1, 0, 0, 0], [0, 0, 0, 1.0]])
    Zn = np.vstack([Zo[keep], new_rows])
    N1 = Zn.shape[0]
    Sel = np.zeros((N1, N0))
    for inew, iold in enumerate(keep):
        Sel[inew, iold] = 1.0
    Mn = spd(rng, N1, 6.0) * 2.0
    Wn = np.diag(rng.uniform(0.5, 1.5, N1))
    En_ = Mn - Wn
    ubar = np.concatenate([u[keep], rng.standard_normal(2)])
    cvec = E @ u
    cmap = Sel @ cvec
    orphan = Zo.T @ cvec - Zn.T @ cmap
    chk.close(f"seed{seed}: To orphan = sum of excess of vanished dofs", orphan, sum(Zo[i] * cvec[i] for i in orph_idx), 1e-12)
    Pn, Pin = proj(Zn)
    rhsW = rng.standard_normal(N1)
    rhsW = rhsW - Pn @ (Zn.T @ rhsW - Zo.T @ W @ u)       # impose Z+^T W+ uW = Z^T W u  (the w-transport conservation)
    uWn = np.linalg.solve(Wn, rhsW)
    for th in thetas:
        rhs = Mn @ ubar + Wn @ (uWn - ubar) + (1 - th) * (cmap - En_ @ ubar)
        upn = np.linalg.solve(Mn, rhs)
        lhs = Zn.T @ Mn @ upn - Zo.T @ M @ u
        rhs_ = th * (Zn.T @ En_ @ ubar - Zo.T @ E @ u) - (1 - th) * orphan
        chk.close(f"seed{seed} th={th}: To momentum with orphans/new dofs", lhs, rhs_, 1e-11)

# ---------- trilemma: counterexamples to the stated condition ----------
rng = np.random.default_rng(777)
Z = build_Z((4, 4, 4, 4), (1, 1))
N = Z.shape[0]
P, Pi = proj(Z)
M, W = make_metric(rng, Z)
dw = dW_conserving(rng, Z, 0.1)
Wp = W + np.diag(dw)
Ct = transport_Ct(rng, Z, dw)
E = M - W
# (i) dE != 0, dE Z = 0 : R1, R2 hold for EVERY theta (all u), although dE u != 0
S = sym_kills_Z(rng, Z, 0.1)
Mp = M + np.diag(dw) + S
dE = (Mp - Wp) - E
chk.close("Tt(i) dE Z = 0", dE @ Z, np.zeros((N, 4)), 1e-13)
u = rng.standard_normal(N)
chk.true("Tt(i) but dE u != 0", np.max(np.abs(dE @ u)) > 1e-3)
for th in (0.0, 0.3, 0.5, 1.0):
    up, _ = theta_update(M, W, Mp, Wp, Ct, u, th)
    mom = np.max(np.abs(Z.T @ Mp @ up - Z.T @ M @ u))
    ucp, _ = theta_update(M, W, Mp, Wp, Ct, Z @ np.ones(4), th)
    const = np.max(np.abs(ucp - Z @ np.ones(4)))
    chk.true(f"Tt(i) th={th}: R1 and R2 both satisfied although dE u != 0", mom < 1e-12 and const < 1e-12)
# (ii) fixed u with Z^T dE u = 0 (u orthogonal to dE Z), generic dE: R1 holds for all theta, R2 at theta=1
Mp2 = M + np.diag(dw) + sym_conserving(rng, Z, 0.1)
dE2 = (Mp2 - (W + np.diag(dw))) - E
B = dE2 @ Z
u2 = rng.standard_normal(N)
u2 = u2 - B @ np.linalg.lstsq(B, u2, rcond=None)[0]   # u2 orthogonal to the columns of dE Z
chk.close("Tt(ii) Z^T dE u2 = 0", Z.T @ dE2 @ u2, np.zeros(4), 1e-12)
chk.true("Tt(ii) dE u2 != 0 and dE Z != 0", np.max(np.abs(dE2 @ u2)) > 1e-3 and np.max(np.abs(B)) > 1e-3)
up, _ = theta_update(M, W, Mp2, W + np.diag(dw), Ct, u2, 1.0)
ucp, _ = theta_update(M, W, Mp2, W + np.diag(dw), Ct, Z @ np.ones(4), 1.0)
chk.true("Tt(ii) theta=1: R1 (for this u2) and R2 both exact although dE u2 != 0",
         np.max(np.abs(Z.T @ Mp2 @ up - Z.T @ M @ u2)) < 1e-12 and np.max(np.abs(ucp - Z @ np.ones(4))) < 1e-12)
# (iii) the correct statement for 'all u': R1 for all u  <=>  theta=0 or dE Z=0 (symmetry of dE)
chk.close("Tt(iii) Z^T dE = (dE Z)^T (dE symmetric)", Z.T @ dE2, (dE2 @ Z).T, 1e-13)
exit_code(chk)

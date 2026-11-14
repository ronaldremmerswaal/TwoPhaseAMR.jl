"""prop:redistribution: existence formula, three properties, necessity  [notes/mimetic_momentum_transport.tex, l.~305-330]
Checks:
  Y1 D = dE Z / 2, P = Z (Z^T Z)^{-1}, Y = D P^T - P D^T is skew and Y Z = D iff Z^T D = 0 (here: Z^T dE Z = 0).
  Y2 NECESSITY: if Z^T dE Z != 0 no skew Y has Y Z = D  (Z^T Y Z skew, Z^T D symmetric): best skew fit leaves a residual > 0.
  DG with Psi = 0:  M+u+ = [DS rhs, theta=1/2] + Y ubar :
  Ga constants M+(u+ - ZU) = 0;  Gb momentum Z^T M+u+ = Z^T M u (exact);
  Gc semi-discrete energy rate 0 (Y = dt*Ydot, Ydot skew with Ydot Z = Edot Z / 2), error O(dt).
  Gd the remark 'DG = DS + Y' also keeps R1,R2,R3 simultaneously although dE u != 0 (so it leaves the theta family).
Run: python3 proof_review/mimetic-momentum-transport/scripts/step05_redistribution.py
"""
import numpy as np
from kit import Checker, exit_code, sym
from absmodel import *

chk = Checker("step05 prop:redistribution")


def skew_fit_residual(D, Z):
    """min over skew Y of ||Y Z - D||_F (Y parametrised by its strict upper triangle)."""
    N = Z.shape[0]
    cols = []
    for i in range(N):
        for j in range(i + 1, N):
            Yij = np.zeros((N, N)); Yij[i, j] = 1; Yij[j, i] = -1
            cols.append((Yij @ Z).reshape(-1))
    A = np.array(cols).T
    b = D.reshape(-1)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.linalg.norm(A @ x - b)


for seed in range(5):
    rng = np.random.default_rng(500 + seed)
    Z = build_Z((3, 3, 3, 3), (1, 1))
    N = Z.shape[0]
    P, Pi = proj(Z)
    M, W = make_metric(rng, Z)
    dw = dW_conserving(rng, Z, 0.1)
    Wp = W + np.diag(dw)
    Ct = transport_Ct(rng, Z, dw)
    E = M - W
    # mass-matched case: Z^T dE Z = 0
    Mp = M + np.diag(dw) + sym_conserving(rng, Z, 0.08)
    Ep = Mp - Wp
    dE = Ep - E
    chk.close(f"seed{seed}: Z^T dE Z = 0", Z.T @ dE @ Z, np.zeros((4, 4)), 1e-12)
    D = 0.5 * dE @ Z
    Y = D @ P.T - P @ D.T
    chk.close(f"seed{seed}: Y1 Y skew", Y + Y.T, np.zeros((N, N)), 1e-13)
    chk.close(f"seed{seed}: Y1 Y Z = D", Y @ Z, D, 1e-12)
    chk.true(f"seed{seed}: Y2 skew fit residual = 0 when Z^T D = 0", skew_fit_residual(D, Z) < 1e-9)
    # necessity: generic dE
    Mp_bad = M + np.diag(dw) + sym(rng.standard_normal((N, N))) * 0.05
    dEb = (Mp_bad - Wp) - E
    Db = 0.5 * dEb @ Z
    r = skew_fit_residual(Db, Z)
    chk.true(f"seed{seed}: Y2 NECESSITY: Z^T dE Z != 0 -> no skew Y with YZ = D", r > 1e-3, f"(residual {r:.3e}, |Z^T D| = {np.max(np.abs(Z.T@Db)):.2e})")
    # also the skew fit residual equals the norm of the part of D orthogonal to the range -> sanity: residual^2 >= |sym part of Z^T D|^2/ something
    # DG update
    u = rng.standard_normal(N)
    U = rng.standard_normal(4)
    up, uW = theta_update(M, W, Mp, Wp, Ct, u, 0.5, extra_rhs=Y @ u)
    chk.close(f"seed{seed}: Gb momentum exact", Z.T @ Mp @ up, Z.T @ M @ u, 1e-11)
    uc = Z @ U
    ucp, _ = theta_update(M, W, Mp, Wp, Ct, uc, 0.5, extra_rhs=Y @ uc)
    chk.close(f"seed{seed}: Ga constants exact", ucp, uc, 1e-11)
    chk.true(f"seed{seed}: Gd dE u != 0 (DG leaves the theta family)", np.max(np.abs(dE @ u)) > 1e-3)
    # (c) semi-discrete
    wd = dW_conserving(rng, Z, 0.3)
    Wdot = np.diag(wd)
    Mdot = Wdot + sym_conserving(rng, Z, 0.3)              # Z^T Edot Z = 0
    Edot = Mdot - Wdot
    chk.close(f"seed{seed}: Z^T Edot Z = 0", Z.T @ Edot @ Z, np.zeros((4, 4)), 1e-12)
    Dd = 0.5 * Edot @ Z
    Yd = Dd @ P.T - P @ Dd.T
    Dw = -0.5 * Wdot @ Z
    A = Dw @ P.T - P @ Dw.T
    S = rng.standard_normal((N, N)); S = S - S.T
    A = A + 0.3 * (np.eye(N) - Pi) @ S @ (np.eye(N) - Pi)
    C = -0.5 * Wdot + A
    errs = []
    for dt in (1e-4, 1e-5):
        Mp_, Wp_ = M + dt * Mdot, W + dt * Wdot
        up, _ = theta_update(M, W, Mp_, Wp_, dt * C, u, 0.5, extra_rhs=dt * Yd @ u)
        rate = (0.5 * up @ Mp_ @ up - 0.5 * u @ M @ u) / dt
        errs.append(abs(rate))
    chk.true(f"seed{seed}: Gc DG energy rate -> 0 as O(dt)", errs[1] < 0.25 * errs[0] + 1e-9 and errs[1] < 1e-2, f"(|rate| {errs[0]:.2e} -> {errs[1]:.2e}; u^T Edot u = {u@Edot@u:.3f})")
    # without Y, theta=1/2 still has constants error (shows that Y is what fixes (a),(b))
    ucp0, _ = theta_update(M, W, Mp, Wp, Ct, uc, 0.5)
    chk.true(f"seed{seed}: DS alone: constants not exact (dE Z != 0)", np.max(np.abs(ucp0 - uc)) > 1e-4)
exit_code(chk)

"""prop:increment, prop:di-momentum (summation by parts, DP restoration)  [notes/mimetic_momentum_transport.tex, l.~335-375]
Checks:
  I1 ||delta||_{M}^2 <= lambda_max(W,M) ||v||_W^2  (delta = M^{-1} W v), equality for the top generalized eigenvector (sharp);
     ||delta||_inf <= ||M^{-1}W||_inf ||v||_inf ; constants: v = 0 -> delta = 0; DI keeps constants.
  I2 algebra gives no cut-volume independence: lambda_max >= max_i W_ii / M_ii  (counterexample with tiny M_ii) -- the 2.7 is a property of the assembly.
  S1 DI momentum telescoping formula  P^N - P^0 = Z^T E^N u^{N-1} - Z^T E^0 u^0 - sum_{n=1}^{N-1} Z^T E^n (u^n - u^{n-1}).
  S2 steady constant field: drift = Z^T (E^N - E^0) Z U bounded.
  P1 DP: Z^T M+ u+ - Z^T M u = sum_j Z^T dE Z_j <ubar>_j  (zero iff Z^T dE Z <u> = 0, in particular with mass matching),
     so  Z^T M+u+ = Z^T M u exactly when Z^T dE Z = 0; constants exact in all cases.
  P2 force bound  ||M^{-1} W Z_k||_inf d_k/(Z_k^T W Z_k) <= ||M^{-1}W||_inf |d_k|/(Z_k^T W Z_k).
Run: python3 proof_review/mimetic-momentum-transport/scripts/step07_increment_dimomentum_dp.py
"""
import numpy as np
from kit import Checker, exit_code, sym, spd
from absmodel import *
from scipy.linalg import eigh

chk = Checker("step07 prop:increment / prop:di-momentum")

# ---------------- increment ----------------
for seed in range(5):
    rng = np.random.default_rng(700 + seed)
    N = 20
    M = spd(rng, N, 8.0) * 2.0
    W = np.diag(rng.uniform(0.2, 2.0, N))
    lam, vec = eigh(W, M)               # W x = lambda M x
    lmax = lam[-1]
    x = vec[:, -1]
    for t in range(20):
        v = rng.standard_normal(N)
        delta = np.linalg.solve(M, W @ v)
        lhs = delta @ M @ delta
        chk.true(f"seed{seed}: I1 bound (random v)", lhs <= lmax * (v @ W @ v) * (1 + 1e-12))
    delta = np.linalg.solve(M, W @ x)
    chk.close(f"seed{seed}: I1 equality at top eigenvector", delta @ M @ delta, lmax * (x @ W @ x), 1e-11)
    A = np.linalg.solve(M, W)
    ninf = np.max(np.sum(np.abs(A), axis=1))
    ok = all(np.max(np.abs(A @ v)) <= ninf * np.max(np.abs(v)) * (1 + 1e-12) for v in rng.standard_normal((20, N)))
    chk.true(f"seed{seed}: I1 sup-norm bound", ok)
    chk.true(f"seed{seed}: I2 lambda_max >= max W_ii/M_ii", lmax >= np.max(np.diag(W) / np.diag(M)) - 1e-12)
# I2 counterexample: tiny diagonal entry
rng = np.random.default_rng(1)
N = 10
M = spd(rng, N, 3.0)
Wd = np.ones(N)
M2 = M.copy()
M2[0, :] = 0; M2[:, 0] = 0; M2[0, 0] = 1e-9           # a 'slaved' value with no tie weight
lam2, _ = eigh(np.diag(Wd), M2)
chk.true("I2 without the tie (M_00 = 1e-9, W_00 = 1) lambda_max = 1e9: algebra alone gives no uniform bound", lam2[-1] > 1e8)

# ---------------- DI telescoping ----------------
for seed in range(4):
    rng = np.random.default_rng(800 + seed)
    Z = build_Z((4, 3, 3, 4), (1, 1))
    N = Z.shape[0]
    M, W = make_metric(rng, Z)
    u = rng.standard_normal(N)
    Ms, Ws, us, Ps = [M], [W], [u], [Z.T @ M @ u]
    nsteps = 6
    for n in range(nsteps):
        dw = dW_conserving(rng, Z, 0.08)
        Wp = Ws[-1] + np.diag(dw)
        Mp = Ms[-1] + sym(rng.standard_normal((N, N))) * 0.03
        Mp = Mp + 3 * np.eye(N) * (np.min(np.linalg.eigvalsh(Mp)) < 0.5)
        Ct = transport_Ct(rng, Z, dw)
        up, _ = theta_update(Ms[-1], Ws[-1], Mp, Wp, Ct, us[-1], 1.0)
        Ms.append(Mp); Ws.append(Wp); us.append(up); Ps.append(Z.T @ Mp @ up)
    Es = [Ms[n] - Ws[n] for n in range(nsteps + 1)]
    lhs = Ps[nsteps] - Ps[0]
    rhs = Z.T @ Es[nsteps] @ us[nsteps - 1] - Z.T @ Es[0] @ us[0] - sum(Z.T @ Es[n] @ (us[n] - us[n - 1]) for n in range(1, nsteps))
    chk.close(f"seed{seed}: S1 telescoped DI momentum drift", lhs, rhs, 1e-10)
    # steady constants
    U = rng.standard_normal(4)
    uc = Z @ U
    Pc = [Z.T @ Ms[0] @ uc]
    ucur = uc
    for n in range(nsteps):
        Ct = transport_Ct(rng, Z, np.diag(Ws[n + 1] - Ws[n]))
        ucur, _ = theta_update(Ms[n], Ws[n], Ms[n + 1], Ws[n + 1], Ct, ucur, 1.0)
        Pc.append(Z.T @ Ms[n + 1] @ ucur)
    chk.close(f"seed{seed}: I1 DI keeps constants over {nsteps} steps", ucur, uc, 1e-10)
    chk.close(f"seed{seed}: S2 steady-field drift = Z^T (E^N-E^0) Z U", Pc[-1] - Pc[0], Z.T @ (Es[-1] - Es[0]) @ uc, 1e-10)

# ---------------- DP ----------------
def dp_update(Z, M, W, Mp, Wp, Ct, u, restore=True):
    N = Z.shape[0]
    ident = (Z.sum(axis=1) > 1.5)
    Zni = Z.copy(); Zni[ident, :] = 0
    E, Ep = M - W, Mp - Wp
    uW = np.linalg.solve(Wp, W @ u - Ct @ u)
    rhs = Mp @ u + Wp @ (uW - u)
    rhs_di = rhs.copy()
    wd = np.diag(Wp)
    means = np.array([(wd * Zni[:, j]) @ u / (wd * Zni[:, j]).sum() for j in range(4)])
    dE = Ep - E
    d = Z.T @ (Ep @ u - E @ u) - (Z.T @ dE @ Z) @ means
    force = np.zeros(N)
    tot = np.array([(wd * Zni[:, k]).sum() for k in range(4)])
    for k in range(4):
        force += wd * Zni[:, k] * d[k] / tot[k]
    if restore:
        rhs = rhs - force
    return np.linalg.solve(Mp, rhs), np.linalg.solve(Mp, rhs_di), d, tot, means

for seed in range(5):
    rng = np.random.default_rng(900 + seed)
    Z = build_Z((4, 4, 3, 4), (1, 1))
    N = Z.shape[0]
    M, W = make_metric(rng, Z)
    dw = dW_conserving(rng, Z, 0.1)
    Wp = W + np.diag(dw)
    Ct = transport_Ct(rng, Z, dw)
    E = M - W
    u = rng.standard_normal(N)
    U = rng.standard_normal(4)
    for matched in (True, False):
        Mp = M + np.diag(dw) + (sym_conserving(rng, Z, 0.08) if matched else sym(rng.standard_normal((N, N))) * 0.03)
        Ep = Mp - Wp
        dE = Ep - E
        up, up_di, d, tot, means = dp_update(Z, M, W, Mp, Wp, Ct, u)
        left = Z.T @ Mp @ up - Z.T @ M @ u
        pred = (Z.T @ dE @ Z) @ means
        chk.close(f"seed{seed} matched={matched}: P1 DP momentum change = Z^T dE Z <u>", left, pred, 1e-10)
        if matched:
            chk.close(f"seed{seed}: P1 mass matched -> DP momentum exact", left, np.zeros(4), 1e-10)
        else:
            chk.true(f"seed{seed}: P1 without mass matching DP is NOT momentum exact (constant part remains)", np.max(np.abs(left)) > 1e-4)
        # DI momentum change before restoration
        chk.close(f"seed{seed} matched={matched}: DI momentum change = Z^T dE u",
                  Z.T @ Mp @ up_di - Z.T @ M @ u, Z.T @ dE @ u, 1e-10)
        # constants
        upc, upc_di, dc, _, _ = dp_update(Z, M, W, Mp, Wp, Ct, Z @ U)
        chk.close(f"seed{seed} matched={matched}: P1 DP constants exact", upc, Z @ U, 1e-10)
        chk.close(f"seed{seed} matched={matched}: d_k = 0 for constants", dc, np.zeros(4), 1e-10)
        # P2: effect of the force
        eff = up - up_di
        bound = np.max(np.sum(np.abs(np.linalg.solve(Mp, Wp)), axis=1)) * np.max(np.abs(d) / tot)
        chk.true(f"seed{seed} matched={matched}: P2 |force effect|_inf <= 4 ||M^-1 W||_inf max|d_k|/tot_k (four fields, crude)",
                 np.max(np.abs(eff)) <= 4 * bound * (1 + 1e-12))
exit_code(chk)

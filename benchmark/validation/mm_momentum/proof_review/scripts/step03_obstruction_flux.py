"""prop:obstruction and prop:flux  [notes/mimetic_momentum_transport.tex, l.~200-235]
Checks (abstract random model, N=18):
  O1 update M+u+ = Mu - L u: momentum conserved for all u iff Z^T L = 0; constants kept iff L Z = -(M+-M) Z.
  O2 both together force Z^T (M+ - M) Z = 0; with Z^T (W+ - W) Z = 0 this is Z^T (E+ - E) Z = 0.
  O3 the necessary condition is also sufficient for a (dense) linear L:  L = D P^T, D = -(M+ - M)Z, P = Z (Z^T Z)^{-1}.
     When Z^T dM Z != 0 the linear system {Z^T L = 0, L Z = -dM Z} is infeasible (least-squares residual > 0).
  F1 flux form  L = C_W-step:  constants preserved  iff  (E+ - E) Z = 0   (checked both directions).
  F2 DF (theta = 0) of the note equals the flux form M+u+ = Mu - Ct u.
Run: python3 proof_review/mimetic-momentum-transport/scripts/step03_obstruction_flux.py
"""
import numpy as np
from kit import Checker, exit_code, sym
from absmodel import *

chk = Checker("step03 prop:obstruction / prop:flux")
for seed in range(5):
    rng = np.random.default_rng(300 + seed)
    Z = build_Z((4, 3, 4, 5), (1, 1))
    N = Z.shape[0]
    P, Pi = proj(Z)
    M, W = make_metric(rng, Z)
    dw = dW_conserving(rng, Z)
    Wp = W + np.diag(dw)
    chk.close(f"seed{seed}: Z^T dW Z = 0", Z.T @ np.diag(dw) @ Z, np.zeros((4, 4)), 1e-13)
    # --- generic dM (not conserving): obstruction nonzero
    dM = sym(rng.standard_normal((N, N))) * 0.05
    Mp = M + dM
    obs = Z.T @ dM @ Z
    chk.true(f"seed{seed}: generic Z^T dM Z != 0", np.max(np.abs(obs)) > 1e-3)
    # unknown L (N x N): constraints Z^T L = 0 (4N eqs), L Z = -dM Z (4N eqs), solved in least squares
    I = np.eye(N)
    A1 = np.kron(Z.T, I)            # vec(L Z) = (Z^T kron I) vec L  (column-major vec)
    A2 = np.kron(I, Z.T)            # vec(Z^T L) = (I kron Z^T) vec L
    A = np.vstack([A1, A2])
    b = np.concatenate([(-(dM @ Z)).reshape(-1, order="F"), np.zeros(4 * N)])
    sol, res, rk, sv = np.linalg.lstsq(A, b, rcond=None)
    resid = np.linalg.norm(A @ sol - b)
    chk.true(f"seed{seed}: O3 generic: system {{Z^T L=0, LZ=-dM Z}} infeasible", resid > 1e-3, f"(residual {resid:.3e})")
    # --- conserving dM (Z^T dM Z = 0): feasible, L = D P^T works
    dM2 = np.diag(dw) + sym_conserving(rng, Z, 0.05)   # Z^T dM2 Z = Z^T dW Z + 0 = 0
    chk.close(f"seed{seed}: Z^T dM2 Z = 0", Z.T @ dM2 @ Z, np.zeros((4, 4)), 1e-13)
    Mp2 = M + dM2
    D = -(dM2 @ Z)
    L = D @ P.T
    chk.close(f"seed{seed}: O3 Z^T L = 0", Z.T @ L, np.zeros((4, N)), 1e-12)
    chk.close(f"seed{seed}: O3 L Z = -dM Z", L @ Z, D, 1e-12)
    u = rng.standard_normal(N)
    up = np.linalg.solve(Mp2, M @ u - L @ u)
    chk.close(f"seed{seed}: O1 momentum conserved", Z.T @ Mp2 @ up, Z.T @ M @ u, 1e-11)
    U = rng.standard_normal(4)
    uc = Z @ U
    ucp = np.linalg.solve(Mp2, M @ uc - L @ uc)
    chk.close(f"seed{seed}: O1 constants preserved", ucp, uc, 1e-11)
    # O1 'only if': an L with Z^T L != 0 violates momentum for some u
    L2 = L + rng.standard_normal((N, N)) * 0.1
    ok_mom = np.max(np.abs(Z.T @ L2)) > 1e-3
    chk.true(f"seed{seed}: O1 Z^T L != 0 -> momentum changes for a suitable u", ok_mom)
    # O2: with W conservation, Z^T dM Z = Z^T dE Z
    chk.close(f"seed{seed}: O2 Z^T dM Z = Z^T dE Z when Z^T dW Z = 0",
              Z.T @ (Mp - M) @ Z, Z.T @ ((Mp - Wp) - (M - W)) @ Z, 1e-12)
    # --- prop:flux
    Ct = transport_Ct(rng, Z, dw)
    chk.close(f"seed{seed}: w-transport Z^T Ct = 0", Z.T @ Ct, np.zeros((4, N)), 1e-12)
    chk.close(f"seed{seed}: w-transport Ct Z = -(W+-W) Z", Ct @ Z, -(Wp - W) @ Z, 1e-12)
    E, Ep = M - W, Mp - Wp
    # flux form: M+ u+ = M u - Ct u   (L = Ct)
    upc = np.linalg.solve(Mp, M @ uc - Ct @ uc)
    # M+ (u+ - ZU) = M ZU - Ct ZU - M+ ZU = (W+-W)ZU - (M+-M) ZU = -(E+-E) Z U
    chk.close(f"seed{seed}: F1 M+(u+ - ZU) = -(E+ - E) Z U", Mp @ (upc - uc), -(Ep - E) @ Z @ U, 1e-11)
    # converse: choose M+ with (E+-E)Z = 0
    S = sym_kills_Z(rng, Z, 0.1)
    Mp3 = Wp + E + S              # E+ = E + S, S Z = 0
    chk.close(f"seed{seed}: F1 construction (E+-E)Z = 0", (Mp3 - Wp - E) @ Z, np.zeros((N, 4)), 1e-12)
    upc3 = np.linalg.solve(Mp3, M @ uc - Ct @ uc)
    chk.close(f"seed{seed}: F1 (E+-E)Z=0 -> constants preserved", upc3, uc, 1e-11)
    # F2: DF of the note
    u = rng.standard_normal(N)
    upDF, uW = theta_update(M, W, Mp, Wp, Ct, u, 0.0)
    chk.close(f"seed{seed}: F2 DF: M+u+ = Mu - Ct u", Mp @ upDF, M @ u - Ct @ u, 1e-11)
    # the E Z invariance can hold with E Z != 0 (static geometry): flux form then preserves constants
    Mp4 = M.copy(); Wp4 = W.copy()
    Ct4 = transport_Ct(rng, Z, np.zeros(N))
    up4 = np.linalg.solve(Mp4, M @ uc - Ct4 @ uc)
    chk.close(f"seed{seed}: F1 static geometry (E Z != 0, E+ = E): constants preserved", up4, uc, 1e-11)
    chk.true(f"seed{seed}: F1 ... although E Z != 0", np.max(np.abs((M - W) @ Z)) > 1e-2)
exit_code(chk)

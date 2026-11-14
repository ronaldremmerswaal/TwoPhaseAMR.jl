"""Abstract random model of the note (Section 1/3): M SPD, W diagonal > 0, E = M - W,
Z an N x 4 0/1 matrix (columns: liquid x, liquid y, gas x, gas y; identified rows have ones in two columns),
W-transport  W+ u_W = W u - Ct(u),  Z^T Ct = 0,  Ct Z = -(W+ - W) Z   (Ct = dt*C_W).
"""
import numpy as np
from kit import spd, sym, nullspace


def build_Z(counts=(4, 4, 4, 4), nid=(1, 1)):
    """counts = (liq x, liq y, gas x, gas y) ordinary values; nid = identified values per axis."""
    rows = []
    for col, n in enumerate(counts):
        for _ in range(n):
            r = np.zeros(4)
            r[col] = 1
            rows.append(r)
    for ax, n in enumerate(nid):
        for _ in range(n):
            r = np.zeros(4)
            r[ax] = 1       # liquid column of this axis
            r[2 + ax] = 1   # gas column of this axis
            rows.append(r)
    return np.array(rows)


def proj(Z):
    P = Z @ np.linalg.inv(Z.T @ Z)          # N x 4 ,  P^T Z = I
    return P, P @ Z.T                        # (P, Pi)


def make_metric(rng, Z, cond=6.0, wscale=1.0):
    N = Z.shape[0]
    M = spd(rng, N, cond) * 2.0
    W = np.diag(rng.uniform(0.5, 1.5, N) * wscale)
    return M, W


def dW_conserving(rng, Z, scale=0.1):
    """diagonal perturbation d with Z^T diag(d) Z = 0 (VOF mass of every column conserved)."""
    N = Z.shape[0]
    rows = []
    for j in range(4):
        for k in range(j, 4):
            rows.append(Z[:, j] * Z[:, k])
    ns = nullspace(np.array(rows))
    d = ns @ rng.standard_normal(ns.shape[1])
    d *= scale / np.max(np.abs(d))
    return d


def sym_conserving(rng, Z, scale=0.1):
    """symmetric S with Z^T S Z = 0."""
    N = Z.shape[0]
    S = sym(rng.standard_normal((N, N)))
    _, Pi = proj(Z)
    S = S - Pi @ S @ Pi
    return scale * S / np.max(np.abs(S))


def sym_kills_Z(rng, Z, scale=0.1):
    """symmetric S with S Z = 0."""
    N = Z.shape[0]
    S = sym(rng.standard_normal((N, N)))
    _, Pi = proj(Z)
    I = np.eye(N)
    S = (I - Pi) @ S @ (I - Pi)
    return scale * S / np.max(np.abs(S))


def transport_Ct(rng, Z, dWdiag):
    """Ct with Z^T Ct = 0 and Ct Z = -diag(dW) Z   (requires Z^T diag(dW) Z = 0)."""
    N = Z.shape[0]
    P, Pi = proj(Z)
    I = np.eye(N)
    K = rng.standard_normal((N, N))
    D = -(dWdiag[:, None] * Z)
    return (I - Pi) @ K @ (I - Pi) * 0.3 + D @ P.T


def theta_update(M, W, Mp, Wp, Ct, u, theta, extra_rhs=None):
    """D_theta for fixed dofs (no orphans):  M+ u+ = M+ u + W+ (uW - u) + (1-theta)(E u - E+ u)."""
    E, Ep = M - W, Mp - Wp
    uW = np.linalg.solve(Wp, W @ u - Ct @ u)
    rhs = Mp @ u + Wp @ (uW - u) + (1 - theta) * (E @ u - Ep @ u)
    if extra_rhs is not None:
        rhs = rhs + extra_rhs
    return np.linalg.solve(Mp, rhs), uW

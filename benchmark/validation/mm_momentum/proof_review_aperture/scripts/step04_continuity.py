"""Claim (a), second half: continuity of the metric at birth/death of a value, and 'O(M_ff) momentum/energy change'.
Label: prop:aperture(a), source aperture.tex ("so the metric restricted to the active values is continuous when a value is
born or dies ... change M-momentum and M-energy by O(M_ff)").
Method: port of the cell block (gram.py), cell blocks embedded in 4x4 (side index) matrices, compared across an event
(line offset -eps / +eps, or aperture b = 1e-9 vs 0).  Cell momentum of a phase = N^T B u = R^T u (outward fluxes u).
FINDING checks pass while the defect of the note is present.
Run: python3 proof_review/mimetic-momentum-transport-aperture/scripts/step04_continuity.py
"""
import math
import numpy as np
from kit import Checker, exit_code
from gram import cell_blocks, line_apertures, CORNER, NORMAL

chk = Checker("step04 continuity at birth/death")
rng = np.random.default_rng(21)
N = np.array(NORMAL)


def embed(res):
    (xs, ys, ex, ey, adm), bl = res
    out = {}
    for ph in (1, 2):
        m, cons, B, sides, lengths, mom, G = bl[ph]
        E = np.zeros((4, 4))
        for i, si in enumerate(sides):
            for j, sj in enumerate(sides):
                E[si, sj] = B[i, j]
        out[ph] = (E, sides, lengths, mom)
    return np.array([xs, ys]), out


def is_cut(a):
    return not (all(x == a[0] for x in a) and a[0] in (0.0, 1.0))


# ---- (1) exact straight line through a corner of the cell, cell stays cut on both sides of the event ----
ratios, nev = [], 0
for eps in (1e-6, 1e-9):
    worst, worst_dx = 0.0, 0.0
    for _ in range(3000):
        th = rng.uniform(0, 2 * math.pi)
        c = CORNER[rng.integers(4)]
        base = math.cos(th) * c[0] + math.sin(th) * c[1]
        a1, s1 = line_apertures(th, base - eps)
        a2, s2 = line_apertures(th, base + eps)
        if not (is_cut(a1) and is_cut(a2)):
            continue
        # skip lines almost parallel to a side (aperture changes by eps / sin, not small)
        if min(abs(math.cos(th)), abs(math.sin(th))) < 1e-2:
            continue
        x1, B1 = embed(cell_blocks(a1, s1))
        x2, B2 = embed(cell_blocks(a2, s2))
        dB = max(np.abs(B1[p][0] - B2[p][0]).max() for p in (1, 2))
        worst = max(worst, dB / eps)
        worst_dx = max(worst_dx, np.abs(x1 - x2).max() / eps)
    ratios.append((eps, worst, worst_dx))
    print(f"   straight line through a corner, cell cut on both sides, eps={eps:g}: max |dB|/eps = {worst:.1f}, max |dx*|/eps = {worst_dx:.1f}")
chk.true("straight-line data, line through a corner (cell stays cut): metric and pressure point Lipschitz in eps (|dB|/eps < 1e3)",
         all(r[1] < 1e3 and r[2] < 1e3 for r in ratios))

# ---- (2) cut -> uncut: a tiny wedge of the minority phase at a corner dies, the cell becomes uncut ----
jumps, xs_all = [], []
for _ in range(3000):
    th = rng.uniform(0, 2 * math.pi)
    k = rng.integers(4)
    c = CORNER[k]
    base = math.cos(th) * c[0] + math.sin(th) * c[1]
    a, st = line_apertures(th, base + 1e-9)       # liquid n.x < off: tiny liquid wedge at the corner iff c is the minimal corner
    if not is_cut(a) or 0.2 < sum(a) < 3.8:
        continue
    x, E = embed(cell_blocks(a, st))
    major = 2 if sum(a) < 2 else 1
    jumps.append(np.abs(E[major][0] - 0.5 * np.eye(4)).max())
    xs_all.append(np.abs(x).max())
jumps = np.array(jumps)
print(f"   cut->uncut (wedge of length ~1e-9): major-phase block vs uncut 1/2 I: median jump {np.median(jumps):.3f}, max {jumps.max():.3f}, "
      f"min {jumps.min():.3f} (n={len(jumps)}); |x*|max median {np.median(xs_all):.3f}")
chk.true("FINDING: at cut->uncut the block of the OTHER phase jumps by O(1) (median jump > 0.1) although the wedge values are O(1e-9)",
         np.median(jumps) > 0.1)

# ---- (3) the jump acts only through the net wetted flux  b^T u  (momentum and energy are x*-independent on b^T u = 0) ----
a = [1e-9, 0.0, 0.0, 1e-9]
st = [True, False, False, False]
x, E = embed(cell_blocks(a, st))
Emajor, sides, lengths, mom = E[2]
bvec = np.array([1 - a[k] for k in range(4)])
B_unc = 0.5 * np.eye(4)
print(f"   wedge a={a}: x* = {x}")
mom_err_free, en_err_free, mom_err_gen, en_err_gen = 0.0, 0.0, 0.0, 0.0
for _ in range(200):
    u = rng.standard_normal(4)
    u_free = u - bvec * (bvec @ u) / (bvec @ bvec)          # net wetted flux zero
    for uu, tag in ((u_free, "free"), (u, "gen")):
        P_cut = N.T @ Emajor @ uu
        P_unc = N.T @ B_unc @ uu
        dP = np.abs(P_cut - P_unc).max()
        dE = abs(uu @ Emajor @ uu - uu @ B_unc @ uu) / 2
        if tag == "free":
            mom_err_free, en_err_free = max(mom_err_free, dP), max(en_err_free, dE)
        else:
            mom_err_gen, en_err_gen = max(mom_err_gen, dP), max(en_err_gen, dE)
print(f"   cell momentum/energy jump at cut->uncut: net-flux-free u: {mom_err_free:.2e}/{en_err_free:.2e};  generic u: {mom_err_gen:.2e}/{en_err_gen:.2e}")
chk.true("momentum and energy jumps vanish (< 1e-6) for u with zero net wetted flux (b^T u = 0)", mom_err_free < 1e-6 and en_err_free < 1e-6)
chk.true("FINDING: for generic u the jump is O(1) (> 0.1): not 'O(M_ff)' with M_ff ~ 1e-10", mom_err_gen > 0.1 and en_err_gen > 0.1)

# ---- (4) non-closed (generic/averaged-trace) data: the crossing set changes at a birth and the pressure point jumps ----
jx, jb = [], []
for _ in range(3000):
    a = [rng.random() if rng.random() < 0.7 else float(rng.integers(0, 2)) for _ in range(4)]
    st = [bool(v) for v in rng.integers(0, 2, 4)]
    q = rng.integers(4)
    a_dead = list(a); a_dead[q] = 0.0
    a_born = list(a); a_born[q] = 1e-9
    if not (is_cut(a_dead) and is_cut(a_born)):
        continue
    x0, E0 = embed(cell_blocks(a_dead, st))
    x1, E1 = embed(cell_blocks(a_born, st))
    jx.append(np.abs(x0 - x1).max())
    jb.append(max(np.abs(E0[p][0] - E1[p][0]).max() for p in (1, 2)))
jx, jb = np.array(jx), np.array(jb)
print(f"   generic aperture data, side q: a_q = 0 vs 1e-9 (cell cut both ways): n={len(jx)}, median |dx*| = {np.median(jx):.3f}, "
      f"frac(|dx*|>0.05) = {(jx > 0.05).mean():.2f}, median max|dB| = {np.median(jb):.3f}")
chk.true("FINDING: generic data: birth of a 1e-9 crossing moves the pressure point by > 0.05 in more than 20% of the cells", (jx > 0.05).mean() > 0.2)

# ---- (5) the size of the change at a death vs M_ff:  entries of the dying row are O(b) = O(sqrt(M_ff)) when M_ff ~ b^2 ----
print("   dying value, straight line through a corner (cell stays cut): row/M_ff")
tab = []
for eps in (1e-3, 1e-5, 1e-7):
    vals = []
    for _ in range(3000):
        th = rng.uniform(0, 2 * math.pi)
        c = CORNER[rng.integers(4)]
        base = math.cos(th) * c[0] + math.sin(th) * c[1]
        if min(abs(math.cos(th)), abs(math.sin(th))) < 1e-2:
            continue
        for sign in (-1, 1):
            a, st = line_apertures(th, base + sign * eps)
            if not is_cut(a):
                continue
            x, E = embed(cell_blocks(a, st))
            for p in (1, 2):
                B, sides, lengths, mom = E[p]
                for i, k in enumerate(sides):
                    b = lengths[i]
                    if 0 < b < 5 * eps / min(abs(math.cos(th)), abs(math.sin(th))) and len(sides) >= 3:
                        off = np.abs(B[k]).sum() - B[k, k]
                        vals.append((b, B[k, k], off))
    vals = np.array(vals)
    r = vals[:, 2] / vals[:, 1]
    tab.append((eps, np.median(r), r.max(), np.median(vals[:, 1] / vals[:, 0] ** 2), np.median(vals[:, 2] / vals[:, 0])))
    print(f"     eps={eps:g}: n={len(vals)}, median offdiag/M_ff = {tab[-1][1]:.1f} (max {tab[-1][2]:.1f}), median M_ff/b^2 = {tab[-1][3]:.2f}, median offdiag/b = {tab[-1][4]:.3f}")
chk.true("dying row: off-diagonal/M_ff grows like 1/b (x30 from eps=1e-3 to 1e-5 at least) in cells with >= 3 wetted sides",
         tab[1][1] > 30 * tab[0][1] / 3 and tab[2][1] > 30 * tab[1][1] / 3)
chk.true("dying row: M_ff/b^2 stays O(1) (median < 50) there -> the event changes M by O(b)=O(sqrt(M_ff)), not O(M_ff)", all(t[3] < 50 for t in tab))
exit_code(chk)

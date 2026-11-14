"""Claim (a), first half: the rows of a Gram block of a value with wetted length b are O(b).
Label: prop:aperture(a), source aperture.tex (proof of (a), lines "The Gram block of a cell has rows ...").
Method: faithful port (gram.py) of _mimetic_gram_block! = R G^+ R^T + S incl. the slaved closing edge, pressure point
with the clamp margin 0.01, unit coordinates (h = 1, rho = 1).  Checks marked FINDING assert a defect and pass while it exists.
Run: python3 proof_review/mimetic-momentum-transport-aperture/scripts/step03_block_rows.py
"""
import math
import numpy as np
from kit import Checker, exit_code
from gram import cell_blocks, line_apertures, CORNER

chk = Checker("step03 block rows O(b)")
rng = np.random.default_rng(11)

# (1) single partial side: the note says 0.01 b rho h^2 (margin) -- exact, and S = 0.
for b in (1.0, 0.5, 1e-1, 1e-3, 1e-6, 1e-9):
    pt, bl = cell_blocks([b, 0, 0, 0], [True, False, False, False])
    m, cons, B, sides, lengths, mom, G = bl[1]
    chk.close(f"single side b={b:g}: block = 0.01 b", B[0, 0], 0.01 * b, tol=1e-9)

# (2) tiny liquid corner (sides 1 and 4 wetted with length b): rows O(b), diag ~ 0.0071 b.  The Gram tensor has the
# eigenvalue b^2/2 (direction (1,-1)); the code drops eigenvalues <= 1e-10 in G^+ (weight 0), i.e. for b <= 1.4e-5 the block
# is rank one, and the stabiliser S vanishes identically for <= 3 edges (the slaved closing edge cancels it).
rows = []
for b in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
    pt, bl = cell_blocks([b, 0, 0, b], [True, False, False, False])
    m, cons, B, sides, lengths, mom, G = bl[1]
    ev = np.linalg.eigvalsh(B)
    rows.append((b, np.abs(B).sum(axis=1).max() / b, B[0, 0] / b, ev[0] / b ** 2, ev[1] / b))
    print(f"   corner b={b:g}: row/b={rows[-1][1]:.4f} diag/b={rows[-1][2]:.4f} lam_min/b^2={rows[-1][3]:.4f} lam_max/b={rows[-1][4]:.4f}")
chk.true("tiny corner: row/b <= 0.0142 for all b (O(b) holds, constant 0.0141)", max(r[1] for r in rows) < 0.0142)
chk.true("tiny corner: lambda_min(B) = b^2/2 for b >= 1e-4 (near-kernel (1,-1))", all(abs(r[3] - 0.5) < 1e-6 for r in rows[:3]))
chk.true("FINDING tiny corner: B numerically singular (lam_min/lam_max < 1e-8) for b <= 1e-5 (pseudo-inverse cut 1e-10, S = 0)",
         all(r[3] * r[0] ** 2 / (r[4] * r[0]) < 1e-8 for r in rows[3:]))
pt, bl = cell_blocks([1e-3, 0, 0, 1e-3], [True, False, False, False])
m, cons, B, sides, lengths, mom, G = bl[1]
Rm = np.array(mom)
chk.close("corner b=1e-3: B = R G^+ R^T exactly, i.e. the stabiliser S (with the slaved closing edge) is identically 0",
          B, Rm @ np.linalg.pinv(G, rcond=1e-12) @ Rm.T, tol=1e-9)


def tiny_stats(sampler, nsamp, tiny_len):
    worst, wcfg, ratios = 0.0, None, []
    for _ in range(nsamp):
        a, st = sampler()
        if all(abs(x - a[0]) < 1e-15 for x in a) and a[0] in (0.0, 1.0):
            continue
        pt, bl = cell_blocks(a, st)
        for ph in (1, 2):
            m, cons, B, sides, lengths, mom, G = bl[ph]
            for i in range(m):
                if 0 < lengths[i] <= tiny_len:
                    r = np.abs(B[i]).sum() / lengths[i]
                    ratios.append(r)
                    if r > worst:
                        worst, wcfg = r, (list(a), list(st), ph, pt[:2])
    return worst, wcfg, np.array(ratios)


# (3) F1: exact straight-PLIC data (closed polygon), line within delta of a corner.
def sampler_line(delta):
    def f():
        th = rng.uniform(0, 2 * math.pi)
        c = CORNER[rng.integers(4)]
        off = math.cos(th) * c[0] + math.sin(th) * c[1] + rng.choice([-1, 1]) * delta * rng.uniform(0.5, 1)
        return line_apertures(th, off)
    return f


for delta in (1e-2, 1e-4, 1e-6, 1e-8):
    w, cfg, r = tiny_stats(sampler_line(delta), 6000, 0.05)
    print(f"   F1 straight line, delta={delta:g}: n={len(r)} sup row/b = {w:.6f}")
    chk.true(f"F1 exact line data, delta={delta:g}: sup row/b <= 1.0001 (row sum of |B_qr| / wetted length)", w <= 1.0001)


# (4) F2: averaged traces (aperture = mean of the traces of two slightly different lines), data not closed.
def sampler_avg(delta, dth, doff):
    def f():
        th = rng.uniform(0, 2 * math.pi)
        c = CORNER[rng.integers(4)]
        off = math.cos(th) * c[0] + math.sin(th) * c[1] + rng.choice([-1, 1]) * delta * rng.uniform(0.5, 1)
        a, st = line_apertures(th, off)
        out, sts = [], []
        for k in range(4):
            a2, st2 = line_apertures(th + rng.normal() * dth, off + rng.normal() * doff)
            if 0 < a[k] < 1 and 0 < a2[k] < 1 and st[k] != st2[k]:
                out.append(a[k]); sts.append(st[k]); continue
            s = st[k] if 0 < a[k] < 1 else (st2[k] if 0 < a2[k] < 1 else False)
            out.append(0.5 * (a[k] + a2[k])); sts.append(s)
        return out, sts
    return f


res = {}
for delta in (1e-3, 1e-6):
    w, cfg, r = tiny_stats(sampler_avg(delta, 0.3, 0.1), 8000, 4 * delta)
    res[delta] = (w, cfg, r)
    print(f"   F2 averaged traces (0.3 rad, 0.1 h), delta={delta:g}: n={len(r)} sup row/b = {w:.1f}, frac(row/b>5) = {(r > 5).mean():.4f}")
    print(f"      worst: a={cfg[0]} starts={cfg[1]} phase={cfg[2]} x*={cfg[3]}")
chk.true("FINDING F2: averaged-trace data give rows >> b (sup row/b > 100 at delta=1e-6)", res[1e-6][0] > 100)

# (5) F3: explicit counterexample with O(1) entry although b = 1e-8 (not closed data, nearly singular G).
a3 = [0.28247641920331645, 0.0, 0.31249918317583625, 1e-8]
st3 = [True, True, True, False]
pt, bl = cell_blocks(a3, st3)
m, cons, B, sides, lengths, mom, G = bl[1]
i = sides.index(3)
print(f"   F3: a={a3}, x*={pt[:2]}, consistent={cons}, G eigs={np.linalg.eigvalsh(0.5*(G+G.T))}")
print(f"       row of the value with b=1e-8: {B[i]}  (entries/b: {B[i]/1e-8})")
chk.true("FINDING F3: counterexample, the row of a value of length 1e-8 has an O(1) entry (>1e-2)", np.abs(B[i]).max() > 1e-2)
chk.true("F3: block is flagged consistent (not caught by the least-squares floor)", cons)

# (6) b-scaling of the diagonal for exact data: M_ff between 0.2 b^2 and O(b) -> the note's 'O(b)' is an upper bound only
dmin, dmax = 1e9, 0.0
for delta in (1e-3, 1e-6):
    for _ in range(4000):
        a, st = sampler_line(delta)()
        if all(abs(x - a[0]) < 1e-15 for x in a):
            continue
        pt, bl = cell_blocks(a, st)
        for ph in (1, 2):
            m, cons, B, sides, lengths, mom, G = bl[ph]
            for i in range(m):
                if 0 < lengths[i] < 1e-2:
                    dmin = min(dmin, B[i, i] / lengths[i] ** 2)
                    dmax = max(dmax, B[i, i] / lengths[i])
print(f"   exact data: min diag/b^2 = {dmin:.3f}, max diag/b = {dmax:.3f}")
chk.true("exact data: diag >= 0.19 b^2 always and diag <= 1.0 b", dmin > 0.19 and dmax <= 1.0001)
exit_code(chk)

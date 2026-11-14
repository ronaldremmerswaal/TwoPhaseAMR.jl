"""Claim (b): lambda_max(W, M) >= m_f / M_ff (and the sharper m_f (M^-1)_ff); 'with the merge lambda_max(W~, M) is bounded'.
Label: prop:aperture(b), source aperture.tex (proposition item (b) and its proof).
Method: (1) random SPD M: the two lower bounds and their equality cases; (2) single-block model (tiny liquid corner);
(3) lattice model (lattice.py: disk interface near a lattice node, ported Gram blocks, aperture activity, small ties OFF,
as the proposition assumes; and ON for contrast), kappa = 1.  FINDING checks pass while the defect of the note is present.
Run: python3 proof_review/mimetic-momentum-transport-aperture/scripts/step05_lambda_bounds.py
"""
import math
import numpy as np
import scipy.linalg as sl
from kit import Checker, exit_code, spd
from gram import cell_blocks
from lattice import Lattice

chk = Checker("step05 lambda_max bounds")
rng = np.random.default_rng(31)


def lam_max(Wd, M):
    return sl.eigh(np.diag(Wd), M, eigvals_only=True)[-1]


# ---- (1) algebra: lambda_max(W,M) >= max_f m_f (M^-1)_ff >= max_f m_f / M_ff ----
ok1 = ok2 = True
for trial in range(300):
    n = int(rng.integers(2, 9))
    M = spd(rng, n, cond=10 ** rng.uniform(0, 6)) * 10 ** rng.uniform(-3, 1)
    m = 10 ** rng.uniform(-3, 1, n)
    lam = lam_max(m, M)
    Mi = np.linalg.inv(M)
    lb1 = np.max(m * np.diag(Mi))
    lb0 = np.max(m / np.diag(M))
    ok1 &= lam >= lb1 * (1 - 1e-9)
    ok2 &= lb1 >= lb0 * (1 - 1e-12)
chk.true("random SPD (300): lambda_max(W,M) >= max m_f (M^-1)_ff", ok1)
chk.true("random SPD (300): max m_f (M^-1)_ff >= max m_f / M_ff", ok2)
# equality cases: M diagonal gives equality in both; the first bound is attained iff e_f is a generalized top eigenvector
D = np.diag(rng.uniform(0.5, 2, 5)); m = rng.uniform(0.5, 2, 5)
chk.close("M diagonal: lambda_max = max m_f/M_ff", lam_max(m, D), np.max(m / np.diag(D)), tol=1e-12)
# the gap (M^-1)_ff vs 1/M_ff is unbounded: M = [[1, 1-e],[1-e, 1]]
for e in (1e-2, 1e-4):
    M = np.array([[1, 1 - e], [1 - e, 1]]); m = np.array([1.0, 1.0])
    lam = lam_max(m, M)
    print(f"   M=[[1,1-e],[1-e,1]], e={e:g}: lambda_max(I,M) = {lam:.3e}, m(M^-1)_ff = {1/(1-(1-e)**2):.3e}, m/M_ff = 1")
    sharp = float(np.max(m * np.diag(np.linalg.inv(M))))
    chk.true(f"e={e:g}: m (M^-1)_ff = {sharp:.3e} <= lambda_max = {lam:.3e} <= n * m (M^-1)_ff (n = 2; diagonal entry vs trace)", sharp <= lam * (1 + 1e-12) and lam <= 2 * sharp * (1 + 1e-9))
    chk.true(f"FINDING (e={e:g}): m/M_ff = 1 but lambda_max(W,M) = {lam:.1e}: with W~ <= diag(M) boundedness does not follow from the cap",
             lam > 1 / (2 * e) * 0.9)

# ---- (2) single block, tiny corner: diag cap does not bound lambda_max ----
print("   isolated corner block (liquid wedge, m_f = 0.5 per value, W~ = diag(min(m, B_ff)) = diag(B)):")
tab = []
for b in (1e-2, 1e-3, 1e-4):
    pt, bl = cell_blocks([b, 0, 0, b], [True, False, False, False])
    B = bl[1][2]
    Wt = np.minimum(0.5, np.diag(B))
    lt, l0 = lam_max(Wt, B), lam_max(np.array([0.5, 0.5]), B)
    tab.append((b, lt, l0))
    print(f"     b={b:g}: B_ff={B[0,0]:.3e}, lambda_max(W~,B) = {lt:.3e} (x b = {lt*b:.3f}), lambda_max(W,B) = {l0:.3e}, m/B_ff = {0.5/B[0,0]:.3e}")
chk.true("isolated corner block: lambda_max(W~,B) * b is constant (0.01..0.05): grows like 1/b although W~ <= diag(B)",
         all(0.01 < t[1] * t[0] < 0.05 for t in tab))

# ---- (3) lattice with a disk interface passing a lattice node ----
node = (5.0, 5.0)


def run(R, phi, delta, kappa=1.0, ties=False):
    c = (node[0] + (R + delta) * math.cos(phi), node[1] + (R + delta) * math.sin(phi))
    L = Lattice(12, c[0], c[1], R, quad=400)
    M, W, Wt, info = L.assemble(kappa, small_ties=ties)
    tot = sum(info['mass'].values())
    dn = info['donors']
    nd = [d for d in range(len(W)) if d not in dn]
    cap = max(Wt[d] / M[d, d] for d in nd)
    return dict(lam=lam_max(W, M), lamt=lam_max(Wt, M), mf=np.max(W / np.diag(M)), cap=cap, tot=tot - Wt.sum(),
                minev=np.linalg.eigvalsh(M)[0], n=len(W), ties=info['ties'])


rows = {}
for tag, ties in (("ties off", False), ("ties on", True)):
    print(f"   lattice, disk R=2.4 near node (5,5), phi=0.6, delta<0 = node inside the disk (tiny wedge of the minority phase), {tag}:")
    for delta in (-1e-2, -1e-3, -1e-4, -1e-6, -1e-8):
        r = run(2.4, 0.6, delta, ties=ties)
        rows[(ties, delta)] = r
        print(f"     delta={delta:g}: lambda_max(W,M)={r['lam']:.3e}  lambda_max(W~,M)={r['lamt']:.3e}  max m/M_ff={r['mf']:.3e}  "
              f"max W~/M_ff (non-donors)={r['cap']:.4f}  mass defect={r['tot']:.1e}  min eig M={r['minev']:.2e}")
for delta in (-1e-2, -1e-4, -1e-8):
    r = rows[(False, delta)]
    chk.true(f"ties off, delta={delta:g}: lambda_max(W,M) >= max m_f/M_ff (claim (b), first half)", r['lam'] >= r['mf'] * (1 - 1e-9))
    chk.true(f"ties off, delta={delta:g}: non-donor W~_f <= kappa M_ff (claim (2)) and Z^T W~ Z = total mass", r['cap'] <= 1 + 1e-12 and abs(r['tot']) < 1e-12)
chk.true("ties off: lambda_max(W,M) exceeds 1e14 at delta=-1e-8 (m_f/M_ff -> inf)", rows[(False, -1e-8)]['lam'] > 1e14)
chk.true("FINDING: with the merge, ties off, lambda_max(W~,M) is NOT bounded: 2.2 at -1e-2, > 1e5 at -1e-8 (scales like 1/b)",
         rows[(False, -1e-2)]['lamt'] < 3 and rows[(False, -1e-8)]['lamt'] > 1e5
         and rows[(False, -1e-8)]['lamt'] / rows[(False, -1e-6)]['lamt'] > 30)
chk.true("ties on: lambda_max(W,M) and lambda_max(W~,M) stay below 3 for all delta (bounded, the measured 2.7)",
         all(rows[(True, d)]['lam'] < 3 and rows[(True, d)]['lamt'] < 3 for d in (-1e-2, -1e-3, -1e-4, -1e-6, -1e-8)))

# robustness: other radii / angles
worst_lamt, worst_ratio = [], []
for R, phi in ((2.4, 0.6), (2.55, 0.9), (2.4, 2.2), (2.45, 3.7), (2.35, 5.1)):
    try:
        a = run(R, phi, -1e-6)
        b2 = run(R, phi, -1e-3)
    except AssertionError:
        continue
    worst_lamt.append(a['lamt']); worst_ratio.append(a['lamt'] / b2['lamt'])
    print(f"   R={R}, phi={phi}: lambda_max(W~,M) = {b2['lamt']:.2f} (delta=-1e-3), {a['lamt']:.2e} (delta=-1e-6)")
chk.true("all tested disks: lambda_max(W~,M) grows by > 100 from delta=-1e-3 to -1e-6 (ties off)", len(worst_lamt) >= 3 and min(worst_ratio) > 100)
# lump threshold: both eigenvalues of G (<= 0.014 b) fall below the pseudo-inverse cut 1e-10 at b ~ 7e-9: block = 0, diagonal 0,
# the value is lumped (M_ff += m_f): W~_f jumps from ~0 to m_f and lambda_max(W~,M) drops back to O(1)
r8 = run(2.4, 0.6, -1e-8)
r3 = run(2.4, 0.6, -3e-9)
print(f"   near the lump threshold: lambda_max(W~,M) = {r8['lamt']:.2e} (delta=-1e-8), {r3['lamt']:.2e} (delta=-3e-9)")
chk.true("FINDING: lambda_max(W~,M) ~ 3.5e-3/b peaks at 3.5e5 just above the lump threshold b ~ 7e-9, then the lump resets it to O(1)",
         r8['lamt'] > 1e5 and r3['lamt'] < 10)
# ---- lattice: the metric on the common dofs jumps by O(1) when the two wedge values are born (cut -> uncut of the other phase) ----
def build(R, phi, delta):
    c = (node[0] + (R + delta) * math.cos(phi), node[1] + (R + delta) * math.sin(phi))
    return Lattice(12, c[0], c[1], R, quad=300).assemble()


for R, phi in ((2.4, 0.6), (2.55, 0.9), (2.4, 2.2), (2.45, 3.7)):
    M1, W1, Wt1, i1 = build(R, phi, -1e-9)
    M2, W2, Wt2, i2 = build(R, phi, +1e-9)
    d1, d2 = i1['dof'], i2['dof']
    common = [f for f in d1 if f in d2]
    a = np.array([d1[f] for f in common]); b = np.array([d2[f] for f in common])
    jump = np.abs(M1[np.ix_(a, a)] - M2[np.ix_(b, b)]).max()
    born = [f for f in d1 if f not in d2] + [f for f in d2 if f not in d1]
    print(f"   lattice R={R}, phi={phi}: values born/dead = {len(born)} (aperture ~1e-9), max |dM| on the common dofs = {jump:.3f}")
    chk.true(f"FINDING lattice R={R}, phi={phi}: M on the common dofs jumps by > 1 at the birth of two values of length 1e-9", jump > 1.0 and len(born) == 2)
# W~ at the lump threshold: the born value has W~ ~ 1e-10 at delta = -1e-8 and the full m_f = 0.248 at delta = -3e-9
M8, W8, Wt8, i8 = build(2.4, 0.6, -1e-8)
M3, W3, Wt3, i3 = build(2.4, 0.6, -3e-9)
f = ('x', 5, 4)
print(f"   W~ of the born value ({f}): {Wt8[i8['dof'][f]]:.2e} at delta=-1e-8, {Wt3[i3['dof'][f]]:.3f} at delta=-3e-9 (lumped, M_ff = m_f)")
chk.true("FINDING: W~ of the newborn value jumps from ~1e-10 to m_f = 0.248 at the lump threshold (the note lists the lump threshold among the remaining jumps)",
         Wt8[i8['dof'][f]] < 1e-9 and abs(Wt3[i3['dof'][f]] - 0.248) < 1e-3)
exit_code(chk)

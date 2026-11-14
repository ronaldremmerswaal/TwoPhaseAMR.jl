"""Formulation claim (2): the merge (keep k_f = min(m_f, kappa M_ff), excess to a same-axis same-phase donor) preserves
Z^T W~ Z = Z^T W Z and gives W~_f <= kappa M_ff for non-donors.
Label: prop:aperture, 'Formulation' (last paragraph), code _mimetic_merged_masses (mimetic_transport.jl:211).
Method: abstract port of the loop of _mimetic_merged_masses (class = axis x phase; donor = another dof of the same class, or none),
random masses and metric diagonals; then the failure cases read off the code: (i) no donor and no dof: mass dropped;
(ii) no donor but a dof: the excess stays, W~_f = m_f > kappa M_ff; (iii) identified donor: ones in two columns; (iv) donors exceed kappa M_ff.
The lattice model (step05) checks the same identities on assembled blocks.
Run: python3 proof_review/mimetic-momentum-transport-aperture/scripts/step02_merge.py
"""
import numpy as np
from kit import Checker, exit_code

chk = Checker("step02 merge")
rng = np.random.default_rng(51)


def merge(m, cls, dof, Md, donor, kappa=1.0):
    """m[f] face masses, cls[f] class, dof[f] = index or -1, Md[dof] diagonal of M, donor[f] = dof index or -1.
    Mirrors the Julia loop: keep = min(m, kappa Md) (0 without dof); excess -> donor if exists else own dof else dropped."""
    N = int(max(dof)) + 1
    Wt = np.zeros(N)
    dropped = 0.0
    for f in range(len(m)):
        d = dof[f]
        keep = min(m[f], kappa * Md[d]) if d >= 0 else 0.0
        ex = m[f] - keep
        if d >= 0:
            Wt[d] += keep
        if ex <= 0:
            continue
        tgt = donor[f] if donor[f] >= 0 else d
        if tgt >= 0:
            Wt[tgt] += ex
        else:
            dropped += ex
    return Wt, dropped


def setup(nf=14, ncls=4, p_dof=0.6, ident=False):
    cls = rng.integers(0, ncls, nf)
    m = rng.uniform(0.1, 2.0, nf)
    dof = -np.ones(nf, int)
    k = 0
    for f in range(nf):
        if rng.random() < p_dof:
            dof[f] = k; k += 1
    Md = 10 ** rng.uniform(-3, 0.3, k)
    donor = -np.ones(nf, int)
    for f in range(nf):
        cand = [dof[g] for g in range(nf) if g != f and cls[g] == cls[f] and dof[g] >= 0]
        if cand:
            donor[f] = int(rng.choice(cand))
    return cls, m, dof, Md, donor


for trial in range(200):
    cls, m, dof, Md, donor = setup()
    if (dof >= 0).sum() < 2:
        continue
    # keep only configurations where every face with excess has a donor (the claimed regime)
    need = [f for f in range(len(m)) if m[f] > (min(m[f], Md[dof[f]]) if dof[f] >= 0 else 0.0)]
    if any(donor[f] < 0 for f in need):
        continue
    Wt, dropped = merge(m, cls, dof, Md, donor)
    # column sums of Z^T W~ Z per class = class mass
    cls_dof = np.zeros(len(Wt), int)
    for f in range(len(m)):
        if dof[f] >= 0:
            cls_dof[dof[f]] = cls[f]
    for c in range(4):
        chk.close(f"[{trial}] class {c}: sum W~ = sum m", Wt[cls_dof == c].sum(), m[cls == c].sum(), tol=1e-12)
    donors = {donor[f] for f in need}
    nd = [d for d in range(len(Wt)) if d not in donors]
    chk.true(f"[{trial}] W~_f <= M_ff for non-donors", all(Wt[d] <= Md[d] * (1 + 1e-14) for d in nd))
    assert dropped == 0.0

# --- continuity of W~ in b -> 0 for fixed masses and a fixed donor (value dies when M_ff = b -> 0) ---
m = np.array([0.7, 1.0, 0.9]); cls = np.array([0, 0, 0]); donor = np.array([1, -1, 1])
errs = []
for b in (1e-2, 1e-4, 1e-6):
    dof = np.array([0, 1, 2]); Md = np.array([b, 1.0, 5.0])
    Wt_b, _ = merge(m, cls, dof, Md, donor)
    dof0 = np.array([-1, 0, 1]); Md0 = np.array([1.0, 5.0])
    donor0 = np.array([0, -1, 0])
    Wt_0, _ = merge(m, cls, dof0, Md0, donor0)
    # compare on the two surviving values (dof 1,2 vs 0,1)
    errs.append(np.abs(Wt_b[1:] - Wt_0).max())
    print(f"   b={b:g}: |W~(b) - W~(0)| on surviving values = {errs[-1]:.2e}, W~_f(b) = {Wt_b[0]:.2e}")
chk.true("W~ is continuous at a death with fixed donor: error <= b", all(e <= b * 1.0001 for e, b in zip(errs, (1e-2, 1e-4, 1e-6))))

# --- FINDING (i)-(iv) failure cases ---
# (i) no dof and no donor: mass dropped, Z^T W~ Z < Z^T W Z
m = np.array([1.0, 0.5]); cls = np.array([0, 1]); dof = np.array([0, -1]); Md = np.array([2.0]); donor = np.array([-1, -1])
Wt, dropped = merge(m, cls, dof, Md, donor)
chk.true("FINDING (i): value without dof and without donor: its mass is dropped (code: `dropped += excess`)", dropped == 0.5 and Wt.sum() == 1.0)
# (ii) dof but no donor: W~_f = m_f > kappa M_ff although not a donor
m = np.array([1.0]); cls = np.array([0]); dof = np.array([0]); Md = np.array([0.01]); donor = np.array([-1])
Wt, dropped = merge(m, cls, dof, Md, donor)
chk.true("FINDING (ii): a value with dof but no donor keeps all of m_f: W~_f = 1.0 > kappa M_ff = 0.01", Wt[0] == 1.0 > Md[0])
# (iii) identified donor: face 0 = liquid value, face 1 = identified value (mass 0.6 liquid + 0.3 gas, no cap: code `continue`)
Zrow = np.array([[1, 0], [1, 1]], float)          # columns: liquid x, gas x
m = np.array([1.0, 0.9]); dof = np.array([0, 1]); Md = np.array([0.1, 1.0]); donor = np.array([1, -1])
Wu = m.copy()                                      # unmerged W: dof masses
# merge: only face 0 is capped (identified face skipped in the loop: its mass is added as is)
keep0 = min(m[0], Md[0]); Wt = np.array([keep0, m[1] + (m[0] - keep0)])
ZWZ_u, ZWZ_m = Zrow.T @ np.diag(Wu) @ Zrow, Zrow.T @ np.diag(Wt) @ Zrow
print(f"   identified donor: Z^T W Z = {ZWZ_u.tolist()}, Z^T W~ Z = {ZWZ_m.tolist()}")
chk.true("FINDING (iii): excess of a liquid value merged into an identified value: liquid column unchanged, gas column grows by the excess (0.9)",
         abs(ZWZ_m[0, 0] - ZWZ_u[0, 0]) < 1e-15 and abs(ZWZ_m[1, 1] - ZWZ_u[1, 1] - 0.9) < 1e-15)
# (iv) donors exceed kappa M_ff
cls, m, dof, Md, donor = np.array([0]*3), np.array([2.0, 2.0, 0.1]), np.array([0, 1, 2]), np.array([0.1, 0.1, 5.0]), np.array([2, 2, -1])
Wt, _ = merge(m, cls, dof, Md, donor)
chk.true("(iv) donors exceed kappa M_ff: W~ = (0.1, 0.1, 0.1+1.9+1.9) vs M_dd = 5 (claim only for non-donors, as stated)", abs(Wt[2] - 3.9) < 1e-12 and Wt[2] < Md[2])
cls, m, dof, Md, donor = np.array([0]*3), np.array([2.0, 2.0, 0.1]), np.array([0, 1, 2]), np.array([0.1, 0.1, 1.0]), np.array([2, 2, -1])
Wt, _ = merge(m, cls, dof, Md, donor)
chk.true("(iv') donor can exceed its own cap: W~_donor = 3.9 > M_dd = 1.0 (so lambda_max(W~,M) can exceed 1 by donors too)", Wt[2] > Md[2])
exit_code(chk)

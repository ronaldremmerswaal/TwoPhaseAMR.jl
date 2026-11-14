"""Claim (c): prop:obstruction, prop:flux, prop:theta, prop:redistribution carry over with W~ and the merged flux covector.
Label: prop:aperture(c) (proof: "use only Z^T C_W = 0, dt C_W(Z) = -Delta W Z and the symmetry of M; the merge keeps the first two").
Model (abstract, NumPy): faces F (two classes = columns of Z), dofs D subset of F, face-level W transport
  m+ u_W = m u_F - dt C u_F,  Z_F^T C = 0,  dt C Z_F = -(m+ - m) Z_F,  C = -mdot/2 + A (A skew),
merge fractions S[d,f] (column-stochastic, class-preserving) as in _mimetic_merged_masses:  W~ = S m;  covector merged by S;
update as in _mimetic_transport_after!  M+ u+ = M+ u + S+ m+ (u_W - u_F) + (1-theta)(E~ u - E~+ u),  E~ = M - W~.
Derived (and checked here):  Delta P = theta Z^T (E~+ - E~) u + mu+(u) - mu(u),  mu(v) = Z^T W~ v - Z_F^T m v_F  (merge momentum defect).
Run: python3 proof_review/mimetic-momentum-transport-aperture/scripts/step06_identities_merge.py
"""
import numpy as np
from kit import Checker, exit_code, spd

chk = Checker("step06 identities with the merge")
rng = np.random.default_rng(61)


def make_model(nf=12, nd=8, sat=True):
    cls = np.array([0, 1] * (nf // 2))
    rng.shuffle(cls)
    dofs = np.sort(rng.choice(nf, nd, replace=False))
    # ensure both classes have >= 3 dofs
    while min((cls[dofs] == 0).sum(), (cls[dofs] == 1).sum()) < 3:
        dofs = np.sort(rng.choice(nf, nd, replace=False))
    dof_of = -np.ones(nf, int)
    dof_of[dofs] = np.arange(nd)
    ZF = np.zeros((nf, 2)); ZF[np.arange(nf), cls] = 1
    Z = ZF[dofs]
    donor = -np.ones(nf, int)
    for f in range(nf):
        cand = [dof_of[g] for g in dofs if g != f and cls[g] == cls[f]]
        donor[f] = int(rng.choice(cand))
    return dict(nf=nf, nd=nd, cls=cls, dofs=dofs, dof_of=dof_of, ZF=ZF, Z=Z, donor=donor)


def fractions(mod, m, Md, kappa=1.0):
    S = np.zeros((mod['nd'], mod['nf']))
    for f in range(mod['nf']):
        d = mod['dof_of'][f]
        keep = min(m[f], kappa * Md[d]) if d >= 0 else 0.0
        if d >= 0:
            S[d, f] += keep / m[f]
        S[mod['donor'][f], f] += (m[f] - keep) / m[f]
    return S


def conserving_pert(mod, scale):
    d = rng.standard_normal(mod['nf'])
    for c in range(2):
        d[mod['cls'] == c] -= d[mod['cls'] == c].mean()
    return scale * d / np.abs(d).max()


def build_C(mod, mdot):
    ZF = mod['ZF']
    P = ZF @ np.linalg.inv(ZF.T @ ZF)
    D = -0.5 * (mdot[:, None] * ZF)
    A = D @ P.T - P @ D.T
    K = rng.standard_normal((mod['nf'],) * 2); K = K - K.T
    Pi = P @ ZF.T
    I = np.eye(mod['nf'])
    A = A + 0.3 * (I - Pi) @ K @ (I - Pi)
    return -np.diag(mdot) / 2 + A


def setup(dt, t_branch=True):
    mod = make_model()
    nf, nd = mod['nf'], mod['nd']
    m0 = rng.uniform(0.8, 1.6, nf)
    m1 = conserving_pert(mod, 0.5)                      # mdot, class sums zero
    B = spd(rng, nd, 4.0)
    M0 = B * 1.0
    # make some dofs saturated (kappa M_dd < m) and some not, robustly away from the switch
    scale = np.where(rng.random(nd) < 0.5, 0.35, 3.0)
    M0 = M0 * np.sqrt(np.outer(scale, scale))
    M1 = rng.standard_normal((nd, nd)); M1 = 0.3 * (M1 + M1.T) / 2
    return mod, m0, m1, M0, M1


def step(mod, m0, m1, M0, M1, dt, u, uext, theta, C, kappa=1.0, frozen_merge=False):
    nf = mod['nf']
    m, mp = m0, m0 + dt * m1
    M, Mp = M0, M0 + dt * M1
    S, Sp = fractions(mod, m, np.diag(M), kappa), fractions(mod, mp, np.diag(Mp), kappa)
    if frozen_merge:
        Sp = S
    Wt, Wtp = S @ m, Sp @ mp
    uF = uext.copy(); uF[mod['dofs']] = u
    uW = (m * uF - dt * (C @ uF)) / mp
    Et, Etp = M - np.diag(Wt), Mp - np.diag(Wtp)
    rhs = Mp @ u + Sp @ (mp * (uW - uF)) + (1 - theta) * (Et @ u - Etp @ u)
    up = np.linalg.solve(Mp, rhs)
    return dict(up=up, S=S, Sp=Sp, Wt=Wt, Wtp=Wtp, Et=Et, Etp=Etp, uF=uF, uW=uW, m=m, mp=mp, M=M, Mp=Mp)


def mu(mod, S, m, uF):
    # mu(v) = Z^T W~ v - Z_F^T m v_F  (vector of 2 columns), W~ = S m acting on dof values
    Wt = S @ m
    return mod['Z'].T @ (Wt * uF[mod['dofs']]) - mod['ZF'].T @ (m * uF)


# ================= (a) constants, (b) momentum, W~ conservation =================
dt = 0.07
maxdiff_const = maxdiff_mom = maxdiff_claim = 0.0
for trial in range(40):
    mod, m0, m1, M0, M1 = setup(dt)
    C = build_C(mod, m1)
    ZF, Z = mod['ZF'], mod['Z']
    chk_id1 = np.abs(ZF.T @ C).max()
    chk_id2 = np.abs(dt * C @ ZF + (m1 * dt)[:, None] * ZF).max()
    assert chk_id1 < 1e-12 and chk_id2 < 1e-12
    uext = rng.standard_normal(mod['nf'])
    u = rng.standard_normal(mod['nd'])
    for theta in (0.0, 0.5, 1.0):
        r = step(mod, m0, m1, M0, M1, dt, u, uext, theta, C)
        # Z^T W~ Z = Z^T W Z (class totals)
        assert np.allclose(Z.T @ (r['Wt'][:, None] * Z), ZF.T @ (r['m'][:, None] * ZF))
        # (a) constants: u = Z U everywhere (also the extension)
        U = rng.standard_normal(2)
        r2 = step(mod, m0, m1, M0, M1, dt, Z @ U, ZF @ U, theta, C)
        lhs = r2['Mp'] @ (r2['up'] - Z @ U)
        rhs = -(1 - theta) * (r2['Etp'] - r2['Et']) @ (Z @ U)
        maxdiff_const = max(maxdiff_const, np.abs(lhs - rhs).max())
        # (b) momentum
        dP = Z.T @ (r['Mp'] @ r['up']) - Z.T @ (r['M'] @ u)
        pred = theta * Z.T @ ((r['Etp'] - r['Et']) @ u) + mu(mod, r['Sp'], r['mp'], r['uF']) - mu(mod, r['S'], r['m'], r['uF'])
        claimed = theta * Z.T @ ((r['Etp'] - r['Et']) @ u)
        maxdiff_mom = max(maxdiff_mom, np.abs(dP - pred).max())
        maxdiff_claim = max(maxdiff_claim, np.abs(dP - claimed).max())
chk.true(f"(a) constants: M+(u+ - ZU) = -(1-theta)(E~+ - E~) Z U with the merge, exactly (max err {maxdiff_const:.1e})", maxdiff_const < 1e-11)
chk.true(f"(b) momentum with the merge: DeltaP = theta Z^T DeltaE~ u + mu+(u) - mu(u), exactly (max err {maxdiff_mom:.1e})", maxdiff_mom < 1e-11)
chk.true(f"FINDING (b): the note's/prop:theta(b) value theta Z^T DeltaE~ u misses the merge defect: max |DeltaP - claimed| = {maxdiff_claim:.2e}", maxdiff_claim > 1e-3)

# when the extension equals the donor's velocity (u_f = u_d for every merged face) the defect vanishes
mod, m0, m1, M0, M1 = setup(dt)
C = build_C(mod, m1)
u = rng.standard_normal(mod['nd'])
S = fractions(mod, m0, np.diag(M0))
uext = np.zeros(mod['nf'])
for f in range(mod['nf']):
    uext[f] = u[mod['donor'][f]]
uF = uext.copy(); uF[mod['dofs']] = u
# for dof faces u_f is the dof's own value: mu = sum_f excess_f (u_d(f) - u_f) Z_f vanishes only if own values are also equal -> use constants per donor class
r = step(mod, m0, m1, M0, M1, dt, u, uext, 1.0, C)
print(f"   |mu| for extension = donor value but own dof values distinct: {np.abs(mu(mod, r['S'], r['m'], r['uF'])).max():.2e} (excess of dof values also matters)")
# with every merged face being a non-dof face (all excess comes from faces without dof) and extension = donor value mu = 0
mod2 = make_model(nf=12, nd=8)
# build masses with dof faces unsaturated: M_dd huge -> keep = m
m = rng.uniform(0.8, 1.6, 12); Md = 1e3 * np.ones(8)
S2 = fractions(mod2, m, Md)
u2 = rng.standard_normal(8)
uext2 = np.array([u2[mod2['donor'][f]] for f in range(12)])
uF2 = uext2.copy(); uF2[mod2['dofs']] = u2
chk.true("mu(u) = 0 if dof faces keep their whole mass and the extension of the dof-less faces equals the donor value",
         np.abs(mu(mod2, S2, m, uF2)).max() < 1e-12)

# ================= which W-transport identities does the merge keep? =================
errA, errB, errC, errD = 0.0, 0.0, 0.0, 0.0
errE = errF = 0.0
for trial in range(30):
    mod, m0, m1, M0, M1 = setup(dt)
    C = build_C(mod, m1)
    ZF, Z = mod['ZF'], mod['Z']
    m, mp = m0, m0 + dt * m1
    S, Sp = fractions(mod, m, np.diag(M0), 1.0), fractions(mod, mp, np.diag(M0 + dt * M1), 1.0)
    Wt, Wtp = S @ m, Sp @ mp
    # (A) merged flux covector Ct = S+ C : first identity holds, second does not
    Ct = Sp @ C
    errA = max(errA, np.abs(Z.T @ Ct).max())                                   # = 0
    err_second = np.abs(dt * Ct @ ZF + (Wtp - Wt)[:, None] * Z).max()           # != 0
    errB = max(errB, err_second)
    # the defect is exactly the reassignment (S+ - S) m Z
    errC = max(errC, np.abs((dt * Ct @ ZF + (Wtp - Wt)[:, None] * Z) - ((Sp @ m - Wt)[:, None] * Z)).max())
    # (B) the flux that the code's update implies for theta = 0: L(u) = -[(W~+ - W~) u + S+ m+ (u_W - u_F)]:  second identity holds, first not
    # L is linear in the face vector u_F (dof values and extension)
    N = mod['nd']
    LF = np.zeros((N, mod['nf']))
    for g in range(mod['nf']):
        uF = np.zeros(mod['nf']); uF[g] = 1
        e = uF[mod['dofs']]
        uW = (m * uF - dt * (C @ uF)) / mp
        LF[:, g] = -((Wtp - Wt) * e + Sp @ (mp * (uW - uF)))
    errD = max(errD, np.abs(LF @ ZF + (Wtp - Wt)[:, None] * Z).max())           # second identity: = 0
    errE = max(errE, np.abs(Z.T @ LF).max())                                    # first identity: != 0
    # (C) both hold for the merged-parcel flux with the reassignment term: Ct2 = S+ C - (S+ - S) m (.)/dt  acting on face values
    R = (Sp - S) @ np.diag(m)
    Ct2 = Sp @ C - R / dt
    errF = max(errF, np.abs(Z.T @ Ct2).max(), np.abs(dt * Ct2 @ ZF + (Wtp - Wt)[:, None] * Z).max())
chk.true(f"merged covector S+ C: Z^T Ct = 0 holds (err {errA:.1e})", errA < 1e-12)
chk.true(f"FINDING: merged covector S+ C: dt Ct(Z) = -(W~+ - W~) Z FAILS, defect {errB:.2e}, equal to the reassignment (S+ - S) m Z (err {errC:.1e})", errB > 1e-3 and errC < 1e-12)
chk.true(f"update-implied flux (code, theta = 0): dt L(Z) = -(W~+ - W~) Z holds (err {errD:.1e})", errD < 1e-12)
chk.true(f"FINDING: update-implied flux: Z^T L != 0 (max |Z^T L| = {errE:.2e}): conservation identity FAILS", errE > 1e-3)
chk.true(f"both identities hold for the covector with the reassignment flux S+ C - (S+ - S) m (.)/dt (err {errF:.1e}); it acts on face values u_F, i.e. on merged-parcel momentum S m u_F, not on W~ u", errF < 1e-12)

# frozen merge (S+ = S): then the merged covector keeps both identities
err1 = err2 = 0.0
for trial in range(20):
    mod, m0, m1, M0, M1 = setup(dt)
    C = build_C(mod, m1)
    S = fractions(mod, m0, np.diag(M0))
    Ct = S @ C
    # frozen fractions: W~+ - W~ := S (m+ - m)
    err1 = max(err1, np.abs(mod['Z'].T @ Ct).max())
    err2 = max(err2, np.abs(dt * Ct @ mod['ZF'] + (S @ (dt * m1))[:, None] * mod['Z']).max())
chk.true(f"frozen fractions S+ = S: Z^T Ct = 0 and dt Ct(Z) = -S(m+ - m) Z both hold (err {max(err1, err2):.1e}), W~+ - W~ is then S (m+ - m)", max(err1, err2) < 1e-12)

# ================= prop:obstruction =================
# needs no transport identity; its last sentence needs Z^T (W~+ - W~) Z = 0, true when the merge keeps class totals
mod, m0, m1, M0, M1 = setup(dt)
S, Sp = fractions(mod, m0, np.diag(M0)), fractions(mod, m0 + dt * m1, np.diag(M0 + dt * M1))
dWt = Sp @ (m0 + dt * m1) - S @ m0
chk.close("prop:obstruction last sentence: Z^T (W~+ - W~) Z = 0 (VOF mass conserved per class, merge class-preserving)", mod['Z'].T @ (dWt[:, None] * mod['Z']), np.zeros((2, 2)), tol=1e-12)
# sharp converse with the merge: an L with Z^T L = 0 and L Z = -dW~ Z exists (dense) iff Z^T dW~ Z = 0
Z = mod['Z']
P = Z @ np.linalg.inv(Z.T @ Z)
Ld = (-(dWt[:, None] * Z)) @ P.T
chk.close("dense L = D P^T: Z^T L = 0", Z.T @ Ld, np.zeros((2, len(dWt))), tol=1e-12)
chk.close("dense L = D P^T: L Z = -dW~ Z", Ld @ Z, -(dWt[:, None] * Z), tol=1e-12)

# ================= prop:redistribution (DG) =================
maxd = maxd2 = 0.0
for trial in range(30):
    mod, m0, m1, M0, M1 = setup(dt)
    C = build_C(mod, m1)
    ZF, Z = mod['ZF'], mod['Z']
    uext = rng.standard_normal(mod['nf']); u = rng.standard_normal(mod['nd'])
    r = step(mod, m0, m1, M0, M1, dt, u, uext, 0.5, C)
    dE = r['Etp'] - r['Et']
    # Y skew with Y Z = dE Z / 2 needs Z^T dE Z = 0:  Z^T dE Z = Z^T dM Z - Z^T dW~ Z ; enforce Z^T dM Z = 0 by projecting M1
    # (mass matching); build such M1:
    Pm = Z @ np.linalg.inv(Z.T @ Z)
    # remove the part of dE that violates it by modifying Mp: M1' = M1 - (Z (Z^T M1 Z) Z^T pseudo) symmetric
    Zp = np.linalg.pinv(Z)
    M1c = M1 - Zp.T @ (Z.T @ M1 @ Z) @ Zp
    r = step(mod, m0, m1, M0, M1c, dt, u, uext, 0.5, C)
    dE = r['Etp'] - r['Et']
    assert np.abs(Z.T @ dE @ Z).max() < 1e-10
    D = 0.5 * dE @ Z
    Y = D @ Pm.T - Pm @ D.T
    assert np.abs(Y @ Z - D).max() < 1e-10 and np.abs(Y + Y.T).max() < 1e-12
    up = np.linalg.solve(r['Mp'], r['Mp'] @ u + r['Sp'] @ (r['mp'] * (r['uW'] - r['uF'])) + 0.5 * (r['Et'] @ u - r['Etp'] @ u) + Y @ u)
    # constants
    U = rng.standard_normal(2)
    r2 = step(mod, m0, m1, M0, M1c, dt, Z @ U, ZF @ U, 0.5, C)
    upc = np.linalg.solve(r2['Mp'], r2['Mp'] @ (Z @ U) + 0.5 * (r2['Et'] @ (Z @ U) - r2['Etp'] @ (Z @ U)) + Y @ (Z @ U))
    maxd = max(maxd, np.abs(upc - Z @ U).max())
    dP = Z.T @ (r['Mp'] @ up) - Z.T @ (r['M'] @ u)
    pred = mu(mod, r['Sp'], r['mp'], r['uF']) - mu(mod, r['S'], r['m'], r['uF'])
    maxd2 = max(maxd2, np.abs(dP - pred).max())
chk.true(f"DG with merge: Y skew, Y Z = dE~ Z/2 exists when Z^T dE~ Z = 0; constants exact (err {maxd:.1e})", maxd < 1e-10)
chk.true(f"DG with merge: momentum change = mu+(u) - mu(u) (err {maxd2:.1e}); not 0 in general", maxd2 < 1e-10)

# ================= prop:theta (c) energy =================
# semi-discrete rate with merge: dE/dt = 1/2 u^T Wt' u - u^T S C u_F - u^T S mdot u_F + (theta - 1/2) u^T Et' u
def rate_formula(mod, m0, m1, M0, M1, C, u, uext, theta, h=1e-6):
    S = fractions(mod, m0, np.diag(M0))
    Sh = fractions(mod, m0 + h * m1, np.diag(M0 + h * M1))
    Wt0, Wt1 = S @ m0, Sh @ (m0 + h * m1)
    Wtdot = (Wt1 - Wt0) / h
    Etdot = M1 - np.diag(Wtdot)
    uF = uext.copy(); uF[mod['dofs']] = u
    return 0.5 * u @ (Wtdot * u) - u @ (S @ (C @ uF)) - u @ (S @ (m1 * uF)) + (theta - 0.5) * u @ Etdot @ u


def energy_rate_discrete(mod, m0, m1, M0, M1, C, u, uext, theta, dt):
    r = step(mod, m0, m1, M0, M1, dt, u, uext, theta, C)
    return (0.5 * r['up'] @ r['Mp'] @ r['up'] - 0.5 * u @ r['M'] @ u) / dt


mod, m0, m1, M0, M1 = setup(dt)
C = build_C(mod, m1)
u = rng.standard_normal(mod['nd']); uext = rng.standard_normal(mod['nf'])
for theta in (0.5, 1.0):
    f0 = rate_formula(mod, m0, m1, M0, M1, C, u, uext, theta)
    errs = []
    for d in (1e-3, 1e-4, 1e-5):
        errs.append(abs(energy_rate_discrete(mod, m0, m1, M0, M1, C, u, uext, theta, d) - f0))
    print(f"   theta={theta}: energy rate formula {f0:.6f}; |discrete - formula| for dt=1e-3,1e-4,1e-5: {errs[0]:.2e} {errs[1]:.2e} {errs[2]:.2e}")
    chk.true(f"semi-discrete energy rate formula with merge verified (theta={theta}): error -> 0 like dt", errs[2] < 2e-3 * max(1.0, abs(f0)) and errs[2] < errs[0] / 20)
# unmerged reference: S = I on dofs, F = D (faces without dof removed): rate (theta=1/2) = 0 -> prop:theta(c) of the note
mod_u = make_model(nf=8, nd=8)
mod_u['dofs'] = np.arange(8); mod_u['dof_of'] = np.arange(8); 
m0u = rng.uniform(0.8, 1.6, 8); m1u = conserving_pert(mod_u, 0.5); M0u = spd(rng, 8, 4.0) * 2; M1u = rng.standard_normal((8, 8)); M1u = 0.3 * (M1u + M1u.T) / 2
Cu = build_C(mod_u, m1u)
def fr_id(mod, m, Md, kappa=1.0):
    return np.eye(mod['nf'])
import types
def rate_unmerged(theta, h=1e-6):
    uu = rng.standard_normal(8)
    S = np.eye(8)
    Wtdot = m1u
    Etdot = M1u - np.diag(Wtdot)
    return 0.5 * uu @ (Wtdot * uu) - uu @ (Cu @ uu) - uu @ (m1u * uu) + (theta - 0.5) * uu @ Etdot @ uu, uu
val, uu = rate_unmerged(0.5)
chk.true(f"unmerged reference (no slaved values, S = I): rate at theta = 1/2 equals 0 up to (theta-1/2) = 0: |{val:.1e}| < 1e-12", abs(val) < 1e-12)
f_half = rate_formula(mod, m0, m1, M0, M1, C, u, uext, 0.5)
chk.true(f"FINDING (c) energy: with the merge the theta = 1/2 energy rate is {f_half:.3f} != 0 (merge defect), so prop:theta(c)/DS/DG exactness is lost", abs(f_half) > 1e-3)
exit_code(chk)

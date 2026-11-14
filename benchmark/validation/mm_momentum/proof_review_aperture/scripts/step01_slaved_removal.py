"""Formulation claim (1): Z^T M Z is unchanged when slaved values (aperture 0, tied to a donor with weight = their mass) are removed.
Label: prop:aperture, 'Formulation' ("does not change with the activity rule, because slaved values had (M Z)_f = 0 (ties vanish on constants)").
Method: random SPD M_A on the active values; full metric = M_A (+) 0 + sum_s w_s (e_s - e_d)(e_s - e_d)^T (the tie quadratic form
_mimetic_add_tie!: weight (u_row - u_donor)^2); Z with the class columns (liquid x, liquid y, gas x, gas y), identified values in two columns.
Checks: (M Z)_s = 0, (M Z)_d unchanged, Z^T M Z unchanged, the momentum Z^T M u independent of the slaved velocities;
FINDING: a slaved value whose donor is an IDENTIFIED grid-line value (dof_at returns the identified dof for both phases) breaks it.
Run: python3 proof_review/mimetic-momentum-transport-aperture/scripts/step01_slaved_removal.py
"""
import numpy as np
from kit import Checker, exit_code, spd

chk = Checker("step01 slaved-value removal")
rng = np.random.default_rng(41)


def build(n_act, n_sl, identified_donor=False):
    # active values: classes 0..3 (liquid x, liquid y, gas x, gas y); one identified x value (ones in columns 0 and 2)
    cls = list(rng.integers(0, 4, n_act - 1)) + ['id']
    Z = np.zeros((n_act + n_sl, 4))
    for i, c in enumerate(cls):
        if c == 'id':
            Z[i, 0] = Z[i, 2] = 1
        else:
            Z[i, c] = 1
    donors, slaved_cls = [], []
    for s in range(n_sl):
        if identified_donor:
            d = n_act - 1                      # donor = the identified value
            c = int(rng.choice([0, 2]))        # same axis (x), liquid or gas slaved value
        else:
            c = int(rng.integers(0, 4))
            cand = [i for i in range(n_act) if i != n_act - 1 and cls[i] == c]
            if not cand:
                c = cls[0] if cls[0] != 'id' else 1
                cls[0] = c
                Z[0] = 0; Z[0, c] = 1
                cand = [0]
            d = int(rng.choice(cand))
        donors.append(d)
        Z[n_act + s, c] = 1
        slaved_cls.append(c)
    return Z, donors


def full_metric(MA, n_sl, donors, w):
    n = MA.shape[0]
    M = np.zeros((n + n_sl, n + n_sl))
    M[:n, :n] = MA
    for s, (d, ws) in enumerate(zip(donors, w)):
        e = np.zeros(n + n_sl)
        e[n + s] = 1
        e[d] = -1
        M += ws * np.outer(e, e)
    return M


for trial in range(30):
    n_act, n_sl = int(rng.integers(6, 12)), int(rng.integers(1, 5))
    MA = spd(rng, n_act, 5.0)
    w = 10 ** rng.uniform(-14, 0, n_sl)           # tie weights down to 1e-14, as in the note
    Z, donors = build(n_act, n_sl)
    Mf = full_metric(MA, n_sl, donors, w)
    ZA = Z[:n_act]
    chk.close(f"[{trial}] (M Z)_s = 0 for slaved values", (Mf @ Z)[n_act:], 0 * (Mf @ Z)[n_act:], tol=1e-12)
    chk.close(f"[{trial}] (M Z)_f unchanged on active values", (Mf @ Z)[:n_act], MA @ ZA, tol=1e-12)
    chk.close(f"[{trial}] Z^T M Z unchanged", Z.T @ Mf @ Z, ZA.T @ MA @ ZA, tol=1e-12)
    u = rng.standard_normal(n_act + n_sl)
    u2 = u.copy()
    u2[n_act:] = rng.standard_normal(n_sl)       # different slaved velocities
    chk.close(f"[{trial}] momentum Z^T M u independent of the slaved velocities", Z.T @ Mf @ u, Z.T @ Mf @ u2, tol=1e-12)

# FINDING: identified donor
for trial in range(5):
    n_act, n_sl = 8, 2
    MA = spd(rng, n_act, 5.0)
    w = rng.uniform(0.1, 1.0, n_sl)
    Z, donors = build(n_act, n_sl, identified_donor=True)
    Mf = full_metric(MA, n_sl, donors, w)
    ZA = Z[:n_act]
    diff = Z.T @ Mf @ Z - ZA.T @ MA @ ZA
    # expected: sum_s w_s (Z_s - Z_d)(Z_s - Z_d)^T
    expected = sum(ws * np.outer(Z[n_act + s] - Z[d], Z[n_act + s] - Z[d]) for s, (d, ws) in enumerate(zip(donors, w)))
    chk.close(f"identified donor [{trial}]: defect = sum w_s (Z_s - Z_d)(Z_s - Z_d)^T", diff, expected, tol=1e-12)
    chk.true(f"FINDING identified donor [{trial}]: Z^T M Z changes by {np.abs(diff).max():.3f} (= w) and (M Z)_s != 0", np.abs(diff).max() > 1e-3 and np.abs((Mf @ Z)[n_act:]).max() > 1e-3)
exit_code(chk)

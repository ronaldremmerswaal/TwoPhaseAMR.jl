"""prop:constant (a),(b),(c)  [notes/mimetic_momentum_transport.tex, l.~150-175]
Random periodic MAC mesh, random shared face data (apertures, wetted midpoints), pressure points on the
symmetry line, blocks m = R G^{-1} R^T (m N = R), ties to donors added (vanish on constants).
Checks:
  (a) (M Z^pi_k)_{(f,pi)} = rho^pi a^pi_f h (x*_{c+} - x*_{c-})_k  (both orientations, tangential offsets)
  (b) Z^pi^T M Z^pi = rho^pi sum_c G^pi_c ;  Z^l^T M Z^g = 0 ;  G^l+G^g = h^2 I (prop:gram(b))
  (c) Z^pi^T W Z^pi = rho^pi V^pi I on a periodic mesh, W = rho h^2 * (alpha_{c-}+alpha_{c+})/2
  (d) FINDING: with an identified (grid-line) value the literal Z^pi of the note breaks (b).
Run: python3 proof_review/mimetic-momentum-transport/scripts/step01_prop_constant.py
"""
import numpy as np
from kit import Checker, exit_code
from meshmodel import Mesh

chk = Checker("step01 prop:constant")
for seed in range(4):
    rng = np.random.default_rng(100 + seed)
    nx, ny = (3, 4) if seed % 2 == 0 else (4, 3)
    h = 0.5 + 0.3 * seed
    m = Mesh(rng, nx, ny, h=h)
    F = m.F
    xs = [m.line_point(c, s=rng.uniform(-0.05, 0.05) * h) for c in range(m.ncell)]
    # symmetric G on the line, G^l + G^g = h^2 I
    for c in range(m.ncell):
        Gl, Gg = m.G(c, 0, xs[c]), m.G(c, 1, xs[c])
        assert abs(Gl[0, 1] - Gl[1, 0]) < 1e-12 and abs(Gg[0, 1] - Gg[1, 0]) < 1e-12
        assert np.allclose(Gl + Gg, h * h * np.eye(2), atol=1e-12)
    # donor ties between parallel same-phase faces (constants are in their kernel)
    ties = [(rng.integers(0, m.nx * m.ny), rng.integers(0, m.nx * m.ny), ph) for ph in (0, 1) for _ in range(5)]
    ties += [(m.nx * m.ny + rng.integers(0, m.nx * m.ny), m.nx * m.ny + rng.integers(0, m.nx * m.ny), ph)
             for ph in (0, 1) for _ in range(5)]
    M = m.assemble(xs, ties=ties, tie_w=0.37)
    Z = m.Zcols()
    chk.close(f"seed{seed}: M symmetric", M, M.T, 1e-12)
    # (a)
    worst = 0.0
    for ph in (0, 1):
        for f in range(F):
            ax = m.axis(f)
            i = f % m.nx if ax == 0 else (f - m.nx * m.ny) % m.nx
            j = (f // m.nx) if ax == 0 else (f - m.nx * m.ny) // m.nx
            cp = m.cell_index(i, j)
            cm = m.cell_index(i - 1, j) if ax == 0 else m.cell_index(i, j - 1)
            e = np.zeros(2)
            e[ax] = h
            a_f = m.a[f] if ph == 0 else 1 - m.a[f]
            expect = m.rho[ph] * a_f * h * (e + xs[cp] - xs[cm])
            for k in (0, 1):
                got = (M @ Z[:, 2 * ph + k])[f + ph * F]
                worst = max(worst, abs(got - expect[k]))
    chk.true(f"seed{seed}: (a) targets, both orientations, incl. tangential offsets", worst < 1e-12, f"(max err {worst:.1e})")
    # (b)
    for ph in (0, 1):
        Zp = Z[:, 2 * ph:2 * ph + 2]
        lhs = Zp.T @ M @ Zp
        rhs = m.rho[ph] * sum(m.G(c, ph, xs[c]) for c in range(m.ncell))
        chk.close(f"seed{seed}: (b) phase {ph}  Z^T M Z = rho sum G", lhs, rhs, 1e-11)
    chk.close(f"seed{seed}: (b) cross-phase block Z^l^T M Z^g = 0", Z[:, 0:2].T @ M @ Z[:, 2:4], np.zeros((2, 2)), 1e-12)
    # (c)
    alpha = rng.uniform(0, 1, m.ncell)
    mass = np.zeros(2 * F)
    for f in range(F):
        ax = m.axis(f)
        i = f % m.nx if ax == 0 else (f - m.nx * m.ny) % m.nx
        j = (f // m.nx) if ax == 0 else (f - m.nx * m.ny) // m.nx
        cp = m.cell_index(i, j)
        cm = m.cell_index(i - 1, j) if ax == 0 else m.cell_index(i, j - 1)
        al = 0.5 * (alpha[cp] + alpha[cm])
        mass[f] = m.rho[0] * h * h * al
        mass[F + f] = m.rho[1] * h * h * (1 - al)
    W = np.diag(mass)
    for ph, V in ((0, alpha.sum() * h * h), (1, (1 - alpha).sum() * h * h)):
        Zp = Z[:, 2 * ph:2 * ph + 2]
        chk.close(f"seed{seed}: (c) phase {ph}  Z^T W Z = rho V I", Zp.T @ W @ Zp, m.rho[ph] * V * np.eye(2), 1e-12)

# (d) identified value: merge the gas dof of one x-face into its liquid dof
rng = np.random.default_rng(7)
m = Mesh(rng, 3, 3, h=0.8)
F = m.F
xs = [m.line_point(c) for c in range(m.ncell)]
M = m.assemble(xs)
Z = m.Zcols()
f0 = 4
P = np.eye(2 * F)
P[F + f0, f0] = 1.0           # old gas dof of f0 := identified value
P = np.delete(P, F + f0, axis=1)
Mi = P.T @ M @ P
Zi = np.delete(Z, F + f0, axis=0)
Zi[f0, 2 + m.axis(f0)] = 1.0  # identified value belongs to both phases
for ph in (0, 1):
    Zp = Zi[:, 2 * ph:2 * ph + 2]
    lhs = Zp.T @ Mi @ Zp
    rhs = m.rho[ph] * sum(m.G(c, ph, xs[c]) for c in range(m.ncell))
    print("   ident: phase", ph, "defect of (b):\n", np.round(lhs - rhs, 6))
    chk.true(f"FINDING (d): (b) fails with an identified value, phase {ph} (needs 'no identified value' or a split Z)",
             np.max(np.abs(lhs - rhs)) > 1e-4)
# the defect is exactly rho^other * M_ii of the other-phase diagonal entry at the identified face (+ cross terms)
e = np.zeros(2 * F); e[F + f0] = 1
chk.close("ident defect (liquid column) = e^T M e for the merged gas dof",
          (Zi[:, 0:2].T @ Mi @ Zi[:, 0:2] - m.rho[0] * sum(m.G(c, 0, xs[c]) for c in range(m.ncell)))[0, 0],
          e @ M @ e, 1e-10)
exit_code(chk)

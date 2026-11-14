"""Small periodic-free lattice model of the liquid metric of TwoPhaseAMR (aperture activity) built from the ported blocks.

Geometry: a disk of radius R in an n x n lattice of unit cells (h = 1, rho = 1, only the LIQUID phase is assembled).
Face data are exact (length of the face inside the disk, liquid at one end), shared by the two cells.
Assembly (as _assemble_cut_cell_metric! with _PHASE_ACTIVITY = :aperture, small ties off):
  dofs = liquid faces with positive aperture and positive staggered mass;  uncut liquid cell: 1/2 on each face value;
  cut cell: Gram block (gram.py) with outward signs OUT = (-1,1,1,-1); lump m_f on a diagonal below 1e-12.
Staggered mass m_f = (alpha_- + alpha_+)/2 (|V_f| = h^2).  Merge (_mimetic_merged_masses): keep min(m, kappa M_ff), excess to the
same-axis face of the ring with the largest aperture (strictly greater, first in order), second ring if none.
"""
import math
import numpy as np
from gram import cell_blocks

OUT = (-1, 1, 1, -1)


class Lattice:
    def __init__(self, n, cx, cy, R, quad=1500):
        self.n, self.R, self.cx, self.cy = n, R, cx, cy
        self.ax = {}   # face -> (aperture, liquid at low end?)  faces: ('x', i, j) at x=i, y in [j,j+1]; ('y', i, j) at y=j, x in [i,i+1]
        n_ = n
        for i in range(n_ + 1):
            for j in range(n_):
                self.ax[('x', i, j)] = self._face_data(i, j, 'x')
        for i in range(n_):
            for j in range(n_ + 1):
                self.ax[('y', i, j)] = self._face_data(i, j, 'y')
        self.alpha = np.zeros((n_, n_))
        xs = (np.arange(quad) + 0.5) / quad
        for i in range(n_):
            for j in range(n_):
                X, Y = np.meshgrid(i + xs, j + xs, indexing='ij')
                self.alpha[i, j] = np.mean((X - cx) ** 2 + (Y - cy) ** 2 < R ** 2)

    def _face_data(self, i, j, kind):
        # segment p(t) = p0 + t e, t in [0,1]
        if kind == 'x':
            p0, e = (i, j), (0.0, 1.0)
        else:
            p0, e = (i, j), (1.0, 0.0)
        dx, dy = p0[0] - self.cx, p0[1] - self.cy
        A = e[0] ** 2 + e[1] ** 2
        B = 2 * (dx * e[0] + dy * e[1])
        C = dx * dx + dy * dy - self.R ** 2
        disc = B * B - 4 * A * C
        if disc <= 0:
            return (0.0, True)
        s = math.sqrt(disc)
        t0, t1 = (-B - s) / (2 * A), (-B + s) / (2 * A)
        lo, hi = max(t0, 0.0), min(t1, 1.0)
        if hi <= lo:
            return (0.0, True)
        a = hi - lo
        if a >= 1.0 - 1e-15:
            return (1.0, True)
        assert (lo == 0.0) != (hi == 1.0), "face crossed twice or liquid in the middle: choose other parameters"
        return (a, lo == 0.0)

    # --- cell data ---
    def cell_sides(self, i, j):
        return [('y', i, j), ('x', i + 1, j), ('y', i, j + 1), ('x', i, j)]

    def cell_ap_starts(self, i, j):
        a, st = [], []
        for k, f in enumerate(self.cell_sides(i, j)):
            ap, low = self.ax[f]
            a.append(ap)
            # code convention: starts_low = (True, True, False, False); start = low if starts_low[k] else not low
            st.append((low if k in (0, 1) else (not low)) if 0 < ap < 1 else False)
        return a, st

    def stag_fraction(self, f):
        kind, i, j = f
        n = self.n
        if kind == 'x':
            cells = [(i - 1, j), (i, j)]
        else:
            cells = [(i, j - 1), (i, j)]
        vals = [self.alpha[c] if 0 <= c[0] < n and 0 <= c[1] < n else 0.0 for c in cells]
        return 0.5 * (vals[0] + vals[1])

    def assemble(self, kappa=1.0, merge=True, small_ties=False):
        n = self.n
        faces = [f for f in self.ax]
        mass = {f: self.stag_fraction(f) for f in faces}
        dof = {}
        for f in faces:
            if self.ax[f][0] > 0 and mass[f] > 0:
                dof[f] = len(dof)
        N = len(dof)
        M = np.zeros((N, N))
        for i in range(n):
            for j in range(n):
                a, st = self.cell_ap_starts(i, j)
                sides = self.cell_sides(i, j)
                if all(x == 1.0 for x in a):
                    for f in sides:
                        if f in dof:
                            M[dof[f], dof[f]] += 0.5
                    continue
                if all(x == 0.0 for x in a):
                    continue
                pt, bl = cell_blocks(a, st)
                m, cons, B, sd, lengths, mom, G = bl[1]
                for q in range(m):
                    fq = sides[sd[q]]
                    for r in range(m):
                        fr = sides[sd[r]]
                        if fq in dof and fr in dof:
                            M[dof[fq], dof[fr]] += OUT[sd[q]] * OUT[sd[r]] * B[q, r]
        # lumping of a value without inertia
        for f, d in dof.items():
            if not M[d, d] > 1e-12:
                M[d, d] += mass[f]
        ties = 0
        if small_ties:
            # _MIMETIC_SMALL_TIES: dof with 0 < staggered fraction < 1 and aperture < 1/2 tied to the ring donor of larger aperture,
            # weight (1 - 2 a) m_f (quadratic form w (u_f - u_g)^2)
            for f, d in list(dof.items()):
                a = self.ax[f][0]
                if not (0 < mass[f] < 1 and a < 0.5):
                    continue
                g = self.donor_larger(f, dof, a)
                if g is None:
                    continue
                w = (1 - 2 * a) * mass[f]
                M[d, d] += w
                M[dof[g], dof[g]] += w
                M[d, dof[g]] -= w
                M[dof[g], d] -= w
                ties += 1
        W = np.zeros(N)
        for f, d in dof.items():
            W[d] = mass[f]
        info = {'dof': dof, 'mass': mass, 'faces': faces, 'ties': ties}
        if not merge:
            return M, W, W.copy(), info
        Wt, moved, dropped, donors = self.merge(dof, mass, M, kappa)
        info.update(moved=moved, dropped=dropped, donors=donors)
        return M, W, Wt, info

    def ring(self, f):
        kind, i, j = f
        if kind == 'x':
            r = [('x', i - 1, j), ('x', i + 1, j), ('x', i, j - 1), ('x', i, j + 1)]
        else:
            r = [('y', i, j - 1), ('y', i, j + 1), ('y', i - 1, j), ('y', i + 1, j)]
        return [g for g in r if g in self.ax]

    def donor(self, f, dof, own):
        best, bap = None, -1.0
        for g in self.ring(f):
            if g in dof and dof[g] != own:
                if self.ax[g][0] > bap:
                    best, bap = g, self.ax[g][0]
        if best is not None:
            return best
        for g0 in self.ring(f):
            for g in self.ring(g0):
                if g in dof and dof[g] != own:
                    if self.ax[g][0] > bap:
                        best, bap = g, self.ax[g][0]
        return best

    def donor_larger(self, f, dof, a):
        best, bap = None, -1.0
        for ring in (self.ring(f), [h for g0 in self.ring(f) for h in self.ring(g0)]):
            for g in ring:
                if g in dof and self.ax[g][0] > a + 1e-12 and self.ax[g][0] > bap:
                    best, bap = g, self.ax[g][0]
            if best is not None:
                return best
        return None

    def merge(self, dof, mass, M, kappa):
        N = len(dof)
        Wt = np.zeros(N)
        moved, dropped = 0, 0.0
        donors = set()
        for f, m in mass.items():
            if m <= 0:
                continue
            d = dof.get(f)
            keep = min(m, kappa * M[d, d]) if d is not None else 0.0
            excess = m - keep
            if d is not None:
                Wt[d] += keep
            if excess <= 0:
                continue
            g = self.donor(f, dof, d if d is not None else -1)
            tgt = dof[g] if g is not None else d
            if tgt is not None:
                Wt[tgt] += excess
                if g is not None:
                    moved += 1
                    donors.add(dof[g])
            else:
                dropped += excess
        return Wt, moved, dropped, donors

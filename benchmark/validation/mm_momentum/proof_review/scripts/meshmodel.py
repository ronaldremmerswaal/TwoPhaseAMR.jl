"""Random periodic MAC mesh with shared face data and Gram blocks (for prop:constant, eq:match).

Cell (i,j) = [i h,(i+1)h] x [j h,(j+1)h], local coordinates relative to the cell origin.
x-face fx(i,j) (normal e_x) sits at x=i h between cells (i-1,j) [c_-] and (i,j) [c_+];
y-face fy(i,j) at y=j h between (i,j-1) and (i,j).  Face data (liquid aperture a, tangential
position of the liquid wetted midpoint) are shared by the two cells; the gas midpoint follows
from tiling:  a x^l + (1-a) x^g = side midpoint.
Dofs: (f, liquid) = f, (f, gas) = F + f.
"""
import numpy as np


class Mesh:
    def __init__(self, rng, nx, ny, h=0.7, rho=(1.0, 1e-3), amin=0.1, amax=0.9):
        self.nx, self.ny, self.h, self.rho = nx, ny, h, np.array(rho, float)
        self.F = 2 * nx * ny
        nf = self.F
        self.a = rng.uniform(amin, amax, nf)
        self.t_l = np.empty(nf)  # unit-length tangential coordinate of the liquid centroid
        for f in range(nf):
            a = self.a[f]
            s0 = rng.uniform(0, 1 - a)
            self.t_l[f] = s0 + a / 2
        self.t_g = (0.5 - self.a * self.t_l) / (1 - self.a)
        self.cells = [(i, j) for j in range(ny) for i in range(nx)]
        self.ncell = len(self.cells)
        # per cell: list of 4 sides (face, outward normal, o sign, axis)
        self.sides = []
        for (i, j) in self.cells:
            self.sides.append([
                (self.fx((i + 1) % nx, j), np.array([1.0, 0.0]), +1, 0),   # east
                (self.fx(i, j), np.array([-1.0, 0.0]), -1, 0),             # west
                (self.fy(i, (j + 1) % ny), np.array([0.0, 1.0]), +1, 1),   # north
                (self.fy(i, j), np.array([0.0, -1.0]), -1, 1),             # south
            ])

    def fx(self, i, j):
        return i + self.nx * j

    def fy(self, i, j):
        return self.nx * self.ny + i + self.nx * j

    def axis(self, f):
        return 0 if f < self.nx * self.ny else 1

    def cell_index(self, i, j):
        return (i % self.nx) + self.nx * (j % self.ny)

    def side_point(self, c, k, phase):
        """local position (relative to the cell origin) of the wetted midpoint of side k."""
        h = self.h
        f, n, o, ax = self.sides[c][k]
        t = self.t_l[f] if phase == 0 else self.t_g[f]
        # east: x=h, y=t h ; west: x=0, y=t h ; north: y=h, x=t h ; south: y=0, x=t h
        if k == 0:
            return np.array([h, t * h])
        if k == 1:
            return np.array([0.0, t * h])
        if k == 2:
            return np.array([t * h, h])
        return np.array([t * h, 0.0])

    def b(self, c, k, phase):
        a = self.a[self.sides[c][k][0]]
        return a if phase == 0 else 1 - a

    def N(self, c):
        return np.array([s[1] for s in self.sides[c]])  # 4x2

    def eta_tau(self, c):
        h = self.h
        eta = np.zeros(2)
        tau = 0.0
        for k, (f, n, o, ax) in enumerate(self.sides[c]):
            a = self.a[f]
            x = self.side_point(c, k, 0)
            eta += a * h * n
            tau += a * h * (n[0] * x[1] - n[1] * x[0])
        return eta, tau

    def line_point(self, c, s=0.0):
        """pressure point on the symmetry line eta x x* = tau, at parameter s (length) along eta-hat."""
        eta, tau = self.eta_tau(c)
        e2 = eta @ eta
        p0 = tau / e2 * np.array([-eta[1], eta[0]])
        # centre the parametrisation at the cell centre projected on the line
        ctr = np.array([self.h / 2, self.h / 2])
        sc = (ctr - p0) @ eta / np.sqrt(e2)
        return p0 + (sc + s) * eta / np.sqrt(e2)

    def R(self, c, phase, xs):
        h = self.h
        return np.array([self.b(c, k, phase) * h * (self.side_point(c, k, phase) - xs)
                         for k in range(4)])  # 4x2

    def G(self, c, phase, xs):
        return self.N(c).T @ self.R(c, phase, xs)

    def block(self, c, phase, xs):
        Rm = self.R(c, phase, xs)
        Gm = self.N(c).T @ Rm
        return Rm @ np.linalg.inv(Gm) @ Rm.T, Rm, Gm

    def assemble(self, xstar, ties=None, tie_w=0.0):
        F = self.F
        M = np.zeros((2 * F, 2 * F))
        for c in range(self.ncell):
            for ph in (0, 1):
                m, _, _ = self.block(c, ph, xstar[c])
                idx = [self.sides[c][k][0] + ph * F for k in range(4)]
                o = np.array([self.sides[c][k][2] for k in range(4)], float)
                M[np.ix_(idx, idx)] += self.rho[ph] * (o[:, None] * m * o[None, :])
        if ties:
            for (f, g, ph) in ties:
                if f == g:
                    continue
                v = np.zeros(2 * F)
                v[f + ph * F] = 1
                v[g + ph * F] = -1
                M += tie_w * self.rho[ph] * np.outer(v, v)
        return M

    def Zcols(self):
        F = self.F
        Z = np.zeros((2 * F, 4))
        for f in range(F):
            ax = self.axis(f)
            Z[f, ax] = 1          # liquid, columns 0,1
            Z[F + f, 2 + ax] = 1  # gas, columns 2,3
        return Z

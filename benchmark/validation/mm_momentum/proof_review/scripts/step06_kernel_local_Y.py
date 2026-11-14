"""prop:redistribution, kernel claim and the *local* Y question  [notes/mimetic_momentum_transport.tex, l.~320-330]
Single phase, values with an axis (x or y).  Graph edges (f,g); edge-to-value map B: y -> Y Z for Y = sum_e y_e (e_f e_g^T - e_g e_f^T),
row (f, axis(g)) gets +y_e, row (g, axis(f)) gets -y_e.   Unknown r in R^{N x 2}.
Checks:
  K1 Z S (S symmetric 2x2, supported on one connected component) is in ker B^T   (claim of the note).
  K2 ker B^T = {Z S_C}_C exactly (dim 3 per component) when the x-values and y-values of the component are each connected
     through same-axis edges and at least one x-y edge exists (e.g. the 4 values of a cell, complete graph, cell clusters);
     it is STRICTLY larger otherwise (path x1-y1-x2): then the condition has additional per-value parts.
  K3 D in range(B) <=> D orthogonal to ker B^T <=> sym(Z_C^T D_C) = 0 for every component C (when K2 holds).
  K4 the global condition Z^T dE Z = 0 (obstruction) does NOT imply a local Y: two components with opposite Gram-mass changes
     (Z_1^T dE Z_1 = S, Z_2^T dE Z_2 = -S): a dense Y exists, a Y supported on the component graph does not.
Run: python3 proof_review/mimetic-momentum-transport/scripts/step06_kernel_local_Y.py
"""
import numpy as np
from kit import Checker, exit_code, nullspace, sym

chk = Checker("step06 kernel / local Y")


def cluster_graph(cells, ring=False):
    """cells: list of (i,j).  Values = faces of the cells (shared between neighbours).  Edges = all pairs of values of one cell;
    ring=True adds edges between all same-axis values of two cells that share a face (a stand-in for the face ring)."""
    faces = {}
    def fid(kind, i, j):
        key = (kind, i, j)
        if key not in faces:
            faces[key] = len(faces)
        return faces[key]
    edges = set()
    cellvals = {}
    for (i, j) in cells:
        vs = [fid("x", i, j), fid("x", i + 1, j), fid("y", i, j), fid("y", i, j + 1)]
        cellvals[(i, j)] = vs
        for a in range(4):
            for b in range(a + 1, 4):
                edges.add(tuple(sorted((vs[a], vs[b]))))
    if ring:
        for (i, j) in cells:
            for (di, dj) in ((1, 0), (0, 1)):
                if (i + di, j + dj) in cellvals:
                    A, Bv = cellvals[(i, j)], cellvals[(i + di, j + dj)]
                    for p in (0, 1):                  # x pair, y pair
                        for a in A[2 * p:2 * p + 2]:
                            for b in Bv[2 * p:2 * p + 2]:
                                if a != b:
                                    edges.add(tuple(sorted((a, b))))
    axis = [0] * len(faces)
    for (kind, i, j), idx in faces.items():
        axis[idx] = 0 if kind == "x" else 1
    return len(faces), axis, sorted(edges)


def edge_matrix(N, axis, edges):
    B = np.zeros((2 * N, len(edges)))
    for e, (f, g) in enumerate(edges):
        B[2 * f + axis[g], e] += 1.0
        B[2 * g + axis[f], e] -= 1.0
    return B


def Zmat(N, axis):
    Z = np.zeros((N, 2))
    for f in range(N):
        Z[f, axis[f]] = 1
    return Z


def components(N, edges):
    parent = list(range(N))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for f, g in edges:
        parent[find(f)] = find(g)
    comp = {}
    for f in range(N):
        comp.setdefault(find(f), []).append(f)
    return list(comp.values())


# ---------------- K1/K2 on cell clusters ----------------
cases = (("one cell", [(0, 0)]),
         ("2x1 block", [(0, 0), (1, 0)]),
         ("L-shape", [(0, 0), (1, 0), (0, 1)]),
         ("two separated cells", [(0, 0), (5, 5)]),
         ("block + far cell", [(0, 0), (1, 0), (0, 1), (1, 1), (8, 8)]))
# cell-only graphs: kernel strictly larger than {Z S_C} when neighbouring cells do not share same-axis edges across the cluster
for name, cells in cases:
    N, axis, edges = cluster_graph(cells)
    B = edge_matrix(N, axis, edges)
    K = nullspace(B.T)
    comps = components(N, edges)
    print(f"   cell-only graph, {name}: dim ker B^T = {K.shape[1]} (3 per component would be {3*len(comps)})")
    chk.true(f"{name} (cell-only): ker B^T contains Z S_C (dim >= 3 per component)", K.shape[1] >= 3 * len(comps))
    if name in ("2x1 block", "L-shape", "block + far cell"):
        chk.true(f"{name} (cell-only): FINDING kernel strictly larger than {{Z S_C}}", K.shape[1] > 3 * len(comps))
for name, cells in cases:
    N, axis, edges = cluster_graph(cells, ring=True)
    B = edge_matrix(N, axis, edges)
    Z = Zmat(N, axis)
    comps = components(N, edges)
    K = nullspace(B.T)
    chk.true(f"{name}: K2 dim ker B^T = 3 * (#components) = {3*len(comps)}", K.shape[1] == 3 * len(comps), f"(dim {K.shape[1]})")
    ok = True
    for C in comps:
        for S in (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [1, 0]]), np.array([[0, 0], [0, 1]])):
            r = np.zeros((N, 2))
            r[C] = Z[C] @ S
            ok &= np.max(np.abs(B.T @ r.reshape(-1))) < 1e-12
    chk.true(f"{name}: K1 Z S_C in ker B^T for every component", ok)
    # K3: range characterisation
    rng = np.random.default_rng(1)
    D = (B @ rng.standard_normal(len(edges))).reshape(N, 2)
    condition = max(np.max(np.abs(sym(Z[C].T @ D[C]))) for C in comps)
    chk.true(f"{name}: K3 D = B y satisfies sym(Z_C^T D_C) = 0 for all C", condition < 1e-12)
    # random D with that property, built by projecting out ker B^T: D' = D - K K^T D
    D2 = rng.standard_normal((N, 2))
    D2p = (D2.reshape(-1) - K @ (K.T @ D2.reshape(-1))).reshape(N, 2)
    cond2 = max(np.max(np.abs(sym(Z[C].T @ D2p[C]))) for C in comps)
    sol = np.linalg.lstsq(B, D2p.reshape(-1), rcond=None)[0]
    chk.true(f"{name}: K3 projected D solvable and sym(Z_C^T D_C)=0", np.linalg.norm(B @ sol - D2p.reshape(-1)) < 1e-10 and cond2 < 1e-12)
    # random D violating the per-component condition is not solvable
    sol = np.linalg.lstsq(B, D2.reshape(-1), rcond=None)[0]
    chk.true(f"{name}: K3 generic D not solvable", np.linalg.norm(B @ sol - D2.reshape(-1)) > 1e-6)

# ---------------- K2 negative case: path x1 - y1 - x2 ----------------
axis = [0, 1, 0]
edges = [(0, 1), (1, 2)]
B = edge_matrix(3, axis, edges)
K = nullspace(B.T)
chk.true("path x1-y1-x2: ker B^T has dim 4 > 3 (strictly larger than {Z S})", K.shape[1] == 4, f"(dim {K.shape[1]})")
Z = Zmat(3, axis)
# D with sym(Z^T D) = 0 but not solvable (diagonal entries (x1,x), (x2,x) cannot be produced: they need same-axis edges)
D = np.zeros((3, 2))
D[0, 0] = 1.0
D[2, 0] = -1.0
chk.close("path: sym(Z^T D) = 0", sym(Z.T @ D), np.zeros((2, 2)), 1e-14)
sol = np.linalg.lstsq(B, D.reshape(-1), rcond=None)[0]
chk.true("path: ... yet D not in the range of B (necessary condition not sufficient)", np.linalg.norm(B @ sol - D.reshape(-1)) > 0.5)

# ---------------- K4 two components: global condition holds, local fails ----------------
N, axis, edges = cluster_graph([(0, 0), (6, 6)], ring=True)
B = edge_matrix(N, axis, edges)
Z = Zmat(N, axis)
comps = components(N, edges)
rng = np.random.default_rng(5)
Sg = np.array([[0.3, 0.1], [0.1, -0.2]])
dE = np.zeros((N, N))
for C, sgn in zip(comps, (+1, -1)):
    n = len(C)
    X = sym(rng.standard_normal((n, n)))
    Zc = Z[C]
    Pc = Zc @ np.linalg.inv(Zc.T @ Zc)
    X = X - Pc @ (Zc.T @ X @ Zc) @ Pc.T            # Zc^T X Zc = 0
    X = X + sgn * Pc @ Sg @ Pc.T                    # Zc^T X Zc = sgn * Sg
    dE[np.ix_(C, C)] = X
chk.close("K4 global Z^T dE Z = 0", Z.T @ dE @ Z, np.zeros((2, 2)), 1e-12)
chk.true("K4 per component Z_C^T dE Z_C = +-S != 0", np.max(np.abs(Z[comps[0]].T @ dE[np.ix_(comps[0], comps[0])] @ Z[comps[0]])) > 0.1)
D = 0.5 * dE @ Z
P = Z @ np.linalg.inv(Z.T @ Z)
Y = D @ P.T - P @ D.T
chk.close("K4 dense global Y exists (Y Z = D, skew)", Y @ Z, D, 1e-12)
sol = np.linalg.lstsq(B, D.reshape(-1), rcond=None)[0]
res = np.linalg.norm(B @ sol - D.reshape(-1))
chk.true("K4 local Y on the component graph does NOT exist", res > 1e-2, f"(residual {res:.3e})")
exit_code(chk)

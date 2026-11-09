# Audit: eq:viscosity:cut — single-phase operator with a solid (embedded wall connections, covered unknowns, no band)

- Source: `docs/techref/chapters/09_viscosity.tex:132-176` (text and eq. `cut`), commit `03bad7a` (after #74/#75); code `src/embedded/operators.jl` (`_embedded_wall_connections`, `_wall_connection_measure`, `_cut_cell_diffusion_component`), `src/operators/staggered_diffusion.jl` (`_wall_value`, `_include_wall_connection`)
- Date: 2026-11-09
- Overall status: ⚠️ the formulas and the #75 statement are verified; **two pre-existing code defects (D1, D2)** and three text precisions (no band, covered unknowns, `VelocityBoundaryConditions`)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-viscosity-cut/scripts/run_all.jl`

## After the #79 fix — 2026-11-09

D2 is fixed for both scalar and variable-viscosity diffusion. A face shared by
a pure-fluid cell and a covered cell supplies an embedded wall segment;
adjacent hanging-face pieces are merged before applying the existing cut-wall
conductance. The regression `defect_02_grid_aligned_solid.jl` now checks the
stationary no-slip rate at offsets 0 and `1e-9`, with a fraction floor small
enough to retain the shifted cut.

| Operator | Rate at offset 0 | Rate at offset `1e-9` | Relative difference |
|---|---:|---:|---:|
| Scalar | `-5.65685424949238` | `-5.656854272119798` | `4.00e-9` |
| Variable | `-0.015313708498984762` | `-0.015313708544239602` | `2.96e-9` |

The entries below record the original audit and its pre-fix observations;
they are retained as history rather than claims about the current operator.

## Statement and assumptions
With a solid the constructor from a `GeometryField` uses the cut-cell schedule (weights × apertures, open dual volumes, 1 where they vanish), **no corrected band** ($\lapstag k=-\dualvolmat^{-1}\viscstiff k$ in every row, symmetric up to the volumes), the domain walls of eq. `walls` with each measure multiplied by the open fraction of the wall segment the connection owns (covered segments dropped), and two kinds of embedded connections: for every cut cell $c$ (segment length $\meas{\interface^{\rm s}_c}$, midpoint $\pos{\interface^{\rm s}_c}$) and every unknown $f$ of component $k$ on its sides, $\text{cond}=\tfrac12\varpi_{c,f}\meas{\interface^{\rm s}_c}/\max(|x_f-x_{\interface^{\rm s}_c}|_2,\tfrac14\min_th_t(c))$ (omitted for `FreeSlip`); $\text{cond}=1$ for covered unknowns; wall value zero for every wall type (#75).

## Steps
| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| embedded conductance (cut kind) and covered kind | l. 148-157 | ✅ | independent list (segment length/midpoint from `_local_plic_segment`, everything else recomputed) equals the code's connections with `boundary_axis = 0` (face, conductance $10^{-13}$, tangential flag) on 3 random composite/periodic disc geometries that build (13 checks); **the docs cut fixture of Ch. 8/9 (walled composite mesh, sloped solid $y<0.9+0.1x$) is itself rejected by this constructor (D1)** | `code_01_cut.jl` | the PLIC segment itself is the subject of Ch. 4/7 |
| wall value zero for every wall type (#75) | l. 159-161 | ✅ | zero velocity field, lid data `NoSlip(lid)` / `VelocityBoundaryConditions(top = NoSlip(lid))`: every unknown that has only embedded connections has rate exactly 0 (412 checks), `_wall_value` returns 0 for all embedded connections; the lid still acts on domain-wall unknowns (rate 8.0) | same | |
| domain-wall measure clipped by the open fraction; covered segments dropped | l. 144-147 | ✅ | solid covering the whole bottom wall: 0 bottom connections, 46 top | same | |
| `FreeSlip` omits the cut kind and keeps the covered kind | l. 153, 160-161 | ✅ for `FreeSlip` | `|S_NoSlip − S_FreeSlip|=8.19` | same | **but** see F3 |
| no corrected band | l. 139-141 | ✅ | $\lapstag k-(-V^{-1}S)=0$ exactly; the no-solid operator on the same composite mesh has 286 band rows | same | consequence F4 |
| covered unknowns are "tied to the stationary solid" | l. 161-162 | ⚠️ | covered rows: isolated (no connections), $V=1$, $\text{cond}=1$: row $=-1\cdot u$ (15 and 14 covered unknowns, all isolated). The midpoint step relaxes them to $u/(1+\tfrac12\Delta t\nu)$: a penalty, not a constraint. In the variable-viscosity operator the covered unknowns also enter the vertex shear stencils, and $\bar u^{[2]}$ on covered faces is $2.6\times10^{-5}$ of the maximum (`eq-viscosity-two-velocity`) | same | zero only if the data are zero and $\nu\Delta t\ll1$; the flow zeroes them every step (`enforce_geometry_no_slip!`) |
| constructor works for general geometries | implied | ❌ | **D1** below | `defect_01_constructor.jl` | |
| every solid surface carries the no-slip row | implied | ❌ | **D2** below | `defect_02_grid_aligned_solid.jl` | |

## Errors and consequences

**D1 (pre-existing, code).** `StaggeredDiffusionOperator(geometry; viscosity = 1)` throws `ArgumentError: degenerate AMR dual diffusion connection` for solids that cut cells at a level transition. Minimal reproducer (8×8 level-0 cells, only the lower-left 4×4 block refined, solid $y<2.3+0.1x$): `defect_01_constructor.jl`. Of 29 random composite meshes with a disc, 16 are rejected; the horizontal half plane $y<2.3$ and the variable-viscosity operator on the same geometry build. Cause: the dual face between two vertically adjacent $x$-unknowns on the interface ((4.0, 2.25), (4.0, 2.75)) has absolute measure 0.075 consisting only of an $x$-flux weight ($-0.15\times0.5$) after the aperture scaling, so its distance along $x$ is 0, `_dual_diffusion_connections` skips it (`distance > 256eps(T)*extent || continue`) and, since no axis remains, throws. The single-phase cut-cell operator is therefore only usable on uniform meshes or when no solid crosses a level transition; the test list of the chapter uses a uniform mesh.

> **Fixed by #78 (2026-11-10).** Since PR #88 the cut-cell constructor rebuilds the transverse connections from open-trace overlaps; `defect_01_constructor.jl` is now a regression: the minimal reproducer and all 29 random disc geometries (previously 16 rejected) build, each with an exactly symmetric stiffness.

**D2 (pre-existing, code).** A solid whose boundary lies exactly on cell faces has no cut cell, hence no embedded row: no no-slip condition (tangentially stress free). Reproducer: solid $y<1+\delta$, $u_1\equiv1$, rate of the first open unknown: $0$ for $\delta=0$, $-5.68$ for $\delta=10^{-3}$ (single phase); variable operator $-0.0040$ vs $-0.0154$ (there the covered unknown acts as a wall over a full cell). `defect_02_grid_aligned_solid.jl`.

**F3 (text).** For `VelocityBoundaryConditions` the embedded cut connections are always kept (`boundary_axis = 0 ⇒ true`): the embedded walls are no-slip whatever the sides are (stiffness identical to `NoSlip` with all-FreeSlip sides; $|S_{\rm VBC(FreeSlip)}-S_{\rm NoSlip}|=0$). "(omitted for FreeSlip)" is true only for the `FreeSlip()` type.

**F4 (text).** Without the band the cut-cell operator is not consistent at level transitions: on a composite mesh with the solid far away, truncation error for $u^*=\sin\sin$ at refinement-interface rows $0.0702,0.0685,0.0681$ ($n=16,32,64$) and at the other core rows $0.0169,0.0168,0.0167$ — constant, not decreasing (for comparison $|\Delta u^*|\le0.31$; the no-solid operator with the band: $0.0206\to0.0059$, order 1, and $\le6\times10^{-5}$ in the core). The *solution* converges nevertheless with order $1.6,1.8$ (Helmholtz problem, grid-aligned solid face with a Neumann-compatible $u^*$, composite mesh).

> **Superseded by #79 (2026-11-10).** The solution order $1.6,1.8$ above was measured with $u^*=\sin(\pi x/L)\cos(\pi(y-1)/(2(L-1)))$, which is stress free on the grid-aligned solid face $y=1$ and fitted the operator only while that face had no embedded row (D2). Since #79 the face is an embedded no-slip wall, and U4c of `code_01_cut.jl` uses $u^*=\sin(\pi x/L)\sin(\pi(y-1)/(L-1))$, which vanishes on it. The solution error of the solve ($n=16,32,64$) is first order in every norm: composite mesh maximum $3.8,2.1,1.1\times10^{-2}$ (order $0.86,0.93$), volume-weighted L2 $5.8,3.1,1.6\times10^{-3}$ ($0.92,0.95$), L1 $2.9,1.5,0.73\times10^{-3}$ ($0.99,0.99$); uniform mesh maximum $3.1,1.9,1.1\times10^{-2}$ ($0.71,0.86$), L2 $7.6,4.1,2.1\times10^{-3}$ ($0.91,0.97$), L1 $4.1,2.0,0.99\times10^{-3}$ ($1.02,1.01$). The truncation error $(\mathsf L-\Delta)u^*$ of the rows next to the solid grows like $1/h$ ($1.10,2.21,4.42$ composite, $0.26,0.53,1.05$ uniform), while the interior rows converge with order two ($1.2\times10^{-3}\to8.3\times10^{-5}$): the order is set by the embedded wall rows, not by the level transition (on the composite mesh the largest error sits at the first open unknown above the solid). The truncation numbers above predate #78/#87: the rerun of U4b gives $0.0890,0.0897,0.0899$ at the refinement-interface rows (still order one) and $9.5\times10^{-4},2.4\times10^{-4},6.2\times10^{-5}$ (order two) at the other core rows.

Replacements:

```latex
% after "no corrected band (...)":
no corrected band ($\lapstag{k}=-\dualvolmat^{-1}\viscstiff{k}$ in every row, symmetric up to the volumes, hence with
a truncation error of order one at the rows of a level transition, where the net measures are smaller than the dual faces,
\cref{eq:viscosity:connections}) ...
% eq. cut, second line and the sentence after it:
\wallcond{w}=1\quad(f\text{ covered},\ \opendualvol{f}=0),
...
with wall value zero for every wall type, also for a moving \code{NoSlip(u)} (since \issue{75}); the second kind ties
a covered unknown to the stationary solid for both wall types, as a penalty with unit volume (the row $-u_f$), so the covered
values decay by $1/(1+\tfrac12\dt\visc/\dens)$ per step and are zeroed by the flow after every step. With
\apitype{VelocityBoundaryConditions} the embedded cut connections are always no-slip whatever the sides are.
% add to the Cut cells paragraph:
A solid surface that coincides with cell faces has no cut cell and so no embedded connection (defect, issue to be filed);
the constructor fails with a solid that cuts cells at a level transition (defect, issue to be filed), so the cut-cell
operator is used on uniform meshes only.
```
(the last two sentences are factual statements of the current behaviour, to be removed when the defects are fixed).

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| conductance formula (cut kind) | sharp as an identity with the code | $10^{-13}$ | `code_01_cut.jl` | the $\max(\cdot,\tfrac14h_{\min})$ clip makes the row first order at best |
| "no band, symmetric" | sharp | exact | same | consistency at transitions requires the band or a wider cut-cell stencil |
| tie of covered unknowns | **not a constraint** | row $-u$, relaxation factor $1/(1+a)$; variable operator $2.6\times10^{-5}$ | `code_01_cut.jl`, `eq-viscosity-two-velocity` | pin exactly (eliminate) |
| embedded rows exist for every solid surface | **false** (D2) | $0$ vs $-5.68$ | `defect_02_grid_aligned_solid.jl` | add rows for grid-aligned faces |

## Open points
- File issues for D1 and D2 (minimal reproducers in the scripts); D1 blocks single-phase AMR with a solid across a transition.

## History
- 2026-11-09: first audit.
- 2026-11-10: U4c updated to the post-#79 boundary (embedded no-slip on the aligned face; composite and uniform meshes) and the F4 solution order marked superseded by #79; U4b rerun. `code_01_cut.jl`: PASS. `defect_01_constructor.jl` still asserts the pre-#78 throw and therefore fails since #78 (not updated here).
- 2026-11-10: D1 reproducer converted into a #78 regression (0 of 29 random geometries rejected, all stiffnesses symmetric); U4c also reports the L2/L1 solution errors and the truncation of the wall rows. `run_all.jl`: PASS (3 scripts).

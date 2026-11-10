# Audit: sec-verif-catalogue-spotcheck — five random rows of the catalogue tables of Chapter 14 against the cited tests

- Source: `docs/techref/chapters/14_verification.tex` (`tab:verif:orders`, `tab:verif:invariants`, `tab:verif:benchmarks`), branch `docs/techref-verification`, commit `77f46cb`
- Label: none (table rows; slug `sec-verif-catalogue-spotcheck`)
- Date: 2026-11-10
- Overall status: ✅ verified — all five rows agree with the assertions of the cited tests; the measured values of the one row that was run are reproduced to the printed digits
- Rerun: `julia --project=@claude-symbolics proof_review/sec-verif-catalogue-spotcheck/scripts/run_all.jl`

## Statement and assumptions

The three longtables have 71 data rows (orders 38, invariants 19, benchmarks 14). Five were drawn with `random.seed(1411)` over the list of rows in table order (indices 13, 3, 59, 7, 22). Each row was compared with the `@test` lines of the test it cites (quantity, reference, norm, asserted bound, mesh sequence); the measured column cannot be derived from the test source (it quotes the CI log), so the row whose cost is moderate (the capillary wave) was rerun.

## Steps

| Row (table) | Cited test | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| Staggered convection, composite (`orders`): dual-volume $L^2$ of the tendency, mm velocity, all $>1.3$; $>1.35$ (Lax–Wendroff, limited at transitions) | `test/operators/advection/staggered_advection.jl`, "mixed-level staggered convection convergence" | ✅ | read: error $=\sqrt{\sum V_f e_f^2/\sum V_f}$ over the $u$ faces of a composite tree (`block_size` 4, 8, 16) against the exact tendency of a manufactured field; loop over `LaxWendroff()` and `MonotonizedCentral()` asserts both rates `>1.3`; `LaxWendroff()` with `MonotonizedCentral()` at the interface asserts `>1.35` | – | the table's "mm velocity" is the constant-speed manufactured field $(0.3,-0.2)$ with $\sin(2\pi x)\cos(2\pi y)$; "dual-volume" is the volume-weighted norm |
| Staggered Laplacian, uniform periodic (`orders`): dual-volume $L^2$ of the tendency, both $>1.9$ | `test/operators/staggered_diffusion.jl`, "uniform periodic staggered Laplacian convergence" | ✅ | read: $n=8,16,32$, both rates `> 1.9`, $L^2$ weighted by `component.volumes` | – | |
| Capillary wave, La 3000, DR 1, one velocity (8, 16, 32) (`benchmarks`): $0.0646,0.0126,0.0043$; end $1.96$; decreasing, end $\ge1.8$ | `test/integration/two_phase_waves.jl`, "Prosperetti standing waves on uniform and AMR meshes", `studies[1]=(1.0,:capillary,1.8)` | ✅ | **run**: helpers of the test file evaluated verbatim, `wave_case(:uniform, n, 1.0, :capillary; formulation=CGFM())`: errors $0.064555, 0.012574, 0.0042664$, endpoint rate $\log_2(e_1/e_3)/2=1.9597$; asserted `all(diff(errors) .< 0)` and `endpoint_rate >= 1.8` | `code_01_capillary_row.jl` | "end" in the text is the endpoint rate $\log_2(e_{\rm first}/e_{\rm last})/(m-1)$, as the test defines it; the pairwise rates are $2.36$, $1.56$ |
| Inclined embedded wall (`orders`): maximum, $L^2$, $L^1$; all $\ge0.6$ (max.), $\ge0.8$ ($L^2$, $L^1$); finest error $\le1.02\times$ pinned | `test/operators/embedded_wall_rows.jl`, "inclined embedded-wall Helmholtz comparison (#106)" | ✅ | read: three norms (maximum, volume-weighted $L^2$, $L^1$) of the Helmholtz error on $n=16,32,64$, uniform and composite; `order >= (norm == 1 ? 0.6 : 0.8)` for both levels; `errors[end] <= 1.02 * baseline` | – | the table does not say that both a uniform and a composite mesh are asserted (it is the same row) |
| Anisotropic VOF translation (`orders`): $L^1$ fraction, exact translation, all $>1.8$ | `test/integration/anisotropic_cells.jl`, "anisotropic geometric VOF advection" | ✅ | read: $2n\times n$ cells, $n=16,32,64$; $L^1=\sum|c|\,|\varphi-\varphi_{\rm ref}|$ against a re-initialised translated circle; `all(diff(errors) .< 0)`, `minimum(rates) > 1.8`; additionally a volume defect $\le64\cdot64^2\varepsilon$ (not in the table) | – | |

## Errors and consequences

No discrepancy between the five rows and the assertions. Remarks that do not change the table:
- the "Measured" column of the rows that were only read stays unchecked (`--` entries need no check; the printed values are CI-log values);
- the table does not list the additional assertions of the tests (volume defect of the anisotropic VOF row; both mesh types of the inclined-wall row).

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| Asserted bounds are lower bounds below the measured orders (capillary row) | likely sharp: pairwise rates $2.36$, $1.56$ vs endpoint bound $1.8$ | one rerun | `code_01_capillary_row.jl` | the bound applies to the endpoint rate; the second pair ($1.56$) alone would not satisfy $1.8$ — the text's "end" wording is correct, "all" would not be |

## Open points

- The remaining 66 rows are unchecked in this audit; the other tables' measured values quote the CI logs of commit `e736e6b` / the audited commit.

## History
- 2026-11-10: first audit (branch `docs/techref-verification`, commit `77f46cb`): five rows compared with the tests; the capillary-wave row was rerun.
- 2026-11-10 (end of audit): Final `run_all.jl`: all checks passed (10 assertions; Runic 1.10.0 formatted).

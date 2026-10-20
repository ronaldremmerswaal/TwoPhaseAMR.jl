# Audit: slice `docs` (README, docs/, examples/, test/examples/, Documentation.yml)

Environment: Julia 1.12.7, `-t 1`, BLAS 1 thread, one Julia process at a time. Scratch only in `/tmp/audit-docs/` (docs copy, CairoMakie env, PoC scripts, PoC PNG/GIF in `/tmp/audit-docs/poc/`). No tracked file was modified (`git status` shows only the pre-existing untracked `benchmark/scaling/`, `docs/src/assets/`, `research/`).

## 1. Summary

- **Docs build is clean but static**: Documenter 1.19 build of a /tmp copy took **46 s** (+82 s one-off Documenter precompile, +8 s package precompile); zero missing-docstring / broken-`@ref` errors. Only real warnings: `api.md` (117 KiB) and `api-functions.md` (101 KiB) exceed Documenter's 100 KiB `size_threshold_warn` (hard error at 200 KiB). Only **8 of ~74** Julia blocks are executed (`@example`, getting-started + single-phase tutorial); the rest are static, which is why a **broken Quick-start snippet** (AMR block of `getting-started.md`) was never caught.
- **No figures anywhere** (0 `<img>` in the built site, no image syntax in README/docs). `docs/src/assets/` (untracked, 4.5 MB of benchmark SVG/PNG/JSONL) is unreferenced and would be published wholesale by Documenter if committed.
- **Headless figures work today, but only through the interactive viewer**: `visualize(fields; display=false)` + CairoMakie `save` produced a PNG (PoC), but the figure carries the GUI widgets and a cluttered layout. A clean path exists with exported pieces (`VisualizationSession(fields; viewer=NoVisualization(), checkpoints=N)` → `visualization_checkpoints` → `frame[:name]` + `frame.topology.cell_geometry`): static multi-panel PNG and GIF PoCs both rendered. A small `snapshot_figure`-style API in the Makie ext would remove ~25 lines of hand-rolled mesh code per example.
- **Examples**: ~30 lines of identical viewer/session/`fetch(Threads.@spawn …)` tail in 10 files, 8 copies of the initial-adaptation loop; `lid_driven_cavity` default run is **one single step of dt = 15** (README says "minutes"; real cost 0.26 s); `lgpi` at `max_level=2` threw `ArgumentError: diffusion matrix and stiffness sparsity differ` in one measured run; `--project=.` direct execution only works on this machine because GLMakie sits in the global `@v1.12` env.
- **Proposed shape**: light Diataxis site (≈ 20 pages, down from 25, no dev logs), reference split into 8 grouped `@docs` pages with `@id` headings, gallery = `examples/NN_name/{name.jl, README.md, make_figures.jl, assets/}` copied into `docs/src/examples/` at build time; assets committed (generation ≈ 5–7 min total on 1 thread, too costly for every docs build), sanity-linted by a cheap test.

## 2. Findings table

Severity: H = fix before docs go public, M = should fix, L = nit.

| ID | Sev | Area | Location | Evidence | Proposal | Benefit | Effort |
|---|---|---|---|---|---|---|---|
| D1 | H | Docs correctness | `docs/src/getting-started.md` "AMR state is similarly explicit" block (static ```julia) | Executed in scratch: `InterfaceRefinementIndicator(interface; level = 2)` → `ArgumentError: level must be an integer within mesh limits`. The mesh in the preceding block is `BlockTreeMesh(block_size=(16,16), periodic=(false,false))`; default `max_level = 0` (`src/mesh/topology/mesh.jl:841`). | Add `max_level = 2` to that mesh; turn the two-phase/AMR blocks into `@example` blocks so the build executes them. | Quick start works; drift caught by CI | S |
| D2 | H | Docs verification | all pages | Block census: 8 `@example` (getting-started 5, single-phase-tutorial 3); ~66 static ```julia; 0 `jldoctest`/`@repl`. README quick start is verified only by a hand-copied duplicate (`test/readme.jl`), other README snippets unverified. | Make Quick start, tutorials and how-tos `@example` with hidden shared setup (`#hide`); keep README quick start as the single source (extract it in `test/readme.jl` or point README to the site). | Docs cannot silently rot | M |
| D3 | H | Visuals | docs/ (whole) | `grep` of built HTML: 0 `<img>`; no `![` in `docs/src/*.md`; `docs/Project.toml` has only Documenter + TwoPhaseAMR. | Gallery + 3–4 figures in Quick start / model-choice / visualization (see §5, §6). | Core ask of the owner | M–L |
| D4 | M | Asset hygiene | `docs/src/assets/` (untracked) | 4.5 MB: `stage2-final-*.svg`, `cache-cliff/`, `julia-fortran-matched/` (raw.jsonl 445 KB…); referenced nowhere. Documenter copies every non-`.md` file under `docs/src` into the site. | Keep benchmark artefacts in `benchmark/` or `research/`; reserve `docs/src/assets/` for logo/CSS; example images live in `examples/*/assets/` (copied at build time). | Small deploys, no accidental publication | S |
| D5 | M | API pages | `api.md`, `api-functions.md` | Build log: `Generated HTML over size_threshold_warn limit: api-functions.md 101.56 KiB`, `api.md 117.27 KiB`; hard limit 200 KiB. Anchors are signature-based, e.g. `id="TwoPhaseAMR.simulate!-Union{Tuple{T}, Tuple{FlowSimulation{T}, Any}} where T"`, `TwoPhaseAMR.BlockTreeMesh-Tuple{}`. | Replace the two `@autodocs` dumps by 8 grouped pages with `@docs` lists and `## [Title](@id api-…)` headings (§4). Link the PDF to those `@id` anchors, never to docstring ids. | No size error as API grows; stable links | M |
| D6 | M | `checkdocs` | `docs/make.jl` | `checkdocs = :exports` passes (no missing docstrings in this build). | After the export→`public` split use `checkdocs = :public`; `Private=false` already includes `public` names on Julia ≥ 1.11 (`Documenter/src/expander_pipeline.jl:622`, `DocSystem.jl:326` use `Base.ispublic`). **Caveat**: package supports Julia 1.10 where `public` does not parse; needs `Compat.@compat public` or a version-gated include. | Documented surface = declared surface | S |
| D7 | M | Duplication | README / index / getting-started / how-to / choosing-a-model / api / examples | Two-phase snippet appears 4× (README, getting-started, choosing-a-model, how-to partially); option matrix 3× (README, choosing-a-model, api); legacy→new mapping 2× (api.md, migration.md); example table 3× (README, `examples/README.md`, `docs/src/examples.md`); the compat-window paragraph 3× (migration, api-decisions, api). | README: badges, 3-sentence pitch, install, 12-line quick start, link to site. index.md: pitch + figure + cards. One owner per fact (§3). | −35 % prose, no drift | M |
| D8 | M | Dev notes in user docs | `migration.md`, `api-decisions.md`, `literature-todos.md`, `architecture.md` | `migration.md`: "available throughout the `0.2.x` series, not removed before `0.3`" while `Project.toml` is `0.1.0-DEV` and no release exists; titles "Refactor API migration/decisions"; `architecture.md`: "The Phase 2 files communicate…". `literature-todos.md` is a 4.9 KB contributor TODO list ("TODO: Reference or derive…" ×27). | Delete migration + api-decisions from the site (history stays in git; start a `CHANGELOG.md` at the first tag). Move literature TODOs to the LaTeX reference (open-items appendix) or `TODO_LATER.md`/issues. Keep a short Developer guide. | Public site reads as a product, not a refactor log | S |
| D9 | M | Tutorial uses internals | `single-phase-tutorial.md` | `velocity_error` uses `flow.solver.diffusion.x.local_to_global`, `.volumes`, `flow.solver.mesh.faces`, `flow.velocity.interior` – undocumented struct fields. | Provide a public `l2_error(velocity, exact; time)`-style helper (or use an exported diagnostic) and use it in tutorial + examples. | Tutorial survives refactors | M |
| D10 | M | Hybrid pages | `visualization.md` (318 lines, 12 blocks), `adaptive-refinement.md` (180 lines, incl. references list), `advanced-workflows.md` | Tutorial + how-to + reference + method detail + allocation notes mixed; transfer-operator theory and bibliography in a user page. | Split per Diataxis (§3); theory/refs go to the PDF; keep session API in reference. | Findable, shorter | M |
| D11 | M | Missing pages | — | No Installation, FAQ/Troubleshooting, Citing (`CITATION.cff` absent), Changelog, user-facing "Extending". | Add (§3). Seed FAQ with: 20–60 s first-call latency (measured: first `FlowSimulation` run 22–65 s per two-phase example on 1 thread), `level must be an integer within mesh limits`, GLMakie not found, thread counts, headless rendering. | Fewer support questions | S |
| E1 | H | Example correctness | `examples/lid_driven_cavity.jl` | Measured: `run_example(max_level=0)` and `(max_level=1)` default T=15 → `result.steps == 1`, `times=[0.0, 15.0]`, 0.01 s / 0.26 s. Reason: at rest `stable_timestep` = Inf at t = 0 (printed `dt bound Inf`), so `simulate!` takes one step of the whole interval. With `dt = 0.05`: 300 steps, 0.46 s, KE(15) = 0.0329 vs 0.0625 for the single step. README/catalogue say "minutes". | Give the example a finite default step (e.g. `dt = 0.05` or a `max_dt`), or make `simulate!` cap the first step when the bound is infinite. Fix cost wording. | Example shows an actual transient | S |
| E2 | H | Example robustness | `examples/lgpi.jl` | Measured (first harness run): `run_example(max_level=2, final_time=0.96)` after a successful 3-step warm-up → `ArgumentError: diffusion matrix and stiffness sparsity differ`. Not re-traced in isolation (owner asked to wrap up); `test/examples/lgpi_example.jl` only checks the initial condition, `two_phase_examples.jl` only runs `max_level=0, 1e-6`. | Reproduce with `run_example(max_level=2)` and a backtrace; add a short `max_level≥1` smoke test (e.g. 5 steps) for every AMR example. | Examples with AMR are never run with AMR in tests | S (repro) / ? (fix) |
| E3 | M | Examples environment | `examples/*.jl` footers, `Project.toml` | Footer does `using GLMakie`; GLMakie is not in `Project.toml`, `Manifest.toml` or any `[extras]`; `~/.julia/environments/v1.12/Project.toml` has it (stacked `@v1.12`). `julia --project=. examples/x.jl` (README, examples/README) fails on a fresh machine. | Add `examples/Project.toml` (TwoPhaseAMR dev, GLMakie, CairoMakie) and document `julia --project=examples`. | Reproducible examples and figure script | S |
| E4 | M | Boilerplate | all 10 NS examples | `grep -l`: `checkpoints = 2…`, `asynchronous = true`, `GLMakie.activate` in 11/11 (+`manufactured_poisson`); `fetch(Threads.@spawn …)` 10/11; `for pass in 1:` initial adaptation loop 8/11; same 12-line `if abspath(PROGRAM_FILE)…` footer 12/12. | Owner wants scripts kept as-is → only move the *viewer tail and footer* into `examples/shared/live_viewer.jl` (`attach_viewer(flow; field, diagnostics…)`, `run_live(run_example; …)`), keep `run_example` signatures. Optional. | −300 lines | M |
| E5 | M | API smell | all examples | Keyword `visualize` shadows the exported function `visualize`, forcing `TwoPhaseAMR.visualize(...)` in every script. Return shapes differ: NS examples `(; flow, result, visualization)`, `vortex_reverse` `(; interface, velocity, result, visualization)`, `manufactured_poisson` a different tuple. `phase_rendering` only in `stokes3_wave`; `velocity_model` explicit only in `interfacial_instability`; 6 examples use `final_time=nothing` default, 4 a number. | Rename keyword (`live=false`) or accept as is; document the return contract once in the gallery index; give every two-phase example `velocity_model` and `phase_rendering`. Examples cannot take a headless `visualization=` observer: workaround verified (`run_example(final_time=0.0)` then `simulate!(flow, T; visualization=session)`). | Consistent gallery, headless figures without edits | M |
| E6 | M | Internal APIs in examples | `cylinder_flow.jl:76-90`, `manufactured_poisson.jl:115,153,249` | `TwoPhaseAMR._local_plic_segment`, `_get`, `_uncut_face_dual_volumes`, `_mimetic_face_data`. | Export a built-in `PressureForce`/`PressureDrag` diagnostic (drag is the cylinder's validated output); keep poisson verification in `test/` or use public helpers. | Examples survive refactors | M |
| E7 | M | Test coverage of examples | `test/examples/*.jl` | Every example run once with `max_level=0`, `final_time=1e-6`, `steps == 1`; the `visualize=true` branch, AMR (`max_level>0`), multi-step behaviour are never exercised (E1, E2 would be caught). Measured sum of first-call times in one process ≈ 2–3.5 min on 1 thread (JIT dominated; first-call 6–65 s each). | Add per-example `docs-size` smoke (the same config `make_figures.jl` uses, T small) plus a headless-viewer smoke with `NoVisualization`-session; keep the 1e-6 tests. | Gallery and tests share configs | S |
| E8 | L | Stale docstring | `examples/manufactured_poisson.jl:318-320` | Signature says `method=:commuting`; code accepts only `:cmdgfm, :mimetic, :joint_face` (lines 364-379), default `:mimetic`. | Fix signature. | — | S |
| E9 | L | Catalogue consistency | `README.md`, `docs/src/examples.md`, `examples/README.md` | README and `examples.md` list 11 examples, `examples/README.md` lists 12 (adds `manufactured_poisson`). | Single generated table (§5). | — | S |
| E10 | M | Cost table accuracy | `examples/README.md`, `docs/src/examples.md` | Qualitative "seconds/minutes/long/very long", no hardware/threads. Measured warm (1 thread, after JIT): taylor_green 0.0 s (7 steps), lid_cavity 0.26 s (1 step!), capillary L=2 1.3 s (257 steps), sessile L=2 1.5 s, sloped L=2 6.6 s, stokes3 L=3 7.0 s, vortex_reverse L=3 37 s (1661 steps, 64 regrids). Defaults (`max_level` 3–5) are larger; first call costs 12–65 s of JIT regardless. | Table of measured wall-clock for default and docs-size, with "includes ~30 s first-call compile" note. | Honest expectations | S |
| E11 | M | Prosperetti dependency | `examples/capillary_wave.jl:3-8` | `Base.include(…, "test","integration","prosperetti_amplitude.jl")`; also used by `test/integration/two_phase_waves.jl:3` and `test/examples/two_phase_examples.jl:51`. | Move file to `examples/06_capillary_wave/prosperetti_amplitude.jl`; tests include it from there (tests→examples is already the direction of `test/examples/helpers.jl`). | Examples self-contained | S |
| W1 | L | Workflow | `.github/workflows/Documentation.yml` | Builds on PR with `push_preview`; no `linkcheck`, no doctest step, no CairoMakie/Literate in `docs/Project.toml`, no `[sources]`; relies on `Pkg.develop` each run (fine). No local-build instructions anywhere (`julia --project=docs docs/make.jl`). `prettyurls` correctly keyed on `CI`. | Add `docs/README` line / `[sources] TwoPhaseAMR = {path = ".."}` (docs run on Julia ≥ 1.11 only); set `inventory_version`/`repo` explicitly; optional weekly `linkcheck`. | Reproducible local builds | S |
| W2 | L | Env observation | this machine | Julia 1.12.7 aborted twice during JIT (once in LLVM `llvm-remove-addrspaces`/`CloneFunctionInto` while compiling `capillary_wave.jl` `run_simulation`, once at top level of a PoC); both unreproducible on rerun. Not attributable to the repo. | None; if CI shows it, retry the docs job. | — | — |

## 3. Details (high/medium) and proposed site

### 3.1 Docs build result (item 1)

Command (scratch copy, `deploydocs` removed, `remotes = nothing` because /tmp is not a git repo, package `Pkg.develop`ed): `julia -t 1 --project=. make.jl` in `/tmp/audit-docs/docs` (CI=true in this shell → `prettyurls`).

- Result: success, **46.3 s** wall. Warnings: (a) size threshold ×2 (real); (b) navbar repo link and `inventory_version` (artefacts of the /tmp copy; in the repo they resolve via git remote and `../Project.toml`).
- No missing docstrings under `checkdocs = :exports`, no cross-reference failures. Page HTML under `<name>/index.html`.
- Executed blocks: 8; static blocks: ~66; math blocks: 51 (navier-stokes 11, staggered 15, two-velocity 12, poisson 7, pressure-gradient 5, tutorial 1).

### 3.2 Snippet runs (README + getting-started)

Harness `/tmp/audit-docs/snippets.jl` (each ```julia / ```@example block `include_string`ed in order per file, `-t 1`):

| File | Block | Result |
|---|---|---|
| README | 1 quick start (Taylor–Green 16×16, 2 steps) | OK (24.7 s, JIT) |
| README | 2 two-phase construction | OK (19.2 s, JIT) |
| README | 3 `add_diagnostic!` + `simulate!(flow, 0.1)` (acts on the two-velocity flow from block 2) | OK (44.1 s) |
| README | 4 `VisualizationSession(…; viewer=NoVisualization())` | OK (15.1 s) |
| README | 5 `include("examples/capillary_wave.jl")` … | harness artefact (`include` undefined in my sandbox module); **not verified**. Same call pattern (`velocity_model=TwoVelocityModel()`) is exercised by `test/examples/two_phase_examples.jl` (1e-6) and ran in my measurements (capillary, `max_level=2`). |
| getting-started | blocks 1–8 (incl. summary print) | OK, 0.0–1.2 s each |
| getting-started | **block 9** (`InterfaceRefinementIndicator(interface; level = 2)`) | **FAILED**, see D1 |

README observation: block 3 mutates the flow of block 2 and the `state.solver.mesh.blocks` / `interior(state.pressure, block)` pattern leaks internals; a public `maximum(abs, state.pressure)`-style reduction would be simpler.

### 3.3 Proposed page tree (Diataxis)

```
Home                               index.md            (pitch, 1 figure, 5-line quick start, cards)
Getting started
  Installation                     installation.md      NEW (registered? clone+instantiate, GLMakie/CairoMakie env, threads)
  Quick start                      getting-started.md   (all @example; first figure)
  Choosing a model                 choosing-a-model.md  (decision table, links to PDF chapters)
Tutorials
  Single-phase flow                single-phase-tutorial.md   (Taylor–Green, executed, uses public error helper)
  Two-phase flow                   two-phase-tutorial.md      NEW (capillary wave vs Prosperetti, executed, 2 figures)
  Adaptive refinement              adaptive-refinement.md     (trimmed to criteria + usage, ≤ 80 lines)
How-to guides
  Collect diagnostics              diagnostics.md             (table of built-ins + custom; definitions → PDF)
  Configure numerics and solvers   numerics.md                (how-to.md "methods" + advanced-workflows "solver composition")
  Live visualization               visualization.md           (viewer + sessions, ≤ 150 lines; rest → reference docstrings)
  Static figures and animations    figures.md                 NEW (CairoMakie, snapshot_figure, GIF)
  Threads and performance          performance.md             (short; links PDF; allocation contract)
  Extend the package               extending.md               (merge adding-an-equation-system + custom criteria/diagnostics)
Examples (gallery)                 examples/index.md + examples/NN_*.md   (generated, §5)
Explanation (short)
  Method overview                  method-overview.md         (1 page, links to PDF chapters via named destinations)
  Scope and limitations            scope.md                   (from README/index "Scope")
Reference
  API overview                     api/index.md               (layer table + option matrix, once)
  api/mesh-amr, api/fields-operators, api/physics-models, api/navier-stokes,
  api/interface-geometry, api/diagnostics, api/visualization, api/advanced
Help
  FAQ and troubleshooting          faq.md                     NEW
  Citing                           citing.md                  NEW (thesis DOI + CITATION.cff)
  Changelog                        changelog.md               NEW (include CHANGELOG.md via @eval/@raw or copy)
Developer guide                    developer/{architecture,testing,contributing}.md (merge testing + adding-an-equation-system)
```

Cut / merge:

| Page | Action |
|---|---|
| `two-phase-flow`, `two-velocity-advection`, `staggered-advection`, `pressure-gradient`, `poisson`, `navier-stokes`, `shared-memory-parallelism` (≈ 77 KB) | move to PDF; keep one `method-overview.md` (+ a 3-line summary box on `choosing-a-model.md`). Math-heavy; the 51 math blocks leave the site. |
| `migration.md`, `api-decisions.md` | delete from site (0.1.0-DEV has nothing to migrate; version text says `0.2.x`). `CHANGELOG.md` from first tag. |
| `literature-todos.md` | PDF appendix "open items" or `TODO_LATER.md`. |
| `api.md` "Migration from legacy" + `api.md` option matrix | option matrix stays once in `api/index.md`; legacy table removed. |
| `advanced-workflows.md` | split: explicit-object construction → `extending.md`; solver composition → `numerics.md`; adaptive snippet duplicates `adaptive-refinement.md`; allocation contract → `performance.md`. |
| `how-to.md` | dissolve into diagnostics / numerics / visualization / examples pages. |
| `examples.md` + `examples/README.md` + README table | replaced by generated gallery index. |
| `architecture.md` "Phase 2" paragraph | reword; keep dependency direction + source ownership. |

Links to the PDF: ship the PDF into the site per version (`docs/src/assets/technical-reference.pdf` copied by CI before `makedocs`, so `/v0.1/` links its own PDF); link chapters with PDF named destinations (`…/technical-reference.pdf#nameddest=sec:two-velocity`, from `\label`/hyperref). Links from the PDF to the site use page-level `@id` anchors (`…/stable/api/diagnostics/#api-diagnostics`), never docstring ids. Until the first tag there is no `/stable/`: use `/dev/` and switch once; set `HTML(canonical="…/stable/")`.

### 3.4 API reference structure (item 2)

- 8 pages under `reference/` (or `api/`), each: intro paragraph, one `## [Group](@id api-<group>-<topic>)` heading per topic, then `@docs` listing explicit names (`@docs BlockTreeMesh MeshAdaptationOptions …`). Expected size per page 15–30 KiB, far below the 100 KiB limit.
- `api/advanced.md` for names that become `public` but unexported (still `@docs` listed so `checkdocs = :public` passes).
- `Private=false` keeps working with `public` names on Julia ≥ 1.11; `@docs` lists make the ordering explicit (alphabetical `@autodocs` currently interleaves types and functions; the types page `api.md` lists types/constants after the prose).
- Keep `@docs TwoPhaseAMR` (module docstring) on the overview page only.
- `MimeticProjectionSolver` missing docstring is not caught here because it is not in the exports covered by this build (known finding; not re-checked).

## 4. Visual gallery (items 3 and owner's folder-structure decision)

### 4.1 Headless rendering support today

| Capability | Status | Evidence |
|---|---|---|
| `visualize(registry; display=false)` + `save` with CairoMakie | works; contains Field menu, Checkpoint menu, Live/Pause buttons, checkbox rows, empty `No cell or face selected` label, colorbar detached far right (layout shared with widgets) | `/tmp/audit-docs/poc/tg_viewer_headless.png`, `capillary_snapshot_viewer.png`; 10.6 s first call (JIT) + 2.5 s save |
| Interface line (PLIC) overlay in a static figure | yes (`phase_rendering=:plic`, black line visible) | `capillary_snapshot_viewer.png` |
| Retained frames without a viewer | yes: `VisualizationSession(fields; viewer=NoVisualization(), checkpoints=5)`, `simulate!(flow, T; visualization=session)`, `visualization_checkpoints(session)[i].frame[:name]`, `.frame.topology.cell_geometry` | PoC: 5 checkpoints in 6.8 s incl. JIT; checkpoints are *nearest completed states* (times 0.0, 0.12, 0.18, 0.3, 0.4 for requested 0, .1, .2, .3, .4) |
| Clean static plot from exported pieces | works with ~25 lines (`Makie.mesh!` over `visualization_geometry(mesh)` + cell values + `linesegments!` for blocks) | `/tmp/audit-docs/poc/tg_static_pressure.png` (build 1.7 s, save 0.24 s); `tg_checkpoints.png` (5-panel, 4.3 s + 2.0 s) |
| GIF | works (`record`, FFMPEG_jll available): 5 frames, 131 KB, 1.7 s | `/tmp/audit-docs/poc/tg.gif` |
| Time series from `diagnostic_history` | trivial (`lines!(ax, h.times, h.values)`) | `/tmp/audit-docs/poc/capillary_amplitude.png`: numerical vs Prosperetti, last value 0.00637 vs 0.00615 at `max_level=1` |
| Interface overlay / solid geometry / diagnostics panel from a *frame* | not exported: `_interface_overlay_points(registry, frame)`, `_combined_solid_geometry` are ext-internal | `ext/TwoPhaseAMRMakieExt.jl:258-297` |
| Headless `visualization=` observer through `run_example` | not possible (kwargs go to `FlowSimulation`); workaround `run_example(final_time=0.0)` then `simulate!(flow, T; visualization=session)` verified for Taylor–Green | `poc2.jl` |

**Proposal G1 (M)**: add to the Makie ext a small static API, reusing `_display_field!`, `_makie_mesh`, overlay helpers: `snapshot_figure(source; field, frame=nothing, blocks=true, interface=true, geometry=true, colorbar=true, size=(600,500))` → `Figure` (no widgets); `plot_diagnostics!(ax, flow, names; analytical=…)`; `record_checkpoints(path, session; field, framerate)` → GIF/MP4. `source` = registry or `VisualizationFrame`. Alternatively add `controls=false` to `visualize`. Benefit: every `make_figures.jl` shrinks to ~10 lines.

Measured costs on 1 thread: `using CairoMakie` 2.9 s; first widget-free plot JIT ≈ 2–4 s; first `visualize(display=false)` 10.6 s.

### 4.2 Folder layout, naming, numbering (owner's decision)

```
examples/
  Project.toml              TwoPhaseAMR (dev), GLMakie, CairoMakie   (Manifest gitignored)
  README.md                 generated table + conventions (see 4.5)
  shared/                   figures.jl (CairoMakie helpers; later snapshot_figure), live_viewer.jl (optional E4)
  make_all_figures.jl       loops over folders, `--only 06,07`, `--size docs|default`
  01_taylor_green/          taylor_green.jl  README.md  make_figures.jl  assets/
  02_lid_driven_cavity/
  03_decaying_turbulence/
  04_cylinder_flow/
  05_vortex_reverse/        (VOF only, prescribed carrier velocity)
  06_capillary_wave/        + prosperetti_amplitude.jl (moved from test/integration)
  07_interfacial_instability/
  08_sessile_droplet/
  09_sloped_sessile_droplet/
  10_stokes3_wave/
  11_lgpi/
  12_manufactured_poisson/  (projection verification, not time-dependent; or move to test/verification)
```

Script keeps its name `<name>.jl` and `run_example` API unchanged (`include("examples/06_capillary_wave/capillary_wave.jl")`). Order: single phase (01–04), VOF-only transport (05), flat-interface two-phase with analytic references (06–07), contact line / embedded solid (08–09), nonlinear waves and large benchmark (10–11).

### 4.3 Per-example markdown template

```
# NN – Title
> one-sentence what you will see
![hero](assets/hero.png)                       # or hero.gif
## At a glance
| | |
|---|---|
| **Physics** | single/two-phase; density & viscosity ratio; σ, g, contact angle, embedded solid; dimensionless numbers; domain & BCs |
| **Algorithms** | velocity model; interface transport; pressure projection; surface-tension method; curvature/reconstruction; momentum advection; AMR criteria; linear solvers |
| **Worth varying** | keyword = default → what changes |
| **Validated against** | reference + diagnostic name + observed error at docs size |
| **Cost** | default size (time, cells); docs size (time, cells); "+ ≈ 30 s first-call compile" |
## Setup            (equations/params in 5–10 lines, link to PDF chapter)
## Run              (headless REPL snippet + GLMakie command with `--project=examples`)
## Results          (figures with 1-line captions; numbers)
## Variations       (2–3 runnable one-liners: other velocity model, finer level, other scheme)
## Files            (script, make_figures.jl, extra files)
```
README stays plain GitHub-flavoured markdown (static ```julia, ```math, relative images) so it renders on GitHub *and* in Documenter; no `@example` in it.

### 4.4 "At a glance" demonstrated

**01_taylor_green** (`examples/taylor_green.jl`)

| | |
|---|---|
| Physics | single phase, ρ=1, ν = 1/Re = 0.02 (Re = 50); periodic [0, 2π]²; no gravity/surface tension; exact decaying vortex u = e^{−2νt} sin x cos y |
| Algorithms | `FaceVelocity`; `AdvectionOptions` default `LW()` (printed model: LaxWendroff); implicit-midpoint diffusion (CG, tree-geometric multigrid preconditioner); pressure projection CG + tree GMG (printed: `cg`, `tree_geometric_multigrid`); static mixed-level mesh (`refine`: block (0,0) refined to `max_level`), no dynamic criteria; auto dt (CFL 0.45) |
| Worth varying | `reynolds=50`, `max_level=2` (≥ 2 needed for coarse–fine faces), `final_time=0.2` (too short to see decay; use 5), `numerics=NavierStokesOptions(advection=AdvectionOptions(scheme=MC()))`, `dt=` fixed step |
| Validated | analytic velocity/KE decay — **the script does not compute an error today** (diagnostics: KE, momentum, divergence norm, leaf blocks); the tutorial page does. Add an `analytical_kinetic_energy` diagnostic. |
| Cost | default: 1008 cells, 7 steps, 0.04 s warm setup+run; docs size T=5: ≈ 170 steps ≈ 2 s (estimate); first call 27 s JIT |

**06_capillary_wave** (`examples/capillary_wave.jl`)

| | |
|---|---|
| Physics | two phase, ρ ratio 1000 (`density_ratio`), Laplace number 3000 → ν = √(σλ/La) ≈ 0.018, σ = 1, λ = 1, amplitude 0.01 (ka ≈ 0.063), no gravity; periodic in x, `FreeSlip` in y; domain 1×3 |
| Algorithms | one-velocity model (default; `TwoVelocityModel()` accepted); surface tension numeric shorthand → `CGFM`; `InterfaceField` with `lipschitz=1`, reconstruction `:lhf` (height functions, WY fallback), WY geometric VOF transport; advection `LW()`; AMR `InterfaceRefinementIndicator(level=max_level)`, `MeshAdaptationOptions(update_interval=5, coarsen_delay=2)`; CG + tree GMG for pressure/diffusion |
| Worth varying | `max_level=2`, `laplace` (damping), `amplitude` (nonlinear regime), `density_ratio`, `velocity_model=TwoVelocityModel(...)`, `numerics`, `adaptivity` |
| Validated | Prosperetti (1981) amplitude: diagnostics `:amplitude` vs `:analytical_amplitude` (max_level=1: 0.00637 vs 0.00615 at t = 0.399, one period) |
| Cost | default (`max_level=2`, T = one period 0.4): 960 cells, 257 steps, **1.3 s** warm; first call 65 s JIT (two-phase). `max_level=1`: 91 steps. |

**07_interfacial_instability** (`examples/interfacial_instability.jl`)

| | |
|---|---|
| Physics | two phase, ρ ratio 3, We = 0.5, Bo = 0.75, inviscid (`dynamic_viscosity = (0,0)`), tangential velocity jump (KH) + gravity (RT); σ = (ρ₁+ρ₂)/k³; periodic x, `FreeSlip` y; linear growth rate √(We+Bo−1) = 0.5, run until amplitude doubles (t ≈ 2.63) |
| Algorithms | **two-velocity** (default): `FrozenInterfaceFluxes` + `CMDGFMProjection`; `OneVelocityModel()` gives a density-weighted-mean closure (different physics); `CGFM`; reconstruction `:lhf`; `LW()`; AMR `InterfaceRefinementIndicator(level=max_level)`; liquid/gas initial velocities passed as a tuple |
| Worth varying | `velocity_model`, `weber`, `bond` (sign of 1−We−Bo switches cos/cosh), `density_ratio`, `max_level`, `final_time` |
| Validated | linear theory `cosh(√(−r) t)` / `cos(√r t)`: `:amplitude` vs `:analytical_amplitude` |
| Cost | default `max_level=1`: not measured (run killed during JIT at wrap-up) — **estimate**: similar to capillary (a few seconds warm, 30–65 s first call) |

### 4.5 Asset production, pickup, tests, tables

- **Production**: per-folder `make_figures.jl` (defines `make_figures(; outdir="assets", size=:docs)`) + shared `examples/make_all_figures.jl`; run **manually** (`julia --project=examples examples/make_all_figures.jl --only 06`) with CairoMakie, committed PNG (≤ 150 KB, 900 px) and GIF (≤ 600 KB, ≤ 30 frames, via checkpoints). Measured total for all 11 at docs size ≈ 5–7 min on 1 thread (dominated by 12–65 s JIT per example; shared methods amortise partially) → too slow for each docs build, and Actions are blocked; CI only lints (below). Build-time execution only for the Quick start / two tutorials (seconds). Cache alternative if CI later allows: cache `examples/*/assets` keyed on hash of `src/**`, `examples/**`; regenerate in a scheduled workflow and open a PR.
- **Pickup in Documenter** (`docs/make.jl`, before `makedocs`): for each `examples/NN_name/README.md` write `docs/src/examples/NN_name.md` with rewritten links (`assets/x.png` → `assets/examples/NN_name/x.png` after copying `assets/` into `docs/src/assets/examples/NN_name/`; `[script](name.jl)` → `https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/main/examples/NN_name/name.jl` (or tag via `Documenter.Remotes`); `../NN_other/` → `NN_other.md`); generate `docs/src/examples/index.md` (table from front-matter / `examples/examples.toml`: id, title, model, size, cost) and the `pages` list from `readdir`. `docs/src/examples/` and `docs/src/assets/examples/` are gitignored.
- **Tests** (`test/examples/helpers.jl`): `load_example(name)` resolves `examples/*_$name/$name.jl` (`only(filter(isfile, …))`); update `test/examples/*.jl` callers unchanged. Add a cheap lint testset: every folder has `README.md` with the template headings, every `![](assets/…)`/link target exists, and every `run_example` default is documented. Add docs-size smoke runs sharing `make_figures` configs (E7).
- **Tables**: README.md and docs index link to the gallery instead of duplicating; `examples/README.md` generated from `examples.toml` (one source) with measured costs; `examples/README` ownership of commands (`--project=examples`).
- **Prosperetti** (E11): `examples/06_capillary_wave/prosperetti_amplitude.jl`, `include(joinpath(@__DIR__, …))` from the script; `test/integration/two_phase_waves.jl` and `test/examples/two_phase_examples.jl` include/consume it from there.

## 5. Gallery plan per example (docs size = proposed small size)

Measured = 1 thread, warm, wall-clock of `run_example(max_level=L, final_time=T)` after a first run in the same process (first-call JIT in the last column). Not measured → estimate.

| # | Example | Figures (hero first) | Docs size (L, T) | Measured warm | Steps / cells | First-call JIT | Notes |
|---|---|---|---|---|---|---|---|
| 01 | taylor_green | vorticity snapshots t=0, T/2, T (3-panel PNG, block grid); KE vs exact `exp(-4νt)` | L=2, T=5 | T=0.2: 0.0 s (7 steps, 1008 cells); T=5 est. ≈ 2 s | 7 steps @0.2 | 27 s | add analytical KE diagnostic; mixed-level mesh visible at L=2 |
| 02 | lid_driven_cavity | velocity magnitude + streamlines-ish arrows at steady state; KE vs t | L=1, T=15 with `dt=0.05` | default (1 step, dt=15): 0.26 s; dt=0.05: 0.46 s (300 steps) | 1024 cells | 22 s | **fix E1 first**; compare u(0.5,y) to Ghia et al. (reference data not in repo) |
| 03 | decaying_turbulence | vorticity + AMR block outlines at t=0 and end (2-panel); KE and cell count vs t; GIF (≤ 25 checkpoints) | L=2, T=2 | 0.6 s | 787 steps, 1024 cells, 0 regrids | 6 s (after other single-phase) | at L=2 AMR never triggers (min level 2 = max) → use L=3 (est. ≈ 5–10 s) to show coarsening |
| 04 | cylinder_flow | vorticity + embedded solid + blocks; pressure drag vs t | L=2, T=10 (default 3 / 40) | not measured (killed during JIT) | est. ≈ 10–40 s | est. 30–60 s | needs public drag diagnostic (E6); Strouhal shows only for T ≳ 40 |
| 05 | vortex_reverse | volume fraction + interface at t=0, T/2, T (3-panel) and initial vs final overlay; volume error | L=3, T=8 (default L=5) | 37.1 s | 1661 steps, 1792 cells, 64 regrids, vol_err 4.8e-11 | 26 s | reduce to L=2 or T=4 for < 10 s; shape recovery is the validation |
| 06 | capillary_wave | amplitude vs Prosperetti (verified PoC); interface+volume fraction snapshot; AMR blocks | L=2, T=one period (0.4) | 1.3 s | 257 steps, 960 cells | 65 s | clean snapshot needs G1; one-vs-two-velocity overlay in the amplitude plot |
| 07 | interfacial_instability | amplitude vs linear theory; interface shape at t=0, T; one- vs two-velocity comparison | L=1, T=2.63 | not measured | est. few s | est. 30–60 s | two runs (both models) double the cost |
| 08 | sessile_droplet | interface + velocity vectors/KE at t = 0, 4 t_c, 8 t_c; KE and phase volume vs t; contact angle | L=2, T=0.716 (8 t_c) | 1.5 s | 460 steps, 640 cells | 13 s | spurious-current level visible in KE |
| 09 | sloped_sessile_droplet | embedded wall + interface snapshots; KE vs t | L=2, T=0.256 | 6.6 s | 421 steps, 640 cells | 28 s | needs `register_geometry!`; solid overlay not exported for frames (G1) |
| 10 | stokes3_wave | interface profile vs analytic `elevation(x)` after one period; vorticity with AMR blocks (3 criteria); GIF | L=3, T=2π/ω ≈ 2.2 (default L=5) | 7.0 s | 251 steps, 15 regrids, 2368 cells | 12 s | analytic elevation is in the script (`stokes3_initial_condition`) |
| 11 | lgpi | interface + bathymetry + vorticity at 3 times (3-panel); GIF | L=2, T=0.6 gravity periods ≈ 0.96 | setup+1 step 0.83 s, 163 ms/step (3792 cells); **full run failed (E2)** | est. ≥ 20 steps | 11 s | blocked by E2 |
| 12 | manufactured_poisson | pressure error + gradient error maps (2-panel) for the three methods; error table | `resolution=32` (default) | not measured | — | — | verification, not a flow; consider `test/verification` |

Aggregate: measured warm total (rows 01–03, 05, 06, 08–11 with L as above) ≈ 55 s of simulation + ≈ 210 s summed first-call JIT (partially overlapping) + CairoMakie (3 s load + ≈ 15 s first plots) ⇒ **≈ 5–7 min for the whole gallery on 1 thread** (estimate; rows 04, 07, 12 not measured). Hence committed assets + manual regeneration.

## 6. Other recommendations (docs content)

1. Quick start (`getting-started.md`): all blocks `@example`; end with one figure (vorticity at final time via G1) and the diagnostic plot.
2. `choosing-a-model.md`: add a decision flowchart-style table; embed 2 images from the gallery (one-velocity vs two-velocity interfacial instability).
3. README: ≤ 90 lines; hero GIF; quick start verified by `test/readme.jl` extraction; link table → gallery.
4. FAQ seeds: first-call latency; `--threads`; headless (`CairoMakie.activate!()`); async visualization starvation (known finding) → link to the examples' `fetch(Threads.@spawn …)` pattern until fixed; "level must be an integer within mesh limits".
5. Versioning: `deploydocs(versions = ["stable" => "v^", "v#.#", "dev" => "dev"])` (defaults), badges point to `/stable/` after the first tag; before that keep `/dev/`.

## 7. Nits

- `docs/make.jl`: `edit_link = "main"` is fine; add `inventory_version`/`repo = Remotes.GitHub(...)` explicitly; consider `warnonly = [:linkcheck]` only if `linkcheck = true` is enabled.
- `docs/src/architecture.md` "Phase 2 files" and "refactor" wording in `api-decisions`/`migration` titles.
- `how-to.md` snippets assume undefined `mesh`, `physics`, `interface`, `tracer`, `final_time` (by design, not executable).
- README claims Julia 1.10 support; docs and gallery run on latest Julia only (state it).
- `examples/README.md` says "`manufactured_poisson` … `max_level`" parameters differ (it takes `resolution`), not part of the 'max_level/final_time/visualize' contract stated at the top of the same file.
- PoC artefacts for reference: `/tmp/audit-docs/poc/{tg_viewer_headless.png, tg_static_pressure.png, tg_checkpoints.png, tg.gif, capillary_amplitude.png, capillary_snapshot_viewer.png}`; scripts `/tmp/audit-docs/{snippets,poc1,poc2,poc4,poc5,poc6}.jl`; build log `/tmp/audit-docs/build.log`.
- Not done (owner asked to wrap up): measured warm cost of `interfacial_instability`, `cylinder_flow`, `lgpi` (full run), `manufactured_poisson`, default-size runs of all examples; isolated repro of E2; README block 5.

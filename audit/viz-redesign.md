# TwoPhaseAMR visualization redesign: "Studio" proposal

Status: design proposal for owner review, 2026-11-08. No code changed.
Scope: `ext/TwoPhaseAMRMakieExt.jl`, `src/visualization/**`, `docs/src/visualization.md`,
`test/makie_visualization.jl`, the `visualization` test group, and the examples that use them.
Library facts were checked against current documentation and sources; Appendix A lists the URLs.
Statements marked **[INFERENCE]** have not been checked yet and are flagged for Phase 1.

---

## 0. Executive summary

**Recommendation.** Replace the GLMakie viewer with a browser-based **TwoPhaseAMR Studio**:

- **Front end (TypeScript).** [dockview](https://dockview.dev) handles docking: tabs, splits,
  drag-and-drop, floating and pop-out groups, and layout JSON. A custom **WebGL2 AMR renderer**
  draws the fields. [uPlot](https://github.com/leeoniya/uPlot) draws the diagnostics.
- **Back end (Julia).** A small HTTP/WebSocket server built on HTTP.jl. HTTP.jl 2.x spawns its
  connection handlers on Julia's `:interactive` threadpool, so the server stays responsive while
  the solver keeps every default thread busy.
- **Makie keeps one job.** Makie (CairoMakie when headless) becomes the only code path for static
  figures and GIF/MP4 export. Scripts, CI, and the Studio's export button all use it.

**Five ideas carry the design:**

1. **Separate recording from viewing.** A `Recorder` is the observer passed to `simulate!`. It
   copies owned frames into a bounded `SnapshotStore`:
   - keyframes are scheduled and never dropped;
   - preview frames are thinned within a memory budget, and when the viewer falls behind only the
     newest one is shown.

   The solver never waits for a viewer. View state (live or history, selected time, cameras,
   colour ranges) lives in each client, not in the session.
2. **Draw AMR blocks as textures, not triangles.** Leaf blocks are uniform `nx×ny` tiles, and
   `visualization_values` already returns values block by block
   (`src/visualization/sampling.jl:1-19`). So each block becomes one instanced quad, its values go
   into a texture atlas, and the colormap is applied in the fragment shader. The sharp
   (PLIC/solid) display is done with per-cell plane tests in the shader. A field on 10⁶ cells costs
   4 MB of Float32 data and 4 vertices per block. Today it costs 4 private vertices and 2 triangles
   per cell.
3. **Keep one source of truth for diagnostics.** The plots read the simulation-owned
   `DiagnosticCollection`. It gains metadata: label, units, physical dimension, component names, and
   scale. The default layout gives each dimension its own panel (owner requirement 4).
4. **Use a camera that fills the panel.** The camera is a centre plus one isotropic scale. Zooming
   changes the limits and never shrinks the axis box (owner requirement 2). Makie export uses the
   same semantics via `Axis(aspect = nothing, autolimitaspect = 1)`.
5. **Export from one code path.** `export_animation` builds the figure once, binds the frame to an
   Observable, and defaults to a global colour range. This removes the per-frame axis rebuild and
   the per-frame colour range by construction.

**Owner requirements and where they are met:**

| # | Requirement | Design element | Phase |
|---|---|---|---|
| 1 | Several views; panels closable, resizable, movable | dockview layout of field, diagnostics, probe, stats, and table panels; layouts saved as JSON | 3 |
| 2 | Zoom uses the available space, cells stay square | camera (centre, scale); Makie `autolimitaspect = 1` | 2 (static), 3 (Studio) |
| 3 | GIF/video from stored frames: one panel plus optional diagnostics with a synced time cursor | `export_animation` (scripted) and the Studio export dialog, which calls the same function | 2 (script), 3 (dialog) |
| 4 | Diagnostics in separate panels, grouped by physical dimension | `DiagnosticInfo.dimension`; auto-layout groups by (kind, dimension, scale) | 1 (metadata), 3 (UI) |

All four are met at the end of Phase 3. Phase 4 deletes the old viewer, `VisualizationSession`,
the triangulation code, and the viewer-local diagnostics.

---

## 1. Assessment of the current design

### 1.1 Data flow today

```
simulate!  (src/navier_stokes/simulate.jl:70-74, 159-163, 175-179, 183-187)
   │  update_visualization!(session; event, time, step, velocity, pressure, solver)
   ▼
VisualizationSession  ── holds session_lock for the WHOLE update (updates.jl:102-106)
   ├─ exact side-step solves for off-grid checkpoints   (frame_capture.jl:287-332)
   ├─ diagnostic reductions → own history               (frame_capture.jl:204-227)
   ├─ _capture_frame: copies + derived values + full
   │  triangulated topology on regrid                   (frame_capture.jl:1-112)
   ├─ checkpoints: nearest-state selection, unbounded    (frame_capture.jl:114-142)
   └─ live frame → LatestFrameMailbox (capacity 1)       (checkpoints.jl:31-108)
                         │  @async publisher on the creating thread (session_types.jl:308-347)
                         ▼
MakieAMRVisualization (one figure, one field axis, one diagnostics axis, menus;
                       ext/TwoPhaseAMRMakieExt.jl:6-63, 570-932)
```

### 1.2 What to keep: principles and data model

| Keep | Evidence | Why |
|---|---|---|
| Numerics free of graphics; observer dispatch with a zero-cost default | `AbstractVisualization`/`NoVisualization` (`src/visualization/types.jl:1-12`); Makie only as a weak dependency (`Project.toml:16-24`) | Headless runs and light dependencies |
| Named field registry with label, units, colormap, range, and default registrations | `registry.jl:37-57, 292-417`; `register_default_fields!(registry, flow)` (`src/navier_stokes/flow_simulation.jl:494-527`) | Good vocabulary for users; keep the `register_*` functions |
| Frames own their data and stay valid after AMR | `VisualizationFrame` docstring (`checkpoints.jl:119-134`); `frame_capture.jl:103-110` | Necessary for history and export |
| Topology shared while the leaf set is unchanged | `_session_topology!` (`frame_capture.jl:12-29`) | Saves memory |
| Keyframe schedules by count, interval, or explicit times (incl. initial/final); nearest-state vs exact side-step | `session_types.jl:1-37, 78-128, 233-280`; `frame_capture.jl:114-142, 287-332` | Matches "snapshots at certain time intervals"; exact side-steps are used by tests (`test/navier_stokes/two_phase_navier_stokes.jl:419-462`) |
| Transient frames are latest-wins; scheduled ones are lossless | `LatestFrameMailbox` (`checkpoints.jl:31-108`) | The solver never waits on consumers |
| Cooperative pause at completed-step boundaries | `controls.jl:28-108`; `_pause_at_step_boundary!` (`session_types.jl:369-379`) | Safe solver control |
| Phase display semantics: majority (≥ 50 % liquid of open volume) or sharp PLIC | `docs/src/visualization.md:89-140` | Physically meaningful defaults |
| Only owned values, block by block; no halos | `sampling.jl:1-19` | Maps directly onto texture tiles (§4.6) |
| `simulate!` knows nothing about control state; four events | `simulate.jl:51-53, 70-74, 159-163, 175-187` | Clean seam between numerics and observers |
| Simulation-owned diagnostics with their own schedules | `src/simulation/diagnostics/history.jl:4-34, 115-173` | The right single source of truth |

### 1.3 What is clunky or defective

| # | Problem | Evidence | Consequence | Fix in the redesign |
|---|---|---|---|---|
| C1 | Monolithic single-view viewer: one axis shows one field at a time, chosen from a menu; 57-field struct | `ext/TwoPhaseAMRMakieExt.jl:6-63`, `visualize` 570-932, field menu 748-751 | No side-by-side views; switching fields loses context | Docked panels (§3) |
| C2 | `aspect = DataAspect()` locks the axis box to the aspect of the data limits; zooming shrinks the box | `ext:649` (viewer), `ext:1110` (snapshot); examples work around it: `axis.aspect[] = nothing; xlims!(…)` (`examples/06_capillary_wave/make_figures.jl:48-51`, `07…:52-55`, `08…:33-36`) | Most of a widescreen window stays empty after zooming (requirement 2) | Camera model; `autolimitaspect = 1` (§3.6) |
| C3 | History is a checkpoint drop-down; no scrubbing, playback, or bookmarks; preview frames are never retained | `ext:753-757`; only `stored_checkpoints` persist (`session_types.jl:48`) | History has checkpoint granularity only | Timeline with dense recent history (§3.5, §4.2) |
| C4 | One diagnostics axis for every series, whatever the units, plus a fallback for Makie's empty-axis limits | `ext:714-742`, `ext:100-114` | Energy and cell count share a y-axis (requirement 4) | Groups by dimension (§3.10) |
| C5 | Two diagnostic systems: viewer-local reductions with their own schedule and history, and simulation-owned histories, linked by a bridge; vector diagnostics need `component=` | `types.jl:146-152`, `registry.jl:396-413`, `checkpoints.jl:194-213`, `frame_capture.jl:180-227`, `flow_diagnostics.jl:203-235`, docs `diagnostics.md:143-146` | Duplicated state, two schedules (`diagnostic_every/dt` vs `EveryStep`…) | The simulation-owned collection plus metadata is the only source |
| C6 | Colour range recomputed from each displayed frame | `_set_plot_style!` → `_makie_colorrange` → `_automatic_colorrange` (`ext:417-427, 345-367, 314-343`) | Colours flicker in live view and recordings; no global/fixed policy in the UI | Colour policies backed by store statistics (§3.7) |
| C7 | `record_checkpoints` deletes and rebuilds the axis every frame | `ext:1163-1177` | Frames fall back to Makie's default limits; colour range changes per frame (known defect) | Figure built once, Observables, `:global` range (§3.11) |
| C8 | The GUI runs only when the solver yields: the publisher is `@async` on the creating thread, and yields happen only at step boundaries | `session_types.jl:290-296, 308-347`; `simulate.jl:76, 164, 180`; the docstring admits it (`session_types.jl:29-31`) | Window freezes during long steps; zoom and pan included | Interaction runs in the browser; the server runs on `:interactive` threads (§4.4) |
| C9 | Lock scope: the solver holds `session_lock` during capture and the exact side-step solves; the GUI's `select_checkpoint!`/`live_mode!` take the same lock and render while holding it | `updates.jl:102-106`; `controls.jl:111-140` | GUI stalls behind the solver; in synchronous mode the solver stalls behind rendering | Immutable frames; no lock held across solver or render work |
| C10 | View state lives in the session (`state`, `mode`, `selected_checkpoint`) | `session_types.jl:55-57` | Only one view mode; two viewers (or viewer plus export) cannot look at different times | View state moves to the client |
| C11 | Memory is unbounded and in native precision; nothing persists | `stored_checkpoints` (`session_types.jl:48`); frames hold Float64 copies | Large runs grow without limit; post-hoc viewing only inside the same Julia session | Budget, decimation, spill, on-disk store (§4.2) |
| C12 | Triangulated render topology: 4 private vertices and 2 triangles per cell, plus face tiles; rebuilt on every regrid, and on every registry-bound `update_visualization!(viewer)` call | `types.jl:14-31, 39-52`; `frame_capture.jl:1-10`; `ext:964-976` | [INFERENCE from struct layouts] about 130 B per cell for cell geometry plus a similar amount per face. At 10⁶ cells that is hundreds of MB per topology, before Makie's GPU copies and per-vertex colours | Block textures (§4.6) |
| C13 | Retained-mode work-arounds | z-translate fills below outlines (`ext:429-438`); atomic plot replacement on regrid (`ext:440-489`); picking by vertex index (`ext:884-927`) | Fragile, and tests depend on internals (`test/makie_visualization.jl:7-16, 235-239`) | Analytic picking (point → block → cell); no plot surgery |
| C14 | Ergonomics: viewer and session built in two steps and bound on the GUI task; `live_every = typemax(Int)` to switch off live frames; `wait(scene)` boilerplate; `phase_rendering` is a capture policy on the registry although it is a display choice | `session_types.jl:217-221`; `examples/01_taylor_green/make_figures.jl:14`; `examples/01_taylor_green/taylor_green.jl:62-68`; `types.jl:165-185` | Friction; can't switch to `:plic` after recording | `Recorder(...; preview = nothing)`; `studio(recorder)`; `:cell`/`:plic` per panel |
| C15 | A frame references the mutable live registry; `haskey(frame, name)` checks the registry, not the frame | `checkpoints.jl:126-137` | Registering a field after capture makes `haskey` true while `frame[name]` throws | Each frame carries its own field manifest |
| C16 | A viewer or publisher exception is rethrown into `simulate!` | `_check_publisher!` (`session_types.jl:282-288`); `updates.jl:42-44` | A plotting bug can kill a long run | Consumer failures are isolated and reported in the UI (§4.4) |
| C17 | Missing: solver and AMR stats, probes, line cuts, export UI, saved layouts, themes, shortcuts, remote viewing | — | "Old-fashioned" | §3 |

---

## 2. Technology evaluation

### 2.1 What the requirements imply technically

1. **Docking**: drag a panel to dock, use tabs, close, resize splits, optionally pop out to a second
   monitor, and save/restore layouts. Building this ourselves is a sizeable project. A mature
   docking library gets it nearly for free.
2. **Interaction must not depend on Julia's scheduler.** The solver keeps every default thread busy
   (`_compute_threads() = Threads.nthreads(:default)`, `src/threading.jl:5`; Polyester `@batch`
   loops), and GC pauses stop all Julia threads. Zoom, pan, hover, and playback of cached frames
   should run without Julia; only data fetches need it.
3. **Remote machines.** Viewing over an SSH tunnel favours a browser client with data-only traffic.
   Shipping pixels (VNC/X11) or round-tripping every mouse event works poorly.
4. **Large meshes (10⁵–10⁶ cells).** GPU memory and bandwidth must be O(N) bytes per field, with no
   per-cell vertices. Very large or remote runs need LOD or viewport tiles.
5. **Headless static output in CI.** CairoMakie stays the reference for static figures and video.
6. **Light dependencies.** The core stays graphics-free; everything interactive is opt-in.

### 2.2 Options

**A. Evolve the Makie-native viewer (GLMakie).**
- Facts:
  - Layouts are `GridLayout` with `colsize!`/`rowsize!`. There is no docking framework.
  - Opening several windows works via `display(GLMakie.Screen(), fig)`; the docs call this
    "experimental support".
  - Headless use requires an X server.
  - `Axis(autolimitaspect = 1)` makes the limits adapt to the axis box; `DataAspect()` instead
    "reduces the effective axis size". The docs warn that `autolimitaspect` can cycle when tick
    labels change width.
- Pros: one renderer for interactive and static use; familiar to the owner.
- Cons:
  - Docking, tabs, drag-and-drop, and closing panels would all be custom code on top of Blocks.
  - Widgets look dated.
  - Remote viewing needs X forwarding or VNC.
  - The render loop is a Julia task (see 2.1.2).
  - The per-vertex coloured meshes stay unless a block-texture recipe is written anyway.

**B. WGLMakie figures inside a JS docking layout, through Bonito.**
- Facts:
  - WGLMakie "is the web-based backend, which is mostly implemented in Julia right now".
  - For Makie widgets, "just a hover takes quite a bit of data round-tripping between the browser
    and Julia".
  - `resize_to = :parent` resizes the canvas to its container.
  - "Static HTML export is not fully working yet."
  - Current versions: WGLMakie 0.13.17 pins Makie 0.24.17 and Bonito 5.3.
  - Each figure has its own canvas. Browsers cap the number of live WebGL contexts (Chromium about
    16 per page), and GPU resources cannot be shared between contexts.
- Pros: reuses Makie plotting code; remote-capable; real docking.
- Cons:
  - Interaction latency depends on Julia (2.1.2) and on network round trips.
  - One context per panel, so the same field is uploaded once per panel.
  - Large meshes are serialized as Makie meshes.
  - Each client session instantiates a Julia scene graph.

**C. Native docking UI with Dear ImGui (CImGui.jl) + ImPlot, embedding GLMakie via `MakieFigure`.**
- Facts:
  - CImGui.jl 10.x bundles the Dear ImGui docking branch.
  - The Makie integration is described as "very experimental".
  - It is not thread-safe, and there are reported segfaults.
  - CImGui.jl went through about seven major versions in two years.
- Pros: genuine IDE docking; low latency.
- Cons: desktop only (no remote); risky; hard to test headless.

**D. Custom browser client (TypeScript: dockview + WebGL2 + uPlot) with a Julia HTTP/WebSocket server. Recommended.**
- Facts:
  - dockview 8.8.3 is MIT-licensed and has zero dependencies. It offers tabs, groups, grids,
    splitviews, drag-and-drop, floating groups, pop-out windows, and serialization. The relevant API
    calls are `addPanel`, `removePanel`, `addFloatingGroup`, `addPopoutGroup`, `maximizeGroup`,
    `toJSON`/`fromJSON`, and `onDidLayoutChange`. Version 8 added a commercial "enterprise" tier
    (layout undo, keyboard docking, edge auto-hide); none of it is needed here.
  - uPlot 1.6.32 is MIT-licensed, about 50 KB minified, and draws with Canvas2D. It supports cursor
    sync across charts, multiple y-axes and scales, log scales, and live streaming.
  - WebGL2 practice: allocate with `texStorage` and update with `texSubImage`. A 4096 texture size
    is effectively universal. Render-to-float needs `EXT_color_buffer_float`, and float32 blending
    is not guaranteed; the design avoids both.
  - One full-page canvas scissored to each panel's rectangle avoids the context cap and lets panels
    share textures.
  - HTTP.jl 2.15 spawns connection handlers with `Threads.@spawn :interactive`. It supports binary
    WebSocket messages and permessage-deflate.
  - Julia ≥ 1.12 starts with one interactive thread by default; Julia 1.11 needs `-t N,1`.
- Pros:
  - Modern docking out of the box.
  - Interaction runs entirely in the browser at 60 fps.
  - One shared GPU context; the block-texture renderer scales.
  - Remote access needs only an SSH port forward.
  - Julia-side dependencies are light (HTTP.jl, JSON.jl, ColorSchemes.jl).
- Cons:
  - About 4–5 k lines of TypeScript to own, plus a JS toolchain for front-end contributors.
  - Two renderers (WebGL for interaction, CairoMakie for export). This is mitigated by a shared
    style and camera spec and by a Julia CPU reference rasterizer that both export and tests use
    (§4.6).

**E. Variant of D using Bonito as the app shell** (server, assets, Observables, display integration).
- Facts:
  - Bonito 5.3.0 depends on Deno_jll, MsgPack, and HTTP 2.4.
  - `ES6Module` assets are "bundled using Deno".
  - Server setup is via `configure_server!(; listen_url, listen_port, forwarded_port, proxy_url, …)`.
  - Static export is via `export_static`. In a plain terminal Bonito opens a browser window, and the
    VS Code plot pane displays `Bonito.App`s.
  - Typed numeric arrays travel as MsgPack binary extensions, with optional compression.
- Pros: notebook and VS Code embedding for free.
- Cons:
  - Our client is plain TS, so Bonito's main value (DOM written in Julia) goes unused.
  - Extra dependency weight (Deno_jll).
  - A framing layer sits over our binary protocol.
  - Major-version churn (JSServe → Bonito, then 3 → 4 → 5).
- Verdict: a fallback if notebook embedding becomes a priority (Q3).

**F. Others, rejected.**
- Genie/Stipple and Dash.jl: heavy UI frameworks without an AMR renderer.
- ParaView/Catalyst: excellent for 3-D post-processing but heavy and not REPL-centric. Instead,
  offer VTKHDF export later (Phase 7).
- Pluto: a notebook, not a dashboard.
- Electron (ElectronCall.jl/Blink.jl): not needed. A chromeless window is available through the
  browser's app mode (`--app=URL` in Chromium browsers) as an option.

### 2.3 Comparison matrix (5 = best)

| Criterion | A GLMakie | B WGLMakie+dock | C ImGui | **D TS+HTTP** | E TS+Bonito |
|---|---|---|---|---|---|
| Docking UX (req. 1) | 1 | 4 | 5 | **5** | 5 |
| Interaction independent of Julia/GC/solver | 2 | 1 | 3 | **5** | 5 |
| Remote / SSH-tunnel viewing | 1 | 4 | 0 | **5** | 5 |
| 10⁶-cell fields | 2 (mesh path) | 1 | 2 | **5** (block textures) | 5 |
| Headless static export | 5 | 5 | 1 | **5** (CairoMakie kept) | 5 |
| Dependency weight (opt-in) | 3 | 2 | 3 | **5** | 3 |
| Maintenance risk | 3 | 3 | 1 | **3** (own TS) | 3 |
| Testability in CI | 4 | 3 | 1 | **4** | 4 |

### 2.4 Recommendation

Adopt **D**:
- **Studio**: a new subpackage `lib/TwoPhaseAMRStudio` (Q2). Its dependencies are TwoPhaseAMR,
  HTTP, JSON, ColorSchemes, and CairoMakie (for export; Q13).
- **Front end**: TypeScript sources under `lib/TwoPhaseAMRStudio/frontend`. A built esbuild bundle
  is committed under `assets/` (Q4).
- **Core**: recording, store, diagnostics metadata, camera math, and the CPU rasterizer. No new
  dependencies.
- **Makie extension**: slimmed to the static and export API (§4.8).

**What stays Makie, and why.** CairoMakie is still the best option for headless, publication-grade
figures and for MP4/GIF export through `record` (FFMPEG_jll; `.mkv`, `.mp4`, `.webm`, `.gif`). Its
`rasterize` attribute keeps vector output small. The CairoMakie docs warn that per-vertex-colour
meshes are slow and recommend rasterizing them, so exports draw fields as an `image!` produced by
the CPU rasterizer, with vector overlays on top. Makie's own `datashader` recipe re-aggregates at
screen resolution with `async`, and has to switch it off when saving. `amrfield!` follows the same
pattern.

---

## 3. UX design

### 3.1 Concepts

| Term | Meaning |
|---|---|
| **Run** | One recorded simulation, live or from disk. The Studio can open several runs (Phase 7). |
| **Frame** | An owned snapshot at `(time, step)`. Kinds: `initial`, `keyframe` (scheduled, lossless), `preview` (latest-wins, may be thinned), `manual` (the user's "Keyframe now"), `pause` (state at a pause), `final`. |
| **Diagnostics** | Dense time series owned by the simulation, with metadata. |
| **Panel** | A dock item: field view, diagnostics group, probe/line cut, stats, table, or log. |
| **Playhead** | The time the client is looking at. **LIVE** means it follows the run head; **HISTORY** means it is pinned. This state belongs to each client. |
| **Link group** | A set of field panels sharing a camera (`A`, `B`, …), or diagnostics panels sharing the x-range. |

### 3.2 Default layout (auto) and wireframes

**Auto-layout rules:**
- **Field panels.**
  - Two-phase runs: volume fraction and velocity magnitude.
  - Single-phase runs: vorticity and pressure.
  - Both panels are in link group A, with the interface and solid overlays on.
- **Diagnostics.** One panel per (kind, dimension, scale) group. They form the right-hand column
  when the window is wider than 1.6:1, and the bottom row otherwise.
- **Run panel.** Status, a summary table, and the step log, as a tab behind the first diagnostics
  panel.
- **Global chrome.** The status bar (top) and timeline (bottom) are not dock items.

**W1: live overview (two-phase run, 16:9).**

```
+--------------------------------------------------------------------------------------------------+
| TwoPhaseAMR Studio | capillary_wave    (*) RUNNING  t = 0.4183 / 1.000  step 1832  dt 2.1e-4     |
| 31 steps/s  ETA 2m10s  store 412 MB / 2.0 GB (37 frames)  [|| Pause] [[] Stop] [+ Keyframe now]  |
| Layout [Two-phase v]  [+ Panel v]  [Export...]  [Theme: dark v]  [? Keys]         2 clients      |
+---------------------------------+---------------------------------+------------------------------+
| Volume fraction v [A](c)[o][] x | Velocity magn. v  [A](c)[o][] x | Energy (M L T^-2)       [] x |
|                                 |                                 |  kinetic   __/\___/\__       |
|  field fills the whole panel    |  field fills the whole panel    |  surface   ‾‾\__/‾‾‾‾‾       |
|  ~~~~~ PLIC interface ~~~~~     |  ~~~~~~~~~~~~~~~~~~~~~~~~~~     |  total     -----------  :    |
|                   |---| 0.1     |                   |---| 0.1     |                   cursor ^   |
| 0 [###############] 1  global v | 0 [##############] .31 global v +------------------------------+
| x .4120 y .0870 a=.734 L3 b17   |                                 | Length (L)              [] x |
+---------------------------------+---------------------------------+  amplitude ~~~  linear - - - |
| Probe P1 (0.412, 0.087): volume fraction(t)                  [] x +------------------------------+
|  1 |        ___/‾‾‾\___                                           | Area (L^2)  rel. change [] x |
|  0 |_______/           \_______                                   |  liquid +3e-13  gas -3e-13   |
+-------------------------------------------------------------------+------------------------------+
| |<  <  [> Play]  >  >|   1.0x v   loop [ ]     t = 0.4183     (*) LIVE               [Follow L]  |
| 0.0 |...|...|...|...|...|...|..R|...|...|.*.|......^ head                                    1.0 |
|      | keyframe   . preview   * bookmark   R regrid   ^ playhead                                 |
+--------------------------------------------------------------------------------------------------+
```

**W2: history and analysis (solver paused by a breakpoint).**

```
+--------------------------------------------------------------------------------------------------+
| (#) PAUSED at t = 0.6020  -- breakpoint: divergence_norm > 1e-8      [> Resume] [[] Stop]        |
+--------------------------------+--------------------------------+--------------------------------+
| Vorticity @ t=0.3500  [A]      | Vorticity @ t=0.6020 [A]       | Difference (right - left) [A]  |
| history frame #23 (keyframe)   | pause frame                    | symmetric range +-3.2          |
+--------------------------------+--------------------------------+--------------------------------+
| Line cut L1: volume fraction along (0, .5)-(1, .5)     [ profile | space-time ]                  |
|  t ^ ::::::::::::::::::::::::::::::  <- kymograph (x along the cut, time upward); playhead line  |
+--------------------------------------------------------------------------------------------------+
| |<  <  [> Play]  >  >|  0.5x v  loop [x] 0.30-0.60   t = 0.3500  HISTORY  (+12 new)  [Follow L]  |
+--------------------------------------------------------------------------------------------------+
```

**W3: field panel chrome.**

```
+- Volume fraction v ------------------------------- [A] (c) [o] [cam] [] x -+
|  v   field picker with search; mode :cell | :plic                          |
|  [A] link group: A / B / none          (c) colour policy and colormap      |
|  [o] overlays: interface, blocks, cells, levels, solid, glyphs, contours   |
|  [cam] fit (F), 1:1 cell zoom, named views, copy camera                    |
|  []  maximize (M)      x close (panel returns via "+ Panel")               |
|                                                                            |
|                                         scale bar |----| 0.1   t = 0.4183  |
| colourbar  0 [##########] 1   global (37 frames)  [lock]                   |
| status: x=.4120 y=.0870  value .734 (liquid)  block (L3, 12, 7) cell (5,2) |
+----------------------------------------------------------------------------+
```

**W4: zoom semantics (requirement 2).**

```
Today: DataAspect, box-zoom to 1x1           Proposed: isotropic camera fills the panel
+----------------------------------------+   +----------------------------------------+
|              +--------+                |   |::::::::::::::::::::::::::::::::::::::::|
|    empty     | 1 x 1  |     empty      |   |::: the requested 1x1 box is centred, ::|
|              | zoom   |                |   |::: the extra width shows more domain ::|
|              +--------+                |   |::::::::::::::::::::::::::::::::::::::::|
+----------------------------------------+   +----------------------------------------+
```

**W5: export dialog (requirement 3).**

```
+- Export animation -------------------------------------------------------+
| Panel        [Volume fraction (A) v]   camera: [current view v]          |
| Diagnostics  [x] Energy (M L T^-2)  [ ] Length (L)  [x] Area (L^2)       |
| Arrangement  (o) field left, diagnostics stacked right  ( ) below        |
| Frames       (o) keyframes (21)  ( ) all stored (137)                    |
|              ( ) t in [0.10, 0.80], every [1] frame                      |
| Colour       [global v] colormap [viridis v]  time label [x]  cursor [x] |
| Output       [1280] x [720] px   [12] fps   format [MP4 v] (GIF, WebM)   |
| Theme        [light v]   file: runs/capillary/volume_fraction.mp4        |
|                                                [Preview frame] [Export]  |
| ############........  54 %  rendering frame 72 / 137        [Cancel]     |
+--------------------------------------------------------------------------+
```

### 3.3 Panel types

| Panel | Content | Notes |
|---|---|---|
| **Field view** | One registered field on the AMR mesh | Cell and face fields; `:cell` (majority) or `:plic` (sharp) mode per panel; overlays (§3.8); hover readout; camera links |
| **Diagnostics group** | Time series sharing one y-axis | Component expansion (`x`/`y`, `liquid`/`gas`); transforms: raw, Δ from t₀, relative Δ, log₁₀\|·\|, d/dt; reference curves; event markers; time cursor |
| **Diagnostics stack** | Several groups as vertical splits in one panel, x-linked | Handy for export and small screens |
| **Probe** | Value at a point over time, for one or more fields | Created by clicking a field panel; stays valid across regrids (located per frame) |
| **Line cut** | Profile along a segment at the playhead; a "space–time" toggle shows the kymograph over stored frames | Useful for waves and instabilities |
| **Region stats** | min/max/mean/integral inside a drawn rectangle, per frame | Conservation checks in a sub-region |
| **Difference** | Frame A − frame B (two times, or two runs), symmetric range | "Compare with initial", run comparison |
| **AMR stats** | Cells per level over time (stacked); blocks; regrid events; level histogram at the playhead | From topology snapshots |
| **Solver stats** | dt, CFL bound, projection iterations or cycles, residuals, wall time per step, capture overhead, dropped previews | From the step log (§4.2) |
| **Table** | Diagnostic values at the playhead; frame metadata; run summary | The run summary uses the data behind `print_simulation_summary` (`flow_diagnostics.jl:172-198`) |
| **Log** (Phase 7) | Records captured during the run, linked to time | Clicking an entry jumps the playhead |
| **Filmstrip** | Keyframe thumbnails | Thumbnails come from the CPU rasterizer |

Panels are added with "+ Panel", the command palette, or by context actions (for example, clicking a
cell with `P` creates a probe). Each panel serializes to JSON together with its settings.

### 3.4 Panel behaviour

- Panels can be dragged to any edge or tab strip, resized with splitters, maximized (`M`), floated,
  or popped out to another monitor (dockview pop-out groups). A pop-out window gets its own WebGL
  context (§4.6).
- Closing a panel only removes the view; the data stays in the store.
- An empty layout shows a "+ Panel" hint.
- **Titles and badges.**
  - Titles show the field label and units.
  - A badge shows the frame kind and its age relative to the playhead, for example
    "keyframe · Δt = −0.003".
  - Panels show a "stale" stripe while waiting for data.

### 3.5 Time navigation

- **LIVE vs HISTORY.** In LIVE mode the playhead follows the newest committed frame. Any scrub,
  click on a diagnostic plot, or frame-step switches to HISTORY. `L` or "Follow" returns to LIVE.
  In HISTORY a counter shows "+n new frames". Frames keep arriving in both modes.
- **Playback.** Play/pause (`Space`), speed (`[`/`]`: 0.25×–8× in frames per second, or "simulated
  time per wall second"), loop or ping-pong over a range (shift-drag on the timeline), and
  step ±1 frame (`←`/`→`).
- **Keyframe jumps.** Jump to the previous or next keyframe (`Shift+←/→`); `Home`/`End`.
- **Timeline layers.**
  - keyframes (tall ticks), previews (dots), bookmarks (stars), regrids (R), and warnings (!);
  - the run head and the target `t_final`;
  - hover shows thumbnails.
- **Diagnostics and fields stay in step.**
  - Diagnostics are dense, fields are sparse.
  - The cursor can sit at any time; field panels show the nearest stored frame and badge the
    offset (§3.4).
  - Clicking or dragging on any diagnostics plot moves the playhead.
- **Bookmarks.** `B` adds one at the playhead; scripts call `bookmark!(recorder, "pinch-off")`.
  Automatic events: regrid, keyframe, pause/resume, breakpoint hits.
- **Solver control** (shared by all clients):
  - Pause/Resume (`P`);
  - Stop after the current step (graceful: `simulate!` returns with `stopped = true`);
  - Keyframe now;
  - Breakpoints: `pause_when!(recorder, :divergence_norm, >(1e-8))`. The solver pauses at the step
    boundary where the condition first holds, and the Studio jumps there. This is effectively a
    debugger for runs.

### 3.6 Cameras, zoom and linking (requirement 2)

- **Camera.** `(center, scale)`, where `scale` is world units per CSS pixel and is the same in x and
  y. The visible rectangle is `center ± (W, H)·scale / 2`. It therefore always fills a W×H panel,
  and cells stay square.
- **Operations:**
  - wheel zoom keeps the point under the cursor fixed;
  - drag pans;
  - box zoom sets `scale = max(box_w/W, box_h/H)` and centres on the box, so the box is fully
    visible and the extra width shows more of the domain (W4);
  - fit (`F`) uses `scale = max(Lx/W, Ly/H)`, letterboxing only when the panel's aspect differs from
    the domain's; the area outside the domain gets a subtle pattern;
  - `1` zooms so that one finest cell spans 8 px.
- **Resize.** A panel resize keeps `center` and `scale` and simply shows more or less. In fit mode
  it re-fits instead.
- **Linking.** Panels in one link group share `(center, scale)`, not limits. Each panel keeps its
  own aspect, so linking a square panel with a wide one never distorts or empties either.
- **Named views.** Camera bookmarks (for example "droplet tip") can be saved, applied to a link
  group, and used by export.
- **Makie parity.**
  - `fieldaxis(...)` creates `Axis(aspect = nothing, autolimitaspect = 1)`, and `limits!` is set
    from the camera.
  - Field axes default to minimal decorations (scale bar instead of ticks). This sidesteps the
    tick-width cycling that the Makie docs warn about.
  - For static multi-panel figures we link through our own camera, not `linkaxes!`, because
    `linkaxes!` shares limits.

### 3.7 Colour-range policies

Policies are set per panel, from the colourbar menu or with `C` to cycle:

| Policy | Range | Default for |
|---|---|---|
| `:global` | min/max over all stored frames so far; grows as the run progresses | Export; history playback |
| `:frame` | min/max of the displayed frame (today's behaviour) | Optional |
| `:view` | min/max of the values inside the visible viewport | Zoomed inspection |
| `:window` | min/max over frames in the loop range | Playback of a sub-interval |
| `(lo, hi)` | Fixed, entered or "locked" from the current range | Publication |
| modifier `symmetric` | `±max\|v\|` | Vorticity, as today (`ext:315-327, 347-351`); recommended also for velocity components |
| modifier `percentiles = (1, 99)` | Robust clipping | Fields with spikes |
| modifier `scale = :log` | log₁₀ colour mapping for positive fields | Optional |

Global, window, and percentile ranges come from per-frame statistics computed when the frame is
stored (min, max, absmax, a 64-bin histogram), not from field data. Registered `colorrange` and
colormap values become each panel's initial policy.

### 3.8 Overlays

Each overlay is a toggle on the panel; the defaults follow the registry:

- **Interface (PLIC).** Segments derived from the per-cell planes; width and colour configurable.
- **Solid geometry.** Filled solid in the shader, plus its outline.
- **Block outlines.** Optional tint by refinement level.
- **Cell edges.** Drawn in the shader, so no data is needed; they fade in automatically once cells
  are at least 6 px.
- **Velocity glyphs.** Arrows on a screen-space grid, sampled from the centred velocity; density
  adapts to zoom.
- **Streamlines.** Seeded on a jittered screen grid and recomputed on view or frame change. Line
  integral convolution is optional, in Phase 7.
- **Contours.** Of the displayed field, or of another field (for example the α = 0.5 iso-line on a
  vorticity panel).
- **Probe markers, line-cut segments, named-view rectangles.**
- **Scale bar, time stamp, colourbar.** Inside the panel and collapsible.

### 3.9 Probing and analysis

- **Hover.** Shows position, value (and phase in `:plic` mode), block key `(level, i, j)`, cell
  `(i, j)`, and cell size. The lookup runs in the client against cached values; no round trip.
- **Click.** Pins a probe. `P` opens its time series: the server samples every stored frame at that
  point, locating the containing cell per frame because the topology may differ. The result
  updates live. Probe data can be exported as CSV.
- **Line cut.** `X`, then drag. Shows the profile at the playhead and a space–time kymograph over
  stored frames, computed by the server and drawn as an image with a playhead line.
- **Region statistics.** `R`, then drag.
- **Difference view.** "Compare…" on a field panel: playhead vs a pinned time, vs the initial
  state, or vs another run.
- **Free per-field statistics.** The min/max/mean/L² of every registered field per stored frame
  become frame-resolution diagnostics in a "Field statistics" group.

### 3.10 Diagnostics grouping (requirement 4)

- **Grouping key.** `(kind, dimension, scale)` from `DiagnosticInfo` (§4.3). Kinds:
  - `:physical`: grouped by dimension;
  - `:count`: cells and blocks;
  - `:solver`: dt, CFL, iterations;
  - `:defect`: divergence norm and other errors, on a log scale.
- **Panel titles** use the dimension ("Energy, M L T⁻² per unit depth") or the units string when one
  is given ("J/m").
- **Vector diagnostics** expand into components automatically:
  `LinearMomentum` → x, y; `PhaseVolume` and `PhaseMass` → liquid, gas. No more `component=`.
- **Scale mismatch inside a group** (for example kinetic vs surface energy): a per-series
  "relative change" transform, an optional log scale, and a "split series" action that moves a
  series into its own panel.
- **Reference curves.**
  - Analytic functions passed from Julia: `references = Dict(:amplitude => t -> …)`.
  - Another diagnostic, for example `:analytical_amplitude` from the capillary-wave example.
  - Drawn dashed, with an optional "error" sub-split.
- **x-axis linking.** All diagnostics panels share the playhead. x-range linking is on by default
  and can be toggled.

### 3.11 Export (requirement 3)

- **Animations (GIF/MP4/WebM).**
  - Via the export dialog (W5) or the scripted `export_animation`. Both build the same `ExportSpec`
    and run the same Julia function (§4.8).
  - The figure is built once, frames advance through an Observable, the colour range defaults to
    `:global`, and the camera is the panel's camera or fit.
  - Optional diagnostics panels draw their full series with a moving cursor.
  - Rendering is CairoMakie and headless, so it works on a remote host; the file is offered for
    download.
  - Progress and cancellation show in the dialog.
- **Stills.**
  - One panel or the whole layout as PNG/SVG/PDF. The server maps the dockview grid tree
    (`toJSON`) onto nested Makie `GridLayout`s with `Relative` sizes.
  - Fields are drawn with `image!` from the CPU rasterizer at the target resolution. Interface,
    solid outline, and blocks are vector overlays on top.
  - A quick, WYSIWYG "Copy panel image" (PNG) uses the client canvas.
- **Data.** CSV for diagnostics, probes, and line cuts. VTKHDF frame export for ParaView is in
  Phase 7.

### 3.12 Saved layouts and sessions

- **Layout JSON.** The dockview tree plus each panel's spec (field, mode, overlays, colour policy,
  camera, link group, transforms).
- **Built-in presets.** Two-phase overview, single-phase vorticity, diagnostics-only, solver
  debugging.
- **Per-run file.** `studio.json` next to a persistent store holds bookmarks, probes, named views,
  and the last layout.
- **API.** `studio(...; layout = "two_phase.json" | :auto | :diagnostics)` and
  `save_layout(app, path)`.
- **Sharing.** The URL hash stores the playhead and the active preset, so a link opens the same
  view.

### 3.13 Keyboard shortcuts (shown with `?`; all also in the `Ctrl/Cmd+K` command palette)

| Key | Action | Key | Action |
|---|---|---|---|
| `Space` | Play/pause playback | `P` | Pause/resume solver (with a probe tool active: add probe) |
| `L` | Follow live | `K` | Keyframe now |
| `←`/`→` | Previous/next frame | `Shift+←/→` | Previous/next keyframe |
| `Home`/`End` | First/last frame | `[`/`]` | Playback speed |
| `B` | Bookmark | `F` | Fit view (linked panels too) |
| `1` | 1:1 cell zoom | `M` | Maximize panel |
| `C` | Cycle colour policy | `I`/`G`/`E` | Toggle interface / blocks / cell edges |
| `X`/`R` | Line-cut / region tool | `Ctrl+E` | Export dialog |
| `Ctrl+S` | Save layout | `D` | Light/dark theme |
| `Esc` | Cancel tool, leave maximize | `?` | Shortcut help |

### 3.14 Theming and accessibility

- Light and dark themes (dockview themes, CSS variables, uPlot palettes); the default follows the OS.
- Colormaps do not depend on the theme. Defaults are perceptually uniform and colour-blind safe:
  viridis, cividis, and cmocean `balance` for signed fields.
- Export has its own theme (light by default) so figures stay consistent.
- Everything is reachable by keyboard; tooltips name the shortcuts.

---

## 4. Architecture

### 4.1 Package layout

```
TwoPhaseAMR (core; no new dependencies)
  src/observation/      observer interface, Recorder, schedules, capture, retention
  src/visualization/    field registry (kept), sampling (kept), rasterizer, camera, styles,
                        diagnostic metadata/grouping, layout/panel/export specs (plain data)
  ext/TwoPhaseAMRMakieExt.jl   static and export API only (amrfield!, fieldaxis,
                               diagnosticsgrid!, export_animation, export_figure)
lib/TwoPhaseAMRStudio (subpackage; deps: TwoPhaseAMR, HTTP, JSON, ColorSchemes, CairoMakie)
  src/                  server, protocol, store adapter, jobs (export, probe, kymograph)
  frontend/             TypeScript sources (dockview, uPlot, WebGL2 renderer), vitest tests
  assets/               committed esbuild bundle (studio.js, studio.css, index.html)
```

**[INFERENCE]** Path dependencies for development via Pkg `[sources]`, which is available since
Julia 1.11; to be confirmed in Phase 3.

### 4.2 Data model

**Capture plan (per registry, built once).** A list of primitive copies and derived computations,
deduplicated as today (`frame_capture.jl:31-112`), each with a pooled output buffer:

| Source | Stored per frame |
|---|---|
| `CellField` | `N` values, block by block |
| `FaceField`/`FaceVelocity` component | Face values in schedule order |
| Derived (magnitude, vorticity) | `N` values, computed at capture as today |
| Phase scalars and phase velocities | Liquid **and** gas values, a majority bit mask (N bits), and sparse PLIC planes `(cell, n_x, n_y, d)` for mixed cells. `:cell` vs `:plic` becomes a display choice per panel instead of a capture policy (removes C14). |
| Solid geometry | Sparse solid planes, stored with the topology when the geometry is static |

**Frame** (immutable after commit):

```julia
struct Frame{T}
    id::Int; time::T; step::Int; kind::Symbol; wall_time::Float64
    topology::Int                          # TopologySnapshot id
    manifest::FieldManifest                # names → (placement, eltype, offset, length)
    data::Union{Vector{UInt8}, MmapRegion} # one contiguous buffer, or spilled file region
    planes::SparsePlanes{Float32}          # PLIC planes of mixed cells (if any)
    stats::Dict{Symbol, FieldStats}        # min, max, absmax, mean, finite count, 64-bin histogram
end
frame[name]   # typed view into `data`; owned, never aliases solver memory
```

**TopologySnapshot** (deduplicated; a new one only on a regrid):
- block table: key `(level, i, j)`, origin, spacing;
- `block_size`, domain, periodicity;
- face→tile scatter maps per axis (`Int32` texel index per face, §4.6);
- static solid planes.

Topology changes are detected through the mesh topology-cache generation counter
(`src/mesh/topology/mesh.jl:204-244`) instead of comparing every block key
(`frame_capture.jl:12-29`). **[INFERENCE]** Confirm that the generation changes exactly when the
leaf set changes. Size is O(blocks + faces · 4 B), compared with the hundreds of MB of triangulation
today (C12).

**Step log** (one record per completed step, columnar): time, step, dt, wall-ns, projection
iterations or cycles, initial and final residual, `converged`, topology changed, capture time.
The projection stats need `simulate!` to pass `stats` (`simulate.jl:116-120`) via `StepInfo`
(§4.7).

**Diagnostics mirror.**
- At each step boundary the recorder copies newly appended samples from the solver-owned series
  into an append-only columnar buffer, with vector values split into components.
- This is a thread-safe copy of the same samples on the same schedule, not a second history. Other
  threads never read solver-owned `Vector`s that `push!` may reallocate.

**SnapshotStore.**

- **Index.** An append-only vector of frame records published copy-on-write: readers take a
  snapshot reference, writers append under a short lock. Frames are immutable.
- **Tiers.**
  - Keyframes (`initial`, `keyframe`, `manual`, `pause`, `final`) are never dropped.
  - Preview frames follow `Retention.preview`:
    - `:logarithmic` (default): the newest `recent = 64` previews at full density; older ones
      thinned so density halves each time age doubles. Memory is O(log T), recent history dense.
    - `:uniform`: thin evenly in time.
    - `:none`: keep no previews.
    - `:all`: keep every preview.
- **Budget.** `memory = :auto` means min(2 GiB, 20 % of `Sys.total_memory()`) (Q8). Over budget,
  previews are thinned first. Keyframes then spill to disk when `storage` is a directory or `:auto`
  (a temporary directory, deleted on close unless saved); otherwise they stay in memory with one
  warning. The simulation is never aborted for memory.
- **Precision.** Keyframes keep native precision, so analysis code such as
  `frame[:fraction] == visualization_values(…)` stays exact. Previews are stored as Float32.
  Quantized UInt16 is used for network transfer only (§4.5).
- **On-disk format** (Phase 6; Q7), directory `run.tpstore/`:

  ```
  manifest.json      format version, run metadata, field and diagnostic metadata
  frames.jsonl       one line per frame: id, time, step, kind, topology, byte ranges, stats
  topology/<id>.bin  block table, scatter maps, solid planes (little-endian, raw)
  frames/<id>.bin    contiguous field buffer and sparse planes (mmap-able)
  diagnostics.bin    columns per diagnostic (times, steps, components)
  steps.bin          step log columns
  studio.json        bookmarks, probes, named views, last layout
  ```

  No dependencies beyond the standard library (`Mmap`, JSON written by hand or JSON.jl in the
  Studio). This format serves cluster runs (record to disk headless, then
  `studio("run.tpstore")` on a laptop), post-hoc export, and run comparison.

**Memory estimate at 10⁶ cells, Float32.**
- Single-phase default registration (velocity x/y faces ≈ 2N, magnitude N, vorticity N,
  pressure N; about 5N values): about 20 MB per frame.
- Two-phase default registration (volume fraction N; liquid and gas velocity faces 4N; phase
  magnitude 2N; phase vorticity 2N; pressure N; about 10N values): about 40 MB per frame.
- The recorder prints this estimate at the initial event, together with how many frames fit in the
  budget.

### 4.3 Diagnostics metadata

```julia
struct PhysicalDimension          # exponents of mass, length, time; 2-D quantities per unit depth
    mass::Int8; length::Int8; time::Int8
end
struct DiagnosticInfo
    label::String
    units::String                 # optional display units, e.g. "J/m"; "" for nondimensional runs
    dimension::Union{Nothing, PhysicalDimension}
    components::Vector{String}    # e.g. ["x", "y"], ["liquid", "gas"]
    kind::Symbol                  # :physical | :count | :solver | :defect
    scale::Symbol                 # :linear | :log
end
diagnostic_info(::AbstractDiagnostic)::DiagnosticInfo     # defaults derived from the name
Diagnostic(name, measure; label, units = "", dimension = nothing, components = String[],
           kind = :physical, scale = :linear)
```

Built-in dimensions (2-D, per unit depth), from the formulas in
`src/simulation/diagnostics/interface.jl`:

| Diagnostic | Dimension | Components | Kind / scale |
|---|---|---|---|
| Kinetic, Potential, Surface, Total energy | M L T⁻² | – | physical / linear |
| LinearMomentum | M T⁻¹ | x, y | physical |
| PhaseVolume | L² | liquid, gas | physical |
| PhaseMass | M L⁻¹ | liquid, gas | physical |
| InterfaceMeasure | L | – | physical |
| MaximumVelocity | L T⁻¹ | – | physical |
| CFLTimeStep | T | – | solver |
| DivergenceNorm, `sqrt(Σ V (∇·u)²)` (`interface.jl:414-436`) | L T⁻¹ | – | defect / log |
| LeafBlockCount, CellCount | 1 | – | count |

DivergenceNorm has the same dimension as velocity. The `kind` key keeps the two in separate
panels, which is the right result. Custom diagnostics without a dimension each get their own panel
unless they share a `units` string.

### 4.4 Threading and async contract

```
solver task (REPL / main / user thread)
  step! ─► observe!(recorder, :step, state, info)
             ├─ due?  O(1): schedules, wall-clock throttle, overhead budget
             ├─ mirror newly appended diagnostic samples           O(new samples)
             ├─ topology snapshot, only if the mesh generation changed
             ├─ capture: parallel copy + derive into a pooled buffer (compute threads idle here)
             ├─ hand-off ──► [default pool] commit task: stats, thinning, spill
             │                     └─► store index append (short lock) ─► notify subscribers
             ├─ breakpoints / pause gate (blocks only while paused)
             └─ return; never waits for clients

[:interactive pool] HTTP/WS handlers: forward "frame" events; answer "get" requests by reading
                    immutable frames and sending binary replies
browser: dockview + WebGL2 pulls data, uploads textures, renders; zoom, pan, hover, and playback
         of cached frames are local
```

**Rules:**

1. **Capture happens at step boundaries, on the solver task, in parallel.** The capture uses the
   package's own helpers (`_parallel_block_columns`, `src/threading.jl:23-49`). The compute threads
   are idle at that moment anyway, so moving capture into a background task would only compete with
   the next step. The recorder measures capture time and throttles previews so that capture stays
   below `max_capture_overhead` (default 5 %) of wall time. Keyframes are always captured.
2. **Everything off the critical path runs elsewhere.** Statistics, thinning, spilling, and
   serialization run in spawned tasks on the default pool. The network runs on `:interactive`
   threads, where HTTP.jl 2.x already spawns its handlers.
3. **No lock is held across solver work or rendering.** The store lock covers index appends only.
   Frames are immutable, so readers need no lock. This fixes C9.
4. **View state lives in clients.** The server keeps only per-connection subscriptions. Several
   tabs or users can look at different times. This fixes C10.
5. **Backpressure is client-driven and latest-wins.** Each panel has at most one outstanding data
   request. In LIVE mode, a `frame` notification triggers a request only when the previous one has
   finished, and then for the newest frame. Non-essential notifications (diagnostic appends) are
   coalesced per connection. No server-side queue grows.
6. **Failures are isolated.**
   - Server, export, or statistics failures are logged, surfaced in the UI, and never rethrown
     into `simulate!` (C16).
   - Capture errors that reveal a configuration bug (for example a mesh mismatch) still throw at
     the `:initial` event, before the run starts.
7. **Fallback for a single thread.** With `nthreads(:interactive) == 0` and one default thread, the
   recorder yields at step boundaries, as today. Zoom, pan, hover, and playback of cached frames
   keep working in the browser even during a long step; only new data waits.
8. **Pause** blocks the solver at a step boundary (cooperative, as today) and captures a `pause`
   frame, so the Studio shows the exact paused state. **Stop** makes `simulate!` return after the
   current step.

### 4.5 Wire protocol (v1)

A single WebSocket. Every endpoint requires `?token=`.

- **Text frames (JSON control).**
  - Server → client:
    - `hello{protocol, runs[{id, name, domain, block_size, t0, t_final, fields[meta],
      diagnostics[DiagnosticInfo]}]}`;
    - `frame{run, id, time, step, kind, topology, stats}`;
    - `diag{run, from, n}`;
    - `state{run_state, time, step, dt, rates, eta, store_bytes}`;
    - `event{kind, time, …}`;
    - `job{id, progress, status, url}`.
  - Client → server:
    - `get{topology | field | planes | diagnostics | steplog, …}`;
    - `probe{…}`, `linecut{…}`, `region{…}`;
    - `control{pause | resume | stop | keyframe}`, `bookmark{…}`;
    - `export{ExportSpec}`, `layout{save}`.
- **Binary frames (data).** A 32-byte little-endian header (magic, version, kind, request id,
  dtype, count, scale, offset) followed by the payload, so JS can wrap it as a `Float32Array` or
  `Uint16Array` without copying. Encodings: `f32` (default on localhost); `u16` quantized with a
  per-message scale and offset and `0xFFFF` meaning NaN (remote mode); optional permessage-deflate.
- **Field requests.** `get{field, frame, mode: cell|plic, encoding, lod?, blocks?}`. LOD and block
  subsets (viewport tiles) arrive in Phase 6; v1 sends whole fields.
- **Diagnostics.** A bulk transfer on subscribe, then appends. For very long runs (Phase 6) a
  windowed min/max decimation per pixel column on request.

### 4.6 Rendering pipeline

**Client (WebGL2, one context per browser window).**

- **Shared canvas.** One full-window canvas behind the dock layout. Each panel renders into its own
  `getBoundingClientRect()` with viewport plus scissor. Docked panels draw first, then floating
  groups in z-order. Panel bodies are transparent so dockview's DOM chrome sits on top. Textures
  are shared by all panels in the window.
- **Cell fields: block atlas.**
  - Values are uploaded into an `R32F` texture atlas of `nx×ny` tiles: 4096² texels hold
    16.7 M cells; allocate with `texStorage2D`, update with `texSubImage2D`.
  - Each block is one instance with origin, extent, tile coordinates, and level; there are 4
    vertices per block.
  - The fragment shader samples with nearest filtering, maps the value through a 256-entry colormap
    LUT texture with colour-range uniforms, and makes NaN (or the `0xFFFF` code) transparent.
  - A field of 10⁶ cells is 4 MB as `f32` and 2 MB as `u16`.
- **Face fields: dual tiles without triangles.**
  - Each block gets an x-face tile of `(nx+1) × 2ny` texels and a y-face tile of `2nx × (ny+1)`.
  - For an x-face field, the fragment uses face column `floor(x/dx + 1/2)` and tangential subrow
    `floor(2y/dy)`. This reproduces the staggered dual-cell picture inside each block.
  - Coarse–fine faces: the coarse block's half-width boundary column holds the two fine-face values
    in its two subrows, and each fine block draws its own half cell. **[INFERENCE]** This assumes
    2:1 balance and interface faces stored at fine resolution, as today's dual-tile geometry
    implies; verify against `_face_normal_intervals` (`phase_geometry.jl:337-402`) in Phase 1.
  - Periodic seams fall out naturally because every block draws only its own extent.
  - The server scatters face values into tile layout using the topology's scatter map
    (O(faces), cached per frame and field).
- **Sharp two-phase (`:plic`).**
  - A sparse plane list `(cell, n, d)` is scattered into an `RGBA16F` plane texture.
  - A fragment in a mixed cell evaluates `n·ξ − d` and picks the liquid or gas atlas.
  - Face fields use the plane of the cell that contains the fragment. That matches today's
    "dual cells intersected with adjacent cell reconstructions".
  - Solid geometry works the same way with solid planes (fill colour), plus outline segments.
- **Majority mode (`:cell`).** The server applies the stored majority mask and sends one atlas.
- **Overlays.**
  - Interface segments come from clipping the planes to cells. Thick lines are instanced quads.
  - Block outlines come from the block table; cell edges are drawn procedurally in the shader.
  - Glyphs are instanced arrows sampling the face tiles.
  - Streamlines are integrated on the CPU in JS from cached Float32 arrays, debounced.
- **LOD (Phase 6).** When cells shrink below about 0.5 px, the client switches to a 2×2-averaged
  (or max-|·|) atlas built per block. Remote clients request only the blocks intersecting the
  viewport, at the needed pyramid level, like map tiles keyed by `(frame, field, block, level)`.
  Bandwidth then scales with screen pixels, not cells.
- **Client cache.** LRU by bytes (default 512 MB, adjustable) keyed by `(run, frame, field, mode,
  lod)`. Scrubbing back and forth costs no network traffic.

**CPU reference rasterizer (core Julia).**

```julia
rasterize(frame, field, rect::NTuple{4, Float64}, (w, h); phase_rendering = :cell)::Matrix{Float32}
```

- **Algorithm.** For each block that intersects `rect`, fill its pixel rectangle using the same
  rules as the shader: cell, face columns and subrows, plane tests. The cost is O(pixels + blocks).
- **Used by:**
  - CairoMakie `image!` in exports;
  - filmstrip thumbnails;
  - server-side fallback for clients without WebGL2;
  - the test oracle for both the shader (screenshot comparison) and today's semantics. For
    example, the liquid pixel fraction should approach the liquid area, mirroring the current
    `test/visualization/fields.jl:232-300` area checks.

### 4.7 Public API sketch (Julia)

**Core: `TwoPhaseAMR` (no graphics dependencies).**

```julia
# observer interface (replaces AbstractVisualization / NoVisualization / update_visualization!)
abstract type AbstractObserver end
struct NoObserver <: AbstractObserver end
observe!(observer, event::Symbol, state::SimulationState, info::StepInfo)  # :initial :step :completed :failed
stop_requested(observer)::Bool                                            # polled after each step
struct StepInfo{T, S}
    dt::T; projection::S; topology_changed::Bool; final_time::T
    step_procedure::Any                    # independent step (initial event; exact keyframes)
    exception::Union{Nothing, Exception}   # :failed only
end
simulate!(flow, final_time; observer::AbstractObserver = NoObserver(), dt, cfl, ...)
# the returned named tuple gains `stopped::Bool`

# fields: registry kept; diagnostics and phase_rendering removed from it
fields = VisualizationFieldRegistry()                  # rename is Q9
register_default_fields!(fields, flow)
register_field!(fields, :speed, flow.velocity, CentredMagnitude();
                label = "Speed", units = "m/s", colormap = :viridis, colorrange = :global)

# schedules: existing EveryStep/EveryNSteps/EveryTimeInterval (schedules.jl) plus
AtTimes(times); EquispacedTimes(count)                 # keyframes; count includes t0 and t_final
EveryWallInterval(seconds)                             # preview throttle on wall-clock time

recorder = Recorder(fields;
    diagnostics = flow,                    # FlowSimulation | DiagnosticCollection | nothing
    keyframes = EquispacedTimes(21),       # lossless; nothing = none
    exact_keyframes = false,               # side-step to exact times (fixed topology, as today)
    preview = EveryWallInterval(0.2),      # latest-wins; nothing for headless scripts
    max_capture_overhead = 0.05,
    retention = Retention(memory = :auto, preview = :logarithmic, recent = 64),
    storage = :memory,                     # | :auto (temp spill) | "run.tpstore" (persistent)
    preview_precision = Float32)

result = simulate!(flow, 1.0; observer = recorder)

# control: thread-safe; called from the REPL, the Studio, or scripts
run_state(recorder)      # :created | :running | :pause_requested | :paused | :stopping | :completed | :failed
pause!(recorder); wait_paused(recorder); resume!(recorder)
request_stop!(recorder)                                # simulate! returns at the next boundary
request_keyframe!(recorder)
pause_when!(recorder, :divergence_norm, >(1e-8))       # breakpoint on diagnostics
bookmark!(recorder, "pinch-off"; time = nothing)

# store: read side
store = snapshot_store(recorder)
frames(store); keyframes(store)                        # Vector{FrameInfo}
frame = store[id]; frame = frame_at(store, t)          # nearest stored state
frame[:volume_fraction]                                # owned values
frame_stats(store, :vorticity)                         # feeds colour policies
diagnostic_history(store, :phase_volume)               # mirrored columns, with components
step_log(store)
save_store("run.tpstore", store); store = open_store("run.tpstore")   # Phase 6

# geometry utilities shared by Studio, export, and tests
struct Camera; center::NTuple{2, Float64}; scale::Float64; end       # world units per pixel
fit_camera(domain, (w, h)); zoom(camera, factor, anchor_px, (w, h))
zoom_to_box(camera, box, (w, h)); visible_rect(camera, (w, h))
rasterize(frame, field, rect, (w, h); phase_rendering = :cell)
locate(frame, (x, y))::Union{Nothing, CellRef}
locate_face(frame, (x, y), axis)::Union{Nothing, FaceRef}   # dual tile under the point
diagnostic_info(d); diagnostic_groups(infos)           # grouping used by Studio and Makie
```

**Studio: `TwoPhaseAMRStudio`.**

```julia
using TwoPhaseAMRStudio
app = studio(recorder; layout = :auto, host = "127.0.0.1", port = 0,
             open = :browser,             # | :app (Chromium app window) | :none
             theme = :system)             # prints URL with token and an `ssh -L` hint
app = studio("run.tpstore")                       # post-hoc
app = studio(store_a, store_b; layout = :compare) # run comparison (Phase 7)
url(app); wait(app); close(app)
probes(app); save_layout(app, "two_phase.json")
```

**Makie extension (CairoMakie, GLMakie, WGLMakie).**

```julia
ax = fieldaxis(fig[1, 1]; camera = :fit, decorations = :minimal)  # aspect = nothing, autolimitaspect = 1
amrfield!(ax, frame_or_observable, :velocity_vorticity;
          colorrange = (-3, 3), colormap = :balance, phase_rendering = :cell,
          interface = true, geometry = true, blocks = false, async = false)
axes = diagnosticsgrid!(fig[1, 2], store_or_flow; names = :all, group = :dimension,
                        transform = :raw, cursor = time_observable,
                        references = Dict(:amplitude => t -> a0 * exp(-γ * t)))
export_animation("wave.mp4", store; field = :velocity_vorticity,
                 diagnostics = [:kinetic_energy, :phase_volume], frames = :keyframes,
                 framerate = 12, size = (1280, 720), colorrange = :global,
                 camera = :fit, theme = :light)
export_figure("overview.pdf", store, layout_or_panel; at = 0.5)
```

The example workflow shrinks to:

```julia
recorder = Recorder(fields; diagnostics = flow, keyframes = EquispacedTimes(21))
visualize && (app = studio(recorder))
result = simulate!(flow, final_time; observer = recorder)
```

and, for documentation figures:

```julia
export_animation(path, snapshot_store(recorder); field = :vorticity, diagnostics = [:kinetic_energy])
```

### 4.8 How the static API fits

The static API is a thin Makie layer over core data: store, frame, camera, `rasterize`, style, and
diagnostic groups. The Studio's export jobs call exactly these functions.

| Today | New | Defect removed |
|---|---|---|
| `snapshot_figure`, `snapshot_figure!` (`ext:1069-1142`) | `fieldaxis` + `amrfield!`; `export_figure` | DataAspect boxes (C2); vector meshes too slow for large N |
| `plot_diagnostics!` (`ext:1144-1161`) | `diagnosticsgrid!` (groups, components, transforms, references, cursor) | One axis for mixed units (C4) |
| `record_checkpoints` (`ext:1163-1177`) | `export_animation`: figure built once, Observables, `:global` range, fixed camera | Default limits, per-frame colours (C6, C7) |

`amrfield!` re-rasterizes when the axis limits or viewport change, with optional `async`, like
`datashader`. Makie-based interactive use (GLMakie or WGLMakie) therefore stays possible for users
who want it, without maintaining a GLMakie application.

### 4.9 Security

- Bind to `127.0.0.1` by default. Every HTTP and WS request needs a random 128-bit token in the URL,
  as in Jupyter. On a shared machine, another local user must not be able to pause the run or read
  its data.
- `host = "0.0.0.0"` is explicit opt-in. Remote use is through `ssh -N -L port:127.0.0.1:port host`;
  the printed hint gives the exact command.
- An optional read-only token hides solver control.

### 4.10 Testing strategy (all headless)

1. **Core** (`recording` group, replacing `visualization/session.jl`; plain Julia):
   - schedules: count, interval, and explicit times, including initial/final; nearest vs exact
     (port of today's tests);
   - wall-clock throttle with an injectable clock;
   - retention properties: budget respected, keyframes never dropped, recent window dense, times
     monotone;
   - topology deduplication across regrids;
   - ownership: mutate or adapt the solver state after capture, and the frame is unchanged;
   - non-blocking: a deliberately slow consumer leaves step wall-time unchanged within a tolerance,
     and the dropped-preview count is reported;
   - pause, stop, and breakpoint semantics; failure isolation;
   - diagnostics metadata, grouping, and mirror;
   - CPU rasterizer: pixel-centre values equal `frame[name]`; plane classification; face dual tiles
     including coarse–fine and periodic; area fractions;
   - camera invariants: box zoom contains the box, scale is isotropic, fit fills.
2. **Makie extension** (existing `test/makie` CairoMakie environment, run by CI
   `.github/workflows/CI.yml:113-114`):
   - `amrfield!` pixels match `rasterize`;
   - a `fieldaxis` box equals its layout cell for any limits aspect;
   - `export_animation` regression for C6/C7: rendered frames share `finallimits` and
     `colorrange`, and GIF/MP4 files are written;
   - `diagnosticsgrid!` creates one axis per group.
3. **Studio server** (Julia, no browser):
   - start on port 0 and connect with `HTTP.WebSockets`;
   - a wrong token is rejected;
   - `frame` notifications arrive during a short `simulate!`;
   - decoded binary payloads equal `frame[name]`;
   - a slow client receives only the latest frames;
   - control messages change `run_state`;
   - an export job writes a file.
4. **Front end**:
   - `vitest` for camera math, colour policies, protocol decoding, and layout round trips. Camera
     parity uses shared JSON test vectors generated by the Julia implementation.
   - Optional nightly Playwright job: Chromium needs an explicit software-WebGL opt-in
     (`--enable-unsafe-swiftshader` / `--use-angle=swiftshader`) since the automatic SwiftShader
     fallback was removed. It loads a fixture store, compares field-panel screenshots with the CPU
     rasterizer (tolerance), and exercises docking (close, move, resize) and shortcuts.
5. **CI jobs.**
   - Keep `test/makie`.
   - Add `studio` (Julia server tests) and `frontend` (node, vitest, and a bundle-freshness check:
     the bundle header carries a hash of the sources).
   - Nightly `browser` job.

---

## 5. Migration plan

Effort is in focused developer-days (d). Each phase is one PR (or two, where noted) and keeps
`main` green. The old viewer keeps working, unchanged and frozen, until Phase 4. No compatibility
shims remain afterwards.

| Phase | Content | Deletes | Effort | Acceptance |
|---|---|---|---|---|
| **0** (in flight) | Separate hotfix for `record_checkpoints` limits and colour range | — | — | Already being fixed |
| **1 Core recording** | `AbstractObserver`/`StepInfo` in `simulate!` (stats, stop); `Recorder`, schedules (`AtTimes`, `EquispacedTimes`, `EveryWallInterval`), parallel capture plan, in-memory `SnapshotStore` (tiers, budget, logarithmic thinning, stats), topology snapshots by generation, diagnostics mirror, `DiagnosticInfo` and dimensions for built-ins, step log, pause/stop/breakpoints; core `Camera` and `rasterize` | Nothing yet. `VisualizationSession` is frozen, and during the migration window `simulate!` accepts both `visualization=` and `observer=`; Phase 4 removes `visualization=` | 6–8 d | Core tests in §4.10.1; examples run headless with `Recorder` |
| **2 Static and export API** | Makie extension: `fieldaxis`, `amrfield!` recipe, `diagnosticsgrid!`, `export_animation`, `export_figure`; port every `make_figures.jl` | `snapshot_figure(!)`, `plot_diagnostics!`, `record_checkpoints`; aspect work-arounds in examples; `live_every = typemax(Int)` | 3–4 d | §4.10.2; regenerated gallery assets look the same or better |
| **3 Studio MVP (parity plus the 4 requirements)** | Subpackage, server and token, protocol v1 (whole fields), dockview shell, field panels (cell and face fields, `:cell`/`:plic`, interface, solid, blocks, cell edges, hover and click info), camera (fit, zoom, box zoom, linking), diagnostics panels grouped by dimension with components and cursor, timeline (live/history, play, step, keyframe jumps), solver controls, auto layout, light/dark, export dialog calling Phase 2, committed bundle and frontend CI | — | 9–12 d (2 PRs: server+shell, renderer+panels) | §4.10.3–4; all four owner requirements demonstrated on the capillary-wave and decaying-turbulence examples |
| **4 Cutover** | Examples (`visualize = true` → `studio(recorder)`), README, `docs/src/visualization.md` rewrite (Recording · Studio · Static and export), `api.md`, `diagnostics.md`, `how-to.md`, `navier-stokes.md`; the `visualization` test group becomes `recording` | See §5.1 | 3–4 d | No references to deleted APIs (`grep`); docs build; CI green |
| **5 Studio power features** | Colour-policy UI (`:view`, `:window`, percentiles, log); probes, line cut and kymograph, region stats, difference view; AMR and solver stats panels; bookmarks and events; saved layouts and presets; shortcuts and command palette; glyphs, streamlines, contours; filmstrip | — | 7–9 d | Feature checklist in §3; vitest + Playwright |
| **6 Persistence and scale** | On-disk `run.tpstore` (+ spill), `save_store`/`open_store`, post-hoc `studio(path)`; `u16` transfer, LOD pyramids, viewport tiles, client cache budget; decimated diagnostics for very long runs | — | 5–7 d | Run 10⁶-cell fixture: memory stays within budget; remote (throttled) scrub latency target < 300 ms per frame |
| **7 Extras (optional)** | Multi-run comparison, offline HTML report (read-only client plus embedded subset), log capture panel, LIC, VTKHDF export | — | 5–8 d | Per feature |

Total for the core path (Phases 1–4): about 21–28 d. Through Phase 6: about 33–44 d.

### 5.1 Deleted in Phase 4 (clean cutover)

- **`ext/TwoPhaseAMRMakieExt.jl` interactive viewer:** `MakieAMRVisualization`, `visualize` (both
  methods), `select_visualization_field!`, `selected_cell`, `selected_face`,
  `visualization_figure`, `update_visualization!(viewer, …)`, and the menu, checkbox, button, and
  picking code (about 1000 of 1179 lines).
- **`src/visualization/session_types.jl`, `controls.jl`, `updates.jl`, `frame_capture.jl`:**
  `VisualizationSession`, `LatestFrameMailbox` (its latest-wins principle moves into the client
  protocol), `select_checkpoint!`, `live_mode!`, `wait_visualization!`,
  `dropped_visualization_frames`, `visualization_checkpoints`, `latest_frame`,
  `visualization_diagnostics`, `request_pause!`/`resume_simulation!`/`wait_until_paused`/
  `simulation_state(::VisualizationSession)` (replaced by the recorder controls).
- **`src/visualization/checkpoints.jl`:** `CheckpointCount`/`CheckpointInterval`/
  `ExplicitCheckpointTimes` (replaced by the schedules), `VisualizationTopology`,
  `VisualizationFrame`, `VisualizationCheckpoint`, `VisualizationDiagnosticSample`,
  `VisualizationDiagnosticHistory`.
- **Triangulation:** `types.jl:14-31, 39-68`; the polygon and triangulation parts of
  `phase_geometry.jl` (including `visualization_geometry` at 289 and 405); and the vertex helpers
  in `sampling.jl:467-575` (`_visualization_vertex_values`, `cell_for_vertex`,
  `face_for_vertex`, `_cell_outline`, `_face_outline`). Removed types and functions:
  `AMRCellGeometry`, `AMRFaceGeometry`, `PLICVisualizationGeometry` and its cache. Replaced by
  the rasterizer, the shader, `locate`, and `locate_face`. Source classification
  (`_is_discontinuous_source`, `_render_source_values`) and the centring and curl samplers stay.
- **Kept:**
  - `plic_segments` and `plic_solid_geometry`, which are general geometry utilities used by tests
    (`test/indicator/geometry.jl:499`) and needed for vector overlays;
  - `visualization_values`, and `FaceRef` (`types.jl:33-37`), which `locate_face` returns for
    face-field probes;
  - derived sources (`CentredMagnitude`, `Vorticity`);
  - the field registry.
- **Viewer-local diagnostics:** `register_diagnostic!(registry, …)` (`registry.jl:396-413`),
  `VisualizationDiagnosticEntry`, `diagnostic_names(::VisualizationFieldRegistry)`, the bridge in
  `flow_diagnostics.jl:203-235`, and the `component=` mechanism.
- **Exports in `src/TwoPhaseAMR.jl:97-113`:** updated accordingly. The `AbstractVisualization` →
  `AbstractObserver` rename touches `simulate.jl` and `independent_step.jl:172-175`, and the
  `visualization=` keyword of `simulate!` is removed (only `observer=` remains).
- **Tests:**
  - rewrite `test/makie_visualization.jl`;
  - `test/visualization/session.jl` → recorder tests;
  - `test/visualization/embedded_geometry.jl` and the triangulation parts of
    `test/visualization/fields.jl` → rasterizer tests;
  - update `test/groups.jl:12, 36-40`.
- **Dependencies:** GLMakie leaves `examples/Project.toml`. TwoPhaseAMRStudio comes in, and
  CairoMakie stays.

### 5.2 Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| A TypeScript code base in a Julia repository (toolchain, contributors) | Maintenance | Small dependency set (dockview, uPlot); committed bundle; vitest; strict TS; Julia-generated parity vectors; renderer kept to about 2 k lines |
| Two renderers diverge (WebGL vs CairoMakie) | Visual mismatch | Shared style and LUT spec from ColorSchemes; CPU rasterizer as the oracle; screenshot tests |
| No interactive thread (Julia 1.11 default) | Server latency during steps | Cooperative fallback; document `julia -t auto` (≥ 1.12 includes one interactive thread) or `-t N,1` |
| GC pauses from large captures | Server hiccups | Pooled capture buffers; contiguous frame buffers |
| Memory blow-up at 10⁶ cells | OOM on shared machines | Budgets, thinning, spill; estimate printed at start |
| Browser and GPU quirks (Safari, float textures) | Rendering bugs | WebGL2 core features only (R32F with NEAREST, no float render targets or blending); `u16` path as fallback |
| HTTP.jl 2.x or dockview API churn | Breakage | Compat bounds; thin adapters; dockview 8 MIT core is stable, enterprise not used |
| CairoMakie export in a spawned task while the GUI runs | Global theme/state races | Explicit attributes instead of `with_theme`; one export job at a time |
| Capture overhead with many derived fields | Slower solver | Overhead budget throttles previews; register fewer derived fields (presets) |
| `exact_keyframes` with dynamic AMR | Still unsupported | Unchanged guard (`simulate.jl:58-63`); documented |
| Security on shared hosts | Data leak or control by other users | Localhost bind plus token; read-only token option |

### 5.3 Open questions for the owner (with recommended answers)

| # | Question | Recommendation |
|---|---|---|
| Q1 | Move the interactive viewer to a browser front end (D) instead of evolving GLMakie? | **Yes**: the only option that meets docking, remote viewing, a responsive UI during solves, and 10⁶ cells together |
| Q2 | Packaging: subpackage `lib/TwoPhaseAMRStudio` or a weak-dependency extension triggered by HTTP/Bonito? | **Subpackage**. Discoverable (`using TwoPhaseAMRStudio`), its own dependencies and tests, core untouched |
| Q3 | Server stack: HTTP.jl directly or Bonito? | **HTTP.jl**. Lighter (no Deno_jll), full control of the binary protocol; revisit Bonito only if notebook embedding becomes important |
| Q4 | Ship the JS bundle committed or as a build artifact? | **Committed** (a few hundred KB), with a CI freshness check |
| Q5 | Rename for the cutover: `VisualizationSession` → `Recorder`, checkpoints → keyframes, `visualization=` → `observer=`, `AbstractVisualization` → `AbstractObserver`? | **Yes**. Pre-release; the names should describe recording, not viewing |
| Q6 | Storage precision: keyframes native (Float64), previews Float32? | **Yes**. Keeps analysis exact and halves preview memory |
| Q7 | On-disk store format: custom raw + JSON, or HDF5/JLD2/Zarr? | **Custom raw + JSON** (no dependencies, mmap, append-friendly); VTKHDF as an export, not as the store |
| Q8 | Default memory budget and spill? | `memory = :auto` = min(2 GiB, 20 % RAM); `storage = :memory` by default, `:auto` spill opt-in. Revisit if runs exceed it in practice |
| Q9 | Rename `VisualizationFieldRegistry` (for example `FieldRegistry`)? | **Optional**. Keep the name to limit churn unless you prefer the shorter one |
| Q10 | Remove viewer-local diagnostic reductions entirely (only simulation-owned diagnostics plot)? | **Yes**. One source of truth; low-level `simulate!` callers pass a `DiagnosticCollection` |
| Q11 | Units: own `PhysicalDimension` metadata or a Unitful.jl dependency? | **Own metadata** (three exponents); units strings are optional display text |
| Q12 | Static vector export: fields raster (`image!`) with vector overlays, dropping vector cell meshes? | **Yes**. Scales to 10⁶ cells; CairoMakie recommends rasterizing large meshes |
| Q13 | Should the Studio depend on CairoMakie so GUI export always works? | **Yes**. The Studio is opt-in, and export is a core requirement |
| Q14 | Keep exact (side-step) keyframes? | **Yes**. Tests and verification rely on them; still fixed-topology only |
| Q15 | Observer failures must never abort `simulate!` (except configuration errors at `:initial`)? | **Yes** |
| Q16 | Is 3-D on the roadmap? | If yes, keep the panel and protocol abstractions; slice and volume panels would be added later. The design targets 2-D quadtrees now |
| Q17 | Default bind and remote policy: localhost plus token, explicit `host` opt-in? | **Yes** |

---

## Appendix A. Library facts checked (2026-11-08)

| Fact | Source |
|---|---|
| `aspect = DataAspect()` reduces the axis size to the aspect of the data limits ("excess whitespace"); `autolimitaspect = 1` changes the limits instead; a warning about cyclical updates from tick changes; `linkxaxes!`/`linkyaxes!` | https://docs.makie.org/dev/reference/blocks/axis |
| `DataInspector` hover tooltips; `inspector_label` | https://docs.makie.org/stable/explanations/inspector |
| `record` via FFMPEG_jll; `.mkv` (default), `.mp4`, `.webm`, `.gif`; Observable-driven animation | https://docs.makie.org/stable/explanations/animation |
| CairoMakie: per-vertex-colour meshes slow, selective `rasterize`; `px_per_unit`/`pt_per_unit` | https://docs.makie.org/stable/explanations/backends/cairomakie |
| GLMakie: multiple windows "experimental" (`display(GLMakie.Screen(), fig)`); headless needs an X server; renderloop options | https://docs.makie.org/stable/explanations/backends/glmakie |
| WGLMakie: "mostly implemented in Julia"; hover round trips; `resize_to = :parent`; static HTML export not fully working; remote `Page(listen_url, …)` | https://docs.makie.org/stable/explanations/backends/wglmakie |
| `datashader`: `async` re-aggregation skips zoom updates while busy, disable when saving; `binsize` per screen pixel; example with `autolimitaspect = 1` | https://docs.makie.org/stable/reference/plots/datashader |
| Makie 0.24.17; WGLMakie 0.13.17 (Bonito 5.3) | https://github.com/MakieOrg/Makie.jl (Makie/Project.toml, WGLMakie/Project.toml) |
| Bonito 5.3.0 depends on Deno_jll, HTTP 2.4, MsgPack; `ES6Module` bundled with Deno; `configure_server!`; `export_static`; browser display in terminals; VS Code plot pane | https://github.com/SimonDanisch/Bonito.jl/blob/master/Project.toml, https://bonito.sh/stable/api.html, https://bonito.sh/stable/deployment.html, https://bonito.sh/stable/assets.html |
| Bonito typed numeric arrays as MsgPack binary extensions | https://github.com/SimonDanisch/Bonito.jl/blob/master/src/serialization/msgpack.jl |
| HTTP.jl 2.15: connection handlers `Threads.@spawn :interactive`; binary WebSocket messages; permessage-deflate | https://github.com/JuliaWeb/HTTP.jl (`src/HTTP.jl`, `src/http_server.jl`, `src/http_websockets.jl`, `src/http_websocket_pmce.jl`) |
| Julia starts with 1 worker and 1 interactive thread by default (1.12+) | https://docs.julialang.org/en/v1/manual/multi-threading/, https://julialang.org/blog/2025/10/julia-1.12-highlights |
| dockview 8.8.3 (MIT, zero deps): tabs, groups, grids, splitviews, DnD, floating groups, pop-out windows, serialization; enterprise extras listed | https://github.com/dockview/dockview, https://dockview.dev/blog/dockview-enterprise, https://dockview.dev/typedocs/classes/dockview-core.DockviewApi.html |
| uPlot 1.6.32 (MIT, about 50 KB, Canvas2D): cursor sync, multiple scales, log scales, streaming | https://github.com/leeoniya/uPlot |
| WebGL best practices: `texStorage` + `texSubImage`; MAX_TEXTURE_SIZE 4096 effectively universal; float render targets need `EXT_color_buffer_float`; float32 blending not guaranteed | https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices |
| One canvas with scissored views; contexts don't share resources; Chromium caps live WebGL contexts at about 16 | https://webgl2fundamentals.org/webgl/lessons/webgl-multiple-views.html, https://threejs.org/manual/en/multiple-scenes.html, https://issues.chromium.org/issues/40939743 |
| CImGui.jl: Makie integration "very experimental", not thread-safe; v10 bundles the Dear ImGui docking branch | https://juliaimgui.github.io/ImGuiDocs.jl/cimgui/stable/makie, https://juliaimgui.github.io/CImGui.jl/stable/changelog |
| Chrome removed automatic SwiftShader WebGL fallback; explicit opt-in for CI | https://groups.google.com/a/chromium.org/g/blink-dev/c/yhFguWS_3pM, https://chromium.googlesource.com/chromium/src/+/HEAD/docs/gpu/swiftshader.md |

## Appendix B. Repository references used

- `ext/TwoPhaseAMRMakieExt.jl`: struct 6-63; colour range 300-344, 417-427; z-layer 429-438;
  plot replacement 440-489; `visualize` 570-932 (DataAspect 649, diagnostics axis 714-742,
  checkpoint menu 753-757, `tightlimits!` 786, picking 884-927); live update rebuilds topology
  964-976; snapshot 1069-1142 (DataAspect 1110); `plot_diagnostics!` 1144-1161;
  `record_checkpoints` 1163-1177.
- `src/visualization/`:
  - `types.jl` 1-198;
  - `session_types.jl` 1-73, 182-226, 282-347, 369-379;
  - `checkpoints.jl` 31-108, 110-134, 189-213;
  - `frame_capture.jl` 1-112, 114-142, 204-247, 287-332;
  - `updates.jl` 6-106;
  - `controls.jl` 28-140;
  - `registry.jl` 37-57, 292-417;
  - `sampling.jl` 1-19.
- `src/navier_stokes/simulate.jl` 22-197; `src/navier_stokes/flow_diagnostics.jl` 203-235;
  `src/navier_stokes/flow_simulation.jl` 40-50, 411-440, 494-527.
- `src/simulation/diagnostics/{interface,history,schedules}.jl`; `src/mesh/topology/mesh.jl`
  204-244; `src/threading.jl` 5, 23-49.
- Tests: `test/makie_visualization.jl` (436 lines, CairoMakie env `test/makie`, CI
  `.github/workflows/CI.yml:113-114`); `test/groups.jl:12, 36-40`; `test/visualization/*.jl`.
- Examples: `examples/*/` live paths (for example `01_taylor_green/taylor_green.jl:41-68`) and
  `make_figures.jl` scripts (aspect work-arounds in 06/07/08; `live_every = typemax(Int)` in 01,
  02, 03, 04, 06–10).

# Interactive visualization

Visualization is implemented as an optional Makie extension. Loading GLMakie
or WGLMakie loads Makie and activates the extension; using the numerical solver
alone does not load a graphics stack.

```julia
using TwoPhaseAMR, GLMakie

mesh = BlockTreeMesh(
    block_size=(12, 12), max_level=2,
    refine=(key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
)
field = CellField(mesh)
initialize!(field, (x, y) -> sinpi(2x) * cospi(2y))
viewer = visualize(field)
```

Use `WGLMakie` in place of `GLMakie` for browser rendering. Backends should be
installed into the application environment rather than added as dependencies
of `TwoPhaseAMR`.

The plotted cells contain owned values only; halo entries are not rendered.
Each cell is represented by two triangles with four private vertices, which
preserves piecewise-constant colors and provides an unambiguous mapping from a
picked primitive to `CellRef`. Leaf-block outlines are a separate plot layer.

`update_visualization!(viewer, field)` replaces colors while reusing the mesh
geometry. The field must currently share the exact `BlockTreeMesh` object used
to construct the viewer. Replacing the topology is deferred until solver
regridding exists.

The current API includes:

- `visualize(field; display=true, show_blocks=true, show_cells=false, ...)`;
- `update_visualization!(viewer, field)`;
- `selected_cell(viewer)`; and
- `visualization_figure(viewer)` for further Makie customization.

The checkpoint, live-session, diagnostics, and simulation-control design is
specified in milestone 7 of `DEVELOPMENT.md`.

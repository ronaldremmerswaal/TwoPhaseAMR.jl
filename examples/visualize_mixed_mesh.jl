using TwoPhaseAMR

backend = get(ENV, "TWOPHASEAMR_VISUALIZATION_BACKEND", "GLMakie")
if backend == "GLMakie"
    @eval using GLMakie
    GLMakie.activate!()
elseif backend == "WGLMakie"
    @eval using WGLMakie
    WGLMakie.activate!(use_html_widgets=true)
else
    error("TWOPHASEAMR_VISUALIZATION_BACKEND must be GLMakie or WGLMakie")
end

function visualize_mixed_mesh()
    mesh = BlockTreeMesh(
        block_size=(12, 12), max_level=2,
        refine=(key, origin, extent) ->
            key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 0))
    pressure = CellField(mesh)
    initialize!(pressure, (x, y) -> sinpi(2x) * cospi(2y))
    return visualize(
        pressure; title="Pressure on a mixed-level mesh", show_blocks=true)
end

if abspath(PROGRAM_FILE) == @__FILE__
    visualize_mixed_mesh()
end

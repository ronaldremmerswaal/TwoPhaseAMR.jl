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
        block_size=(12, 12), periodic=(false, false), max_level=2,
        refine=(key, origin, extent) ->
            key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 0))
    pressure = CellField(mesh)
    initialize!(pressure, (x, y) -> sinpi(2x) * cospi(2y))
    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? 0.5 - y : x - 0.5)
    fields = VisualizationFieldRegistry()
    register_field!(
        fields, :pressure, pressure; label="Pressure", colormap=:viridis)
    register_field!(
        fields, :velocity_x, velocity, Face{1}();
        label="Velocity x", units="m/s", colormap=:balance)
    register_field!(
        fields, :velocity_y, velocity, Face{2}();
        label="Velocity y", units="m/s", colormap=:balance)
    register_field!(
        fields, :speed, velocity, CentredMagnitude();
        label="Speed", units="m/s")
    register_field!(
        fields, :vorticity, velocity, Vorticity();
        label="Vorticity", units="1/s", colormap=:balance)
    return visualize(
        fields; field=:pressure, show_blocks=true)
end

if abspath(PROGRAM_FILE) == @__FILE__
    visualize_mixed_mesh()
end

using TwoPhaseAMR

const MANUFACTURED_POISSON_RADIUS = 0.3
const MANUFACTURED_POISSON_CENTRE = (0.5, 0.5)
const MANUFACTURED_POISSON_DENSITY = (1000.0, 1.0)
const MANUFACTURED_POISSON_SCALE = 40.0

"""Continuous manufactured pressure, with a discontinuous first derivative."""
function manufactured_pressure(x, y)
    # Equal x/y amplitudes make the normal and tangential surface-RMS parts
    # of [grad(p)/rho] equal on the circular interface by symmetry.
    background = sinpi(2x) + cospi(2y)
    dx = x - MANUFACTURED_POISSON_CENTRE[1]
    dy = y - MANUFACTURED_POISSON_CENTRE[2]
    level = dx^2 + dy^2 - MANUFACTURED_POISSON_RADIUS^2
    liquid_correction = level < 0 ?
        MANUFACTURED_POISSON_SCALE * level * (x - y) : 0.0
    return background + liquid_correction
end

"""Exact physical pressure gradient in phase `1` (liquid) or `2` (gas)."""
function manufactured_pressure_gradient(x, y, phase)
    gx = 2pi * cospi(2x)
    gy = -2pi * sinpi(2y)
    if phase == 1
        dx = x - MANUFACTURED_POISSON_CENTRE[1]
        dy = y - MANUFACTURED_POISSON_CENTRE[2]
        level = dx^2 + dy^2 - MANUFACTURED_POISSON_RADIUS^2
        factor = x - y
        gx += MANUFACTURED_POISSON_SCALE * (2dx * factor + level)
        gy += MANUFACTURED_POISSON_SCALE * (2dy * factor - level)
    elseif phase != 2
        throw(ArgumentError("phase must be 1 (liquid) or 2 (gas)"))
    end
    return (gx, gy)
end

"""Divergence-free projected velocity used by the manufactured solution."""
function manufactured_projected_velocity(x, y, phase)
    phase == 2 && return (0.0, 0.0)
    phase == 1 || throw(
        ArgumentError("phase must be 1 (liquid) or 2 (gas)"),
    )
    dx = x - MANUFACTURED_POISSON_CENTRE[1]
    dy = y - MANUFACTURED_POISSON_CENTRE[2]
    factor = cospi(2(dx^2 + dy^2))
    return (2dy * factor, -2dx * factor)
end

function _manufactured_predictor(mesh, density)
    return ntuple(2) do phase
        FaceVelocity(
            mesh,
            (x, y, time, axis) ->
                manufactured_projected_velocity(x, y, phase)[axis] +
                manufactured_pressure_gradient(x, y, phase)[axis] /
                    density[phase],
        )
    end
end

function _multiply_velocity(field::FaceVelocity, factor)
    result = copy(field)
    result.interior .*= factor
    result.boundary .*= factor
    return result
end

function _velocity_difference(first::FaceVelocity, second::FaceVelocity)
    result = copy(first)
    result.interior .-= second.interior
    result.boundary .-= second.boundary
    return result
end

function _gauge_aligned_pressure(pressure, exact_pressure)
    displayed = copy(pressure)
    error = CellField(pressure.mesh)
    weighted_difference = 0.0
    total_volume = 0.0
    for block_index in eachindex(pressure.mesh.blocks)
        volume = pressure.mesh.blocks[block_index].volume
        numerical = interior(pressure, block_index)
        exact = interior(exact_pressure, block_index)
        weighted_difference += volume * sum(numerical .- exact)
        total_volume += volume * length(numerical)
    end
    gauge_offset = weighted_difference / total_volume
    for block_index in eachindex(pressure.mesh.blocks)
        numerical = interior(displayed, block_index)
        exact = interior(exact_pressure, block_index)
        numerical .-= gauge_offset
        interior(error, block_index) .= numerical .- exact
    end
    fill_halo!(displayed)
    fill_halo!(error)
    return displayed, error, gauge_offset
end

function _pressure_error_norms(error::CellField)
    error_squared = 0.0
    error_maximum = 0.0
    total_volume = 0.0
    for block_index in eachindex(error.mesh.blocks)
        volume = error.mesh.blocks[block_index].volume
        values = interior(error, block_index)
        error_squared += volume * sum(abs2, values)
        error_maximum = max(error_maximum, maximum(abs, values))
        total_volume += volume * length(values)
    end
    return (l2 = sqrt(error_squared / total_volume), linf = error_maximum)
end

function _gradient_error_norms(error, mesh, face_data)
    dual_volumes = TwoPhaseAMR._uncut_face_dual_volumes(mesh, Float64)
    error_squared = 0.0
    error_maximum = 0.0
    total_weight = 0.0
    for face_index in eachindex(mesh.faces)
        fractions = (
            face_data.liquid_fraction[face_index],
            1 - face_data.liquid_fraction[face_index],
        )
        for phase in 1:2
            fraction = fractions[phase]
            weight = fraction * dual_volumes[face_index]
            value = error[phase].interior[face_index]
            error_squared += weight * abs2(value)
            total_weight += weight
            fraction > sqrt(eps(Float64)) &&
                (error_maximum = max(error_maximum, abs(value)))
        end
    end
    return (l2 = sqrt(error_squared / total_weight), linf = error_maximum)
end

function _xi_fields(workspace, mesh, density, mixed_faces)
    numerical = FaceVelocity(mesh; init = NaN)
    analytical = FaceVelocity(mesh; init = NaN)
    error = FaceVelocity(mesh; init = NaN)
    for face_index in mixed_faces
        face = mesh.faces[face_index]
        axis = face.axis
        x, y = face.center
        exact =
            manufactured_pressure_gradient(x, y, 2)[axis] / density[2] -
            manufactured_pressure_gradient(x, y, 1)[axis] / density[1]
        numerical.interior[face_index] = workspace.jump[face_index]
        analytical.interior[face_index] = exact
        error.interior[face_index] = workspace.jump[face_index] - exact
    end

    dual_volumes = TwoPhaseAMR._uncut_face_dual_volumes(mesh, Float64)
    component_norms = ntuple(2) do axis
        error_squared = 0.0
        error_maximum = 0.0
        total_weight = 0.0
        for face_index in mixed_faces
            mesh.faces[face_index].axis == axis || continue
            value = error.interior[face_index]
            weight = dual_volumes[face_index]
            error_squared += weight * abs2(value)
            error_maximum = max(error_maximum, abs(value))
            total_weight += weight
        end
        (
            l2 = total_weight > 0 ?
                sqrt(error_squared / total_weight) : 0.0,
            linf = error_maximum,
        )
    end
    return (;
        numerical = (FaceField{1}(numerical), FaceField{2}(numerical)),
        analytical = (
            FaceField{1}(analytical), FaceField{2}(analytical),
        ),
        error = (FaceField{1}(error), FaceField{2}(error)),
        norms = (x = component_norms[1], y = component_norms[2]),
    )
end

"""Surface-RMS and maximum normal/tangential values of exact `[grad(p)/rho]`."""
function _analytical_jump_components(density; samples = 4096)
    normal_squared = 0.0
    tangential_squared = 0.0
    normal_maximum = 0.0
    tangential_maximum = 0.0
    for sample in 0:(samples - 1)
        angle = 2pi * sample / samples
        normal = (cos(angle), sin(angle))
        tangent = (-normal[2], normal[1])
        x = MANUFACTURED_POISSON_CENTRE[1] +
            MANUFACTURED_POISSON_RADIUS * normal[1]
        y = MANUFACTURED_POISSON_CENTRE[2] +
            MANUFACTURED_POISSON_RADIUS * normal[2]
        liquid = manufactured_pressure_gradient(x, y, 1)
        gas = manufactured_pressure_gradient(x, y, 2)
        jump = (
            gas[1] / density[2] - liquid[1] / density[1],
            gas[2] / density[2] - liquid[2] / density[1],
        )
        normal_value = jump[1] * normal[1] + jump[2] * normal[2]
        tangential_value = jump[1] * tangent[1] + jump[2] * tangent[2]
        normal_squared += abs2(normal_value)
        tangential_squared += abs2(tangential_value)
        normal_maximum = max(normal_maximum, abs(normal_value))
        tangential_maximum = max(
            tangential_maximum, abs(tangential_value),
        )
    end
    normal_rms = sqrt(normal_squared / samples)
    tangential_rms = sqrt(tangential_squared / samples)
    return (;
        normal = (rms = normal_rms, linf = normal_maximum),
        tangential = (
            rms = tangential_rms, linf = tangential_maximum,
        ),
        rms_ratio = tangential_rms / normal_rms,
    )
end

function _manufactured_poisson_fields(
        pressure, solver, workspace, density,
    )
    mesh = pressure.mesh
    exact_pressure = CellField(mesh, manufactured_pressure)
    displayed_pressure, pressure_error, gauge_offset =
        _gauge_aligned_pressure(pressure, exact_pressure)

    # The projection workspace stores grad(p)/rho and its face-normal jump
    # xi directly. Multiplication by the phase density recovers the physical
    # grad(p) displayed by the phase-gradient views.
    pressure_gradient = (
        _multiply_velocity(workspace.liquid_gradient, density[1]),
        _multiply_velocity(workspace.gas_gradient, density[2]),
    )
    exact_pressure_gradient = ntuple(2) do phase
        FaceVelocity(
            mesh,
            (x, y, time, axis) ->
                manufactured_pressure_gradient(x, y, phase)[axis],
        )
    end
    pressure_gradient_error = ntuple(2) do phase
        _velocity_difference(
            pressure_gradient[phase], exact_pressure_gradient[phase],
        )
    end
    face_data = TwoPhaseAMR._mimetic_face_data(solver, workspace)
    xi = _xi_fields(workspace, mesh, density, face_data.mixed)
    errors = (;
        pressure = _pressure_error_norms(pressure_error),
        gradient = _gradient_error_norms(
            pressure_gradient_error, mesh, face_data,
        ),
        xi = xi.norms,
    )
    return (;
        pressure = displayed_pressure,
        exact_pressure,
        pressure_error,
        pressure_gradient,
        exact_pressure_gradient,
        pressure_gradient_error,
        xi = xi.numerical,
        analytical_xi = xi.analytical,
        xi_error = xi.error,
        gauge_offset,
        errors,
    )
end

function _manufactured_poisson_visualization(interface, fields)
    registry = VisualizationFieldRegistry()
    register_field!(
        registry, :pressure, fields.pressure;
        label = "Pressure after projection", colormap = :balance,
    )
    register_field!(
        registry, :pressure_gradient_x, fields.pressure_gradient,
        interface, Face{1}();
        label = "Pressure gradient x", colormap = :balance,
    )
    register_field!(
        registry, :pressure_gradient_y, fields.pressure_gradient,
        interface, Face{2}();
        label = "Pressure gradient y", colormap = :balance,
    )
    register_field!(
        registry, :pressure_error, fields.pressure_error;
        label = "Signed pressure error", colormap = :balance,
    )
    register_field!(
        registry, :pressure_gradient_error_x,
        fields.pressure_gradient_error, interface, Face{1}();
        label = "Pressure gradient error x", colormap = :balance,
    )
    register_field!(
        registry, :pressure_gradient_error_y,
        fields.pressure_gradient_error, interface, Face{2}();
        label = "Pressure gradient error y", colormap = :balance,
    )
    register_field!(
        registry, :xi_error_x, fields.xi_error[1];
        label = "Scaled gradient-jump error xi x", colormap = :balance,
    )
    register_field!(
        registry, :xi_error_y, fields.xi_error[2];
        label = "Scaled gradient-jump error xi y", colormap = :balance,
    )
    viewer = TwoPhaseAMR.visualize(
        registry; field = :pressure, show_blocks = false,
        show_cells = false,
    )
    return (; registry, viewer)
end

"""
    run_example(; resolution=32, density=(1000.0, 1.0),
                method=:commuting, visualize=false, ...)

Solve the periodic manufactured two-velocity Poisson problem. The prescribed
predictor is the sum of a divergence-free velocity and `grad(p)/rho`; one
projection should therefore recover the manufactured pressure and velocity.
The default densities are `(rho_l,rho_g)=(1000,1)`. Equal-amplitude periodic
pressure variations in `x` and `y`, together with the circular interface,
give nearly equal normal and tangential interface-RMS values of the scaled
jump `xi=[grad(p)/rho]`. They are returned in
`analytical_jump_components`.

The optional visualization contains the projected pressure, both components
of the phase pressure gradient, the signed pressure and phase-gradient errors,
and both components of the scaled gradient-jump error
`xi_h - xi`, where `xi=[grad(p)/rho]`. The jump errors are shown only on mixed
staggered faces. Pressure is shifted by its volume-mean error before display
because the periodic Poisson problem determines it only up to a constant.
Additional keywords are passed to the selected projection solver.
Set `method=:cmdgfm` for CMDGFM, `:mimetic` for the default jointly fitted
smoothed-aperture projection, or `:joint_face` for its face-row variant.
"""
function run_example(;
        resolution = 32,
        density = MANUFACTURED_POISSON_DENSITY,
        method = :mimetic,
        tolerance = 1.0e-9,
        max_iterations = 5000,
        visualize = false,
        kwargs...,
    )
    mesh = BlockTreeMesh(
        block_size = (resolution, resolution), periodic = (true, true),
    )
    interface = InterfaceField(
        mesh,
        (x, y) -> hypot(
            x - MANUFACTURED_POISSON_CENTRE[1],
            y - MANUFACTURED_POISSON_CENTRE[2],
        ) - MANUFACTURED_POISSON_RADIUS;
        lipschitz = 1,
    )
    predictor = _manufactured_predictor(mesh, density)
    projected_velocity = (copy(predictor[1]), copy(predictor[2]))
    raw_pressure = CellField(mesh)
    solver, workspace = if method === :cmdgfm
        selected_solver = CMDGFMProjectionSolver(
            interface; density, kwargs...,
        )
        selected_solver, CMDGFMProjectionWorkspace(selected_solver)
    elseif method in (:mimetic, :joint_face)
        selected_solver = MimeticProjectionSolver(
            interface; density,
            variant = method === :mimetic ? :joint_smoothed : :joint_face,
            kwargs...,
        )
        selected_solver, MimeticProjectionWorkspace(selected_solver)
    else
        throw(
            ArgumentError(
                "method must be :cmdgfm, :mimetic, or :joint_face",
            ),
        )
    end
    stats = project!(
        projected_velocity, raw_pressure, solver, 1.0, workspace;
        tolerance, max_iterations,
    )
    fields = _manufactured_poisson_fields(
        raw_pressure, solver, workspace, density,
    )
    analytical_jump_components = _analytical_jump_components(density)
    visualization = visualize ?
        _manufactured_poisson_visualization(interface, fields) : nothing
    return (;
        mesh, interface, predictor, projected_velocity, raw_pressure,
        method, density, solver, workspace, stats, fields...,
        analytical_jump_components, visualization,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    using GLMakie
    GLMakie.activate!()
    run = run_example(visualize = true)
    println(run.stats)
    println("pressure errors: ", run.errors.pressure)
    println("gradient errors: ", run.errors.gradient)
    println("analytical scaled-jump components: ",
        run.analytical_jump_components)
    println(
        "Select pressure, pressure-gradient, or error fields from the menu.",
    )
    scene = visualization_figure(run.visualization.viewer).scene
    scene.events.window_open[] && wait(scene)
end

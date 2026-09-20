using Printf
using TwoPhaseAMR

const POISSON_RADIUS = 0.3
const POISSON_CENTRE = (0.5, 0.5)
const POISSON_PRESSURE_SCALE = 40.0

function exact_velocity(x, y, phase)
    phase == 2 && return (0.0, 0.0)
    dx = x - POISSON_CENTRE[1]
    dy = y - POISSON_CENTRE[2]
    factor = cospi(2(dx^2 + dy^2))
    return (2dy * factor, -2dx * factor)
end

function exact_pressure_gradient(x, y, phase, density)
    # A periodic background replaces the affine x-y term of pt2_poisson.tex;
    # the circular pressure jump and solenoidal final velocity are unchanged.
    gx = 2pi * cospi(2x)
    gy = -pi * sinpi(2y)
    if phase == 1
        dx = x - POISSON_CENTRE[1]
        dy = y - POISSON_CENTRE[2]
        level = dx^2 + dy^2 - POISSON_RADIUS^2
        factor = x - y
        gx += POISSON_PRESSURE_SCALE * (2dx * factor + level)
        gy += POISSON_PRESSURE_SCALE * (2dy * factor - level)
    end
    return (gx / density[phase], gy / density[phase])
end

function manufactured_velocity(mesh, density)
    return ntuple(2) do phase
        FaceVelocity(
            mesh,
            (x, y, time, axis) ->
                exact_velocity(x, y, phase)[axis] +
                exact_pressure_gradient(x, y, phase, density)[axis],
        )
    end
end

function error_norms(velocity, mesh, data)
    dual_volumes = TwoPhaseAMR._uncut_face_dual_volumes(mesh, Float64)
    error_squared = exact_squared = weight = 0.0
    error_maximum = exact_maximum = 0.0
    for (face_index, face) in pairs(mesh.faces)
        fractions = (
            data.liquid_fraction[face_index],
            1 - data.liquid_fraction[face_index],
        )
        for phase in 1:2
            fraction = fractions[phase]
            local_weight = fraction * dual_volumes[face_index]
            exact = exact_velocity(face.center..., phase)[face.axis]
            error = velocity[phase].interior[face_index] - exact
            error_squared += local_weight * abs2(error)
            exact_squared += local_weight * abs2(exact)
            weight += local_weight
            if fraction > sqrt(eps(Float64))
                error_maximum = max(error_maximum, abs(error))
                exact_maximum = max(exact_maximum, abs(exact))
            end
        end
    end
    absolute_l2 = sqrt(error_squared / weight)
    absolute_linf = error_maximum
    return (;
        absolute_l2, absolute_linf,
        relative_l2 = sqrt(error_squared / exact_squared),
        relative_linf = error_maximum / exact_maximum,
    )
end

function mimetic_gradient_decomposition(
        solver, workspace, mesh, data, density,
    )
    active_faces = workspace.active_faces
    incidence = TwoPhaseAMR._mimetic_incidence(mesh, Float64)
    pressure_indices = workspace.pressure_indices
    pressure_values = view(
        workspace.solution, 1:length(pressure_indices),
    )
    dual_volumes = isnothing(solver.interface.geometry) ?
        TwoPhaseAMR._uncut_face_dual_volumes(mesh, Float64) :
        TwoPhaseAMR._geometry_face_dual_volumes(
            solver.interface.geometry,
        )
    relative_diagonal = data.relative_mass[active_faces] .*
        dual_volumes[active_faces]
    pressure_jump = data.coupling[active_faces] .*
        (incidence[:, pressure_indices] * pressure_values)[active_faces] ./
        relative_diagonal

    common_error = jump_error = pressure_jump_error = zero(Float64)
    multiplier_magnitude = zero(Float64)
    for (local_face, global_face) in pairs(active_faces)
        face = mesh.faces[global_face]
        liquid = exact_pressure_gradient(
            face.center..., 1, density,
        )[face.axis]
        gas = exact_pressure_gradient(
            face.center..., 2, density,
        )[face.axis]
        liquid_mass = data.liquid_mass[global_face]
        gas_mass = data.gas_mass[global_face]
        exact_common = (
            liquid_mass * liquid + gas_mass * gas
        ) / (liquid_mass + gas_mass)
        exact_jump = gas - liquid
        numerical_jump = workspace.jump[global_face]
        multiplier_jump = numerical_jump - pressure_jump[local_face]
        common_error = max(
            common_error,
            abs(
                workspace.pressure_gradient.interior[global_face] -
                    exact_common
            ),
        )
        jump_error = max(jump_error, abs(numerical_jump - exact_jump))
        pressure_jump_error = max(
            pressure_jump_error,
            abs(pressure_jump[local_face] - exact_jump),
        )
        multiplier_magnitude = max(
            multiplier_magnitude, abs(multiplier_jump),
        )
    end
    return (;
        common_linf = common_error, jump_linf = jump_error,
        pressure_jump_linf = pressure_jump_error,
        multiplier_jump_linf = multiplier_magnitude,
    )
end

function run_case(
        resolution, method; density = (10.0, 1.0),
        commuting_accuracy_factor = 1.0,
    )
    mesh = BlockTreeMesh(
        block_size = (resolution, resolution), periodic = (true, true),
    )
    interface = InterfaceField(
        mesh,
        (x, y) -> hypot(
            x - POISSON_CENTRE[1], y - POISSON_CENTRE[2],
        ) - POISSON_RADIUS;
        lipschitz = 1,
    )
    velocity = manufactured_velocity(mesh, density)
    if method === :cmdgfm
        solver = CMDGFMProjectionSolver(interface; density)
        workspace = CMDGFMProjectionWorkspace(solver)
    else
        solver = MimeticProjectionSolver(
            interface; density, jump_interpolant = method,
            commuting_accuracy_factor,
        )
        workspace = MimeticProjectionWorkspace(solver)
    end
    stats = project!(
        velocity, CellField(mesh), solver, 1.0, workspace;
        tolerance = 1.0e-9, max_iterations = 5000,
    )
    data = TwoPhaseAMR._mimetic_face_data(solver, workspace)
    norms = error_norms(velocity, mesh, data)
    decomposition = stats isa MimeticProjectionStats ?
        mimetic_gradient_decomposition(
            solver, workspace, mesh, data, density,
        ) :
        (;
            common_linf = NaN, jump_linf = NaN,
            pressure_jump_linf = NaN, multiplier_jump_linf = NaN,
        )
    diagnostic = if stats isa MimeticProjectionStats
        (;
            commuting = stats.commuting_constraint_rows,
            exact = stats.exact_commuting_constraint_rows,
            approximate = stats.approximate_commuting_constraint_rows,
            fallback = stats.fallback_constraint_rows,
            residual = stats.commuting_residual,
            pressure_commuting =
                stats.pressure_commuting_constraint_rows,
            pressure_fallback = stats.pressure_fallback_constraint_rows,
            pressure_residual = stats.pressure_commuting_residual,
        )
    else
        (; commuting = 0, exact = 0, approximate = 0,
            fallback = 0, residual = 0.0, pressure_commuting = 0,
            pressure_fallback = 0, pressure_residual = 0.0)
    end
    return (;
        resolution, method, norms..., decomposition..., stats,
        diagnostic...,
    )
end

function run_study(
        resolutions = (8, 16, 32);
        methods = (:cmdgfm, :divergence, :face, :cell, :commuting),
    )
    results = [run_case(n, method) for method in methods for n in resolutions]
    println(
        "method          N       rel-L2     rel-Linf exact approx fallback " *
        "residual p-fit p-fallback p-residual",
    )
    for result in results
        @printf(
            "%-12s %4d  %10.3e  %10.3e %5d %6d %8d %8.1e %5d %10d %10.1e\n",
            string(result.method), result.resolution, result.relative_l2,
            result.relative_linf, result.exact, result.approximate,
            result.fallback, result.residual, result.pressure_commuting,
            result.pressure_fallback, result.pressure_residual,
        )
    end
    println(
        "\nAbsolute Linf decomposition on active mixed faces " *
        "(mimetic methods)",
    )
    println(
        "method          N       common   total-jump   pressure-jump  " *
        "multiplier-mag",
    )
    for result in results
        result.stats isa MimeticProjectionStats || continue
        @printf(
            "%-12s %4d  %11.3e %12.3e %15.3e %15.3e\n",
            string(result.method), result.resolution,
            result.common_linf, result.jump_linf,
            result.pressure_jump_linf, result.multiplier_jump_linf,
        )
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_study()
end

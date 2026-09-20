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

function run_case(resolution, method; density = (10.0, 1.0))
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
        )
        workspace = MimeticProjectionWorkspace(solver)
    end
    stats = project!(
        velocity, CellField(mesh), solver, 1.0, workspace;
        tolerance = 1.0e-9, max_iterations = 5000,
    )
    data = TwoPhaseAMR._mimetic_face_data(solver, workspace)
    norms = error_norms(velocity, mesh, data)
    diagnostic = if stats isa MimeticProjectionStats
        (;
            commuting = stats.commuting_constraint_rows,
            fallback = stats.fallback_constraint_rows,
            residual = stats.commuting_residual,
        )
    else
        (; commuting = 0, fallback = 0, residual = 0.0)
    end
    return (; resolution, method, norms..., stats, diagnostic...)
end

function run_study(
        resolutions = (8, 16, 32);
        methods = (:cmdgfm, :divergence, :face, :cell, :commuting),
    )
    results = [run_case(n, method) for method in methods for n in resolutions]
    println(
        "method          N       rel-L2     rel-Linf   commute fallback residual",
    )
    for result in results
        @printf(
            "%-12s %4d  %10.3e  %10.3e  %7d %8d %8.1e\n",
            string(result.method), result.resolution, result.relative_l2,
            result.relative_linf, result.commuting, result.fallback,
            result.residual,
        )
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_study()
end

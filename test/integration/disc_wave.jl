using Test, TwoPhaseAMR
using LinearAlgebra: dot

# The exact Bessel standing waves of example 13 and its error measures.
include(joinpath(@__DIR__, "..", "..", "examples", "13_disc_wave", "disc_modes.jl"))
include(joinpath(@__DIR__, "..", "..", "examples", "13_disc_wave", "disc_wave_analysis.jl"))

# The implicit-midpoint wave solver of example 13 on the package API: both
# components of u_tt = Δu on the disc r < 1 in the box [-1.25, 1.25]² (n × n
# cells in 8 × 8 blocks), `NoSlip()` on the circle, from the discrete Helmholtz
# projection of the default modes, one shifted solve per step.
function disc_wave_integration_run(
        n, final_time, steps; closure = LSStagClosure(), tolerance = 1.0e-12,
    )
    mesh = BlockTreeMesh(;
        base_mesh = (n, n), block_size = (8, 8), origin = (-1.25, -1.25),
        extent = (2.5, 2.5), periodic = (false, false),
    )
    geometry = GeometryField(mesh, (x, y) -> 1.0 - hypot(x, y); lipschitz = 1)
    operator = StaggeredDiffusionOperator(geometry; boundary = NoSlip(), closure)
    shifted(diffusion) = StaggeredEllipticSolver(
        StaggeredEllipticOperator(operator; reaction = 1, diffusion);
        preconditioner_options = (; geometry),
    )
    modes = DISC_WAVE_MODES
    displacement, rate, midpoint, source = (FaceVelocity(mesh) for _ in 1:4)
    shift = (2.5 / n)^2
    for axis in 1:2
        nodes = value_points(operator, axis)
        for (index, face) in pairs(nodes.faces)
            iszero(nodes.volumes[index]) && continue
            x, y = nodes.points[index]
            source.interior[face] = sum(
                mode -> mode.axis == axis ?
                    (1 + shift * (mode.root / 1.0)^2) * mode_shape(mode, x, y, 1.0) : 0.0,
                modes,
            )
        end
    end
    copyto!(displacement.interior, source.interior)
    solve!(displacement, shifted(shift), source; tolerance, max_cycles = 500).converged ||
        error("Helmholtz projection did not converge")
    dt = final_time / steps
    solver = shifted((dt / 2)^2)
    function energy()
        total = 0.0
        for component in (operator.x, operator.y)
            values = displacement.interior[component.local_to_global]
            kinetic = 0.0
            for (index, face) in pairs(component.local_to_global)
                kinetic += component.volumes[index] * rate.interior[face]^2
            end
            total += kinetic + dot(values, component.stiffness * values)
        end
        return total / 2
    end
    energies = [energy()]
    tracker = PhaseTracker()
    state(step) = (; step, displacement, rate, operator, modes, radius = 1.0, speed = 1.0)
    tracker(state(0))
    for step in 1:steps
        @. source.interior = displacement.interior + dt / 2 * rate.interior
        copyto!(midpoint.interior, source.interior)
        solve!(midpoint, solver, source; tolerance, max_cycles = 200).converged ||
            error("implicit-midpoint solve did not converge")
        @. rate.interior = 4 / dt * (midpoint.interior - displacement.interior) - rate.interior
        @. displacement.interior = 2midpoint.interior - displacement.interior
        push!(energies, energy())
        tracker(state(step))
    end
    run = (;
        operator, displacement, rate, modes, radius = 1.0, speed = 1.0,
        time = final_time, dt, steps,
    )
    return (;
        run..., errors = disc_wave_errors(run), frequencies = mode_frequencies(tracker, run),
        drift = maximum(abs, energies ./ first(energies) .- 1),
        explicit_limit = explicit_limit(operator),
    )
end

const DISC_WAVE_NORMS = (:displacement_l2, :displacement_max, :rate_l2, :rate_max)

disc_wave_frequency_errors(run) =
    [abs(entry.spatial / entry.exact - 1) for entry in run.frequencies]

@testset "Disc modes: Bessel zeros and the Dirichlet wall" begin
    # Abramowitz and Stegun, table 9.5.
    for (m, k, value) in (
            (0, 1, 2.404825557695773), (1, 1, 3.831705970207512),
            (2, 1, 5.135622301840683), (3, 2, 9.76102312998167),
        )
        @test bessel_zero(m, k) ≈ value rtol = 4eps()
    end
    for mode in DISC_WAVE_MODES, angle in range(0, 2pi; length = 7)
        @test abs(mode_shape(mode, cos(angle), sin(angle), 1.0)) <= 1.0e-14
    end
end

@testset "Disc wave: LS-STAG is second order in space" begin
    # The README's study: t = 1 in 2000 steps, solver tolerance 1e-12. Each
    # relative error e(n) is at most C (32/n)² plus 5 %, C the largest
    # e(n) (n/32)² of the README table over 32–256 cells. Measured orders from
    # 32 to 64 and 64 to 128 cells: u 1.91/1.99 (L2) and 1.89/1.97 (max),
    # ∂u/∂t 1.98/1.98 in both norms, frequencies 1.91–1.99.
    resolutions = (32, 64, 128)
    runs = [disc_wave_integration_run(n, 1.0, 2000) for n in resolutions]
    constants = (; displacement_l2 = 2.191e-2, displacement_max = 3.376e-2, rate_l2 = 0.1794, rate_max = 0.2831)
    for name in DISC_WAVE_NORMS
        errors = [relative_error(run.errors, name) for run in runs]
        println("disc_wave LS-STAG $name errors=$errors orders=$(log2.(errors[1:2] ./ errors[2:3]))")
        @test all(errors .<= 1.05 .* constants[name] .* (32 ./ collect(resolutions)) .^ 2)
        @test all(log2.(errors[1:2] ./ errors[2:3]) .>= 1.8)
    end
    frequencies = disc_wave_frequency_errors.(runs)
    @test all(log2.(frequencies[1] ./ frequencies[2]) .>= 1.8)
    @test all(log2.(frequencies[2] ./ frequencies[3]) .>= 1.8)
    # Each solve changes the energy by at most about its relative tolerance.
    @test all(run -> run.drift <= run.steps * 1.0e-12, runs)

    # The legacy wall-distance closure is first order at the curved wall:
    # measured order 1.03, errors 3.3 and 6.1 times the LS-STAG ones, and a
    # (0,1) frequency error 14 times larger at 32 cells.
    wall = [disc_wave_integration_run(n, 1.0, 2000; closure = WallDistanceClosure()) for n in (32, 64)]
    errors = [relative_error(run.errors, :displacement_l2) for run in wall]
    println("disc_wave wall distance displacement_l2 errors=$errors order=$(log2(errors[1] / errors[2]))")
    @test log2(errors[1] / errors[2]) < 1.5
    @test all(errors .> 3 .* [relative_error(run.errors, :displacement_l2) for run in runs[1:2]])
    @test disc_wave_frequency_errors(wall[1])[1] > 10 * frequencies[1][1]
end

@testset "Disc wave: implicit midpoint is second order in time" begin
    # Differences at t = 1 on 32 cells to 3200 steps, relative to the exact
    # norms, for u and ∂u/∂t in L2 and max. Measured orders 1.93–2.00 from 50
    # to 200 steps (1.70–1.98 from 25 to 50).
    reference = disc_wave_integration_run(32, 1.0, 3200)
    norms = reference.errors.norms
    weights = zeros(length(reference.displacement.interior))
    for axis in 1:2
        nodes = value_points(reference.operator, axis)
        weights[nodes.faces] .= nodes.volumes
    end
    fluid = weights .> 0
    function difference(first, second, l2, maximum_norm)
        error = abs.(first.interior .- second.interior)[fluid]
        return sqrt(sum(weights[fluid] .* error .^ 2) / sum(weights[fluid])) / l2, maximum(error) / maximum_norm
    end
    differences = map((50, 100, 200)) do steps
        run = disc_wave_integration_run(32, 1.0, steps)
        (
            difference(run.displacement, reference.displacement, norms.displacement_l2, norms.displacement_max)...,
            difference(run.rate, reference.rate, norms.rate_l2, norms.rate_max)...,
        )
    end
    orders = [log2.(differences[i] ./ differences[i + 1]) for i in 1:2]
    println("disc_wave time differences=$differences orders=$orders")
    @test all(order -> all(order .>= 1.9), orders)
end

@testset "Disc wave: energy at steps far above the explicit limit" begin
    # One period of the (0,1) mode in 128 steps on 128 cells: 31 times the
    # leapfrog limit of the small cut cells. Measured largest relative energy
    # change 7.4e-12 at tolerance 1e-10.
    run = disc_wave_integration_run(128, 2pi / bessel_zero(0, 1), 128; tolerance = 1.0e-10)
    println("disc_wave large step: dt/limit=$(run.dt / run.explicit_limit) drift=$(run.drift)")
    @test run.dt > 30 * run.explicit_limit
    @test run.drift <= 1.0e-10
end

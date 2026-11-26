using Test, TwoPhaseAMR
using LinearAlgebra: dot

# The exact Bessel standing waves of example 13 and its error measures.
include(joinpath(@__DIR__, "..", "..", "examples", "13_disc_wave", "disc_modes.jl"))
include(joinpath(@__DIR__, "..", "..", "examples", "13_disc_wave", "disc_wave_analysis.jl"))

# The box [-1.25, 1.25]² in n × n cells in 8 × 8 blocks; with `composite` the
# base blocks that the circle r = 1 crosses are refined once (example 13).
function disc_wave_integration_mesh(n; composite = false)
    crosses_wall(lower, upper) =
        hypot(max.(lower, 0.0, .-upper)...) <= 1.0 <=
        hypot(max.(abs.(lower), abs.(upper))...)
    return BlockTreeMesh(;
        base_mesh = (n, n), block_size = (8, 8), origin = (-1.25, -1.25),
        extent = (2.5, 2.5), periodic = (false, false),
        max_level = composite ? 1 : 0,
        refine = (key, origin, extent) ->
        composite && key.level == 0 && crosses_wall(origin, origin .+ extent),
    )
end

# The implicit-midpoint wave solver of example 13 on the package API: both
# components of u_tt = Δu on the disc r < 1, `NoSlip()` on the circle, from the
# discrete Helmholtz projection of the default modes (`initial_data =
# :sampled` starts from their values at the value points instead), one shifted
# solve per step.
function disc_wave_integration_run(
        n, final_time, steps; closure = LSStagClosure(), tolerance = 1.0e-12,
        composite = false, initial_data = :projected,
    )
    mesh = disc_wave_integration_mesh(n; composite)
    geometry = GeometryField(mesh, (x, y) -> 1.0 - hypot(x, y); lipschitz = 1)
    operator = StaggeredDiffusionOperator(geometry; boundary = NoSlip(), closure)
    shifted(diffusion) = StaggeredEllipticSolver(
        StaggeredEllipticOperator(operator; reaction = 1, diffusion);
        preconditioner_options = (; geometry),
    )
    modes = DISC_WAVE_MODES
    displacement, rate, midpoint, source = (FaceVelocity(mesh) for _ in 1:4)
    shift = (2.5 / n)^2
    if initial_data === :sampled
        for axis in 1:2
            nodes = value_points(operator, axis)
            for (index, face) in pairs(nodes.faces)
                iszero(nodes.volumes[index]) && continue
                x, y = nodes.points[index]
                displacement.interior[face] =
                    disc_wave_exact(modes, x, y, 0.0, axis; radius = 1.0)[1]
            end
        end
    else
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
    end
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
    # relative error e(n) is at most 1.05 C (32/n)², where C is the largest
    # scaled error e(n) (n/32)² of the README's 32–256 table. That is its
    # 256-cell row (the four scaled rows increase with n): u L2
    # 3.424e-4·64 = 2.191e-2, u max 5.275e-4·64 = 3.376e-2, ∂u/∂t L2
    # 2.803e-3·64 = 0.1794 and ∂u/∂t max 4.423e-3·64 = 0.2831; the
    # 32/64/128-cell rows scale to 2.03/2.15/2.18e-2, 3.03/3.27/3.34e-2,
    # 0.173/0.176/0.177 and 0.272/0.276/0.280. Measured orders from 32 to 64
    # and 64 to 128 cells: u 1.91/1.99 (L2) and 1.89/1.97 (max), ∂u/∂t
    # 1.98/1.98 in both norms, frequencies 1.91–1.99.
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

    # The composite mesh (the blocks that the circle crosses refined once) at
    # 32 and 64 cells, the same study (t = 1 in 2000 steps). Measured orders
    # from 32 to 64 cells 1.92 (u, L2), 2.16 (u, max), 1.83 (∂u/∂t, L2) and
    # 1.75 (∂u/∂t, max), and composite/uniform error ratios
    # 0.945/1.174/0.858/0.844 at 32 cells and 0.940/0.971/0.950/0.985 at 64.
    # The README reports orders 1.91–1.95 and errors within 6 % below to 2 %
    # above the uniform ones for its 128→256 pair, where the refined ring is a
    # smaller fraction of the mesh; the band holds at 64 cells here, while the
    # 32-cell maximum norm is 17 % above the uniform one. The order gate is
    # 1.65 against the measured 1.75 of the ∂u/∂t maximum norm at the coarse
    # pair (64→128 costs 100 s and is not run here). The two runs add about
    # 33 s to the group here (8.4 s and 24.9 s, one thread).
    composite = [disc_wave_integration_run(n, 1.0, 2000; composite = true) for n in (32, 64)]
    for name in DISC_WAVE_NORMS
        errors = [relative_error(run.errors, name) for run in composite]
        uniform = [relative_error(run.errors, name) for run in runs[1:2]]
        println(
            "disc_wave composite $name errors=$errors ",
            "order=$(log2(errors[1] / errors[2])) ratios=$(errors ./ uniform)",
        )
        @test log2(errors[1] / errors[2]) >= 1.65
        @test all(ratio -> 0.8 <= ratio <= 1.2, errors ./ uniform)
    end
    @test all(run -> run.drift <= run.steps * 1.0e-12, composite)

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

@testset "Disc wave: sampled initial data converges at second order in u" begin
    # The README's sampled-data study: the exact modes sampled at the value
    # points replace the Helmholtz projection (`initial_data = :sampled`), so
    # the truncation error of the smallest cut cells stays in the stiff modes
    # that the energy-conserving scheme does not damp. t = 1 in 2000 steps,
    # solver tolerance 1e-12. Measured relative errors (u L2, u max,
    # ∂u/∂t L2, ∂u/∂t max) 0.01720/0.02470/0.1750/0.2799 at 32 cells,
    # 0.005117/0.007673/0.04428/0.06915 at 64 and
    # 0.001342/0.002062/0.01114/0.04161 at 128; the README's ∂u/∂t max errors
    # 0.280/0.069/0.042 at 32–128 cells match (all relative to the exact
    # max norm). At 128 cells
    # the u L2 error is 1.3 % below the projected start's 0.001360. C is the
    # largest scaled error e(n) (n/32)² measured here, its 128-cell value, and
    # the 5 % margin is the same one the uniform study uses. The three runs
    # add about 55 s to the group here (2.5 s, 10.1 s and 41.9 s, one thread).
    resolutions = (32, 64, 128)
    sampled = [
        disc_wave_integration_run(n, 1.0, 2000; initial_data = :sampled)
            for n in resolutions
    ]
    constants = (;
        displacement_l2 = 2.147e-2, displacement_max = 3.299e-2, rate_l2 = 0.1782,
    )
    for name in keys(constants)
        errors = [relative_error(run.errors, name) for run in sampled]
        println(
            "disc_wave sampled $name errors=$errors ",
            "orders=$(log2.(errors[1:2] ./ errors[2:3]))",
        )
        @test all(errors .<= 1.05 .* constants[name] .* (32 ./ collect(resolutions)) .^ 2)
        # Measured 1.75/1.93 (u L2), 1.69/1.90 (u max) and 1.98/1.99 (∂u/∂t L2)
        # from 32 to 64 and 64 to 128 cells; the 32-cell pair is gated lower.
        @test log2(errors[1] / errors[2]) >= 1.6
        @test log2(errors[2] / errors[3]) >= 1.8
    end
    # The maximum error of ∂u/∂t stalls instead: 0.2799/0.06915/0.04161
    # absolute (orders 2.02 and 0.73), as the README reports (0.280/0.069/
    # 0.042/0.033 with orders 2.02/0.73/0.32 over four grids): the sampled
    # stiff cut-cell modes keep oscillating at their own frequency, so this
    # norm has no second-order asymptote.
    rate_max = [relative_error(run.errors, :rate_max) for run in sampled]
    println(
        "disc_wave sampled rate_max errors=$rate_max ",
        "orders=$(log2.(rate_max[1:2] ./ rate_max[2:3]))",
    )
    @test log2(rate_max[1] / rate_max[2]) > 1.8
    @test log2(rate_max[2] / rate_max[3]) < 1.5
    @test all(run -> run.drift <= run.steps * 1.0e-12, sampled)
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

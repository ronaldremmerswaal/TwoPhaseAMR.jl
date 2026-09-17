using Test, TwoPhaseAMR

const TV = TwoPhaseAMR

function _periodic_distance(value, centre)
    return mod(value - centre + 0.5, 1) - 0.5
end

function _band_case(resolution, method)
    mesh = BlockTreeMesh(
        block_size = (resolution, resolution), periodic = (true, true),
    )
    levelset = (x, y) ->
        (abs(_periodic_distance(x + y, 0.7)) - 0.18) / sqrt(2)
    interface = InterfaceField(mesh, levelset; lipschitz = 1)
    scalar = PhaseScalar(
        interface;
        liquid = (x, y) -> 1 + 0.2sinpi(2x) + 0.1cospi(2y),
        gas = (x, y) -> 2 + 0.15cospi(2x) - 0.1sinpi(2y),
    )
    one_function = PhaseScalar(interface)
    velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
    workspace = WYAdvectionWorkspace(
        interface, velocity; interface_method = method,
    )
    initial = (
        fraction = copy(interface.fraction),
        liquid = copy(scalar.liquid), gas = copy(scalar.gas),
        volume = integrate(interface),
        liquid_amount = integrate(scalar, :liquid),
        gas_amount = integrate(scalar, :gas),
    )
    return (; mesh, interface, scalar, one_function, velocity, workspace,
        initial)
end

function _set_band_velocity!(case, amplitude)
    inverse_root_two = inv(sqrt(2.0))
    normal = (inverse_root_two, inverse_root_two)
    tangent = (-inverse_root_two, inverse_root_two)
    for phase in 1:2
        tangent_speed = phase == 1 ? 0.08 : -0.1
        vector = (
            0.12normal[1] + tangent_speed * tangent[1],
            0.12normal[2] + tangent_speed * tangent[2],
        )
        component = case.velocity[phase]
        for (index, face) in pairs(case.mesh.faces)
            component.interior[index] = amplitude * vector[face.axis]
        end
        fill!(component.boundary, 0)
    end
    return case.velocity
end

function _band_errors(case, maximum_reset, maximum_partition, cycles, steps)
    fraction_error = scalar_error = one_error = 0.0
    for (block, leaf) in pairs(case.mesh.blocks)
        initial_fraction = interior(case.initial.fraction, block)
        fraction_error += leaf.volume * sum(abs.(
            interior(case.interface.fraction, block) .- initial_fraction
        ))
        scalar_error += leaf.volume * sum(
            initial_fraction .* abs.(
                interior(case.scalar.liquid, block) .-
                    interior(case.initial.liquid, block)
            ) +
                (1 .- initial_fraction) .* abs.(
                    interior(case.scalar.gas, block) .-
                        interior(case.initial.gas, block)
                )
        )
        one_error = max(
            one_error,
            maximum(abs.(interior(case.one_function.liquid, block) .- 1)),
            maximum(abs.(interior(case.one_function.gas, block) .- 1)),
        )
    end
    return (;
        fraction = fraction_error, scalar = scalar_error,
        one = one_error,
        volume = abs(integrate(case.interface) - case.initial.volume),
        liquid_amount = abs(
            integrate(case.scalar, :liquid) - case.initial.liquid_amount
        ),
        gas_amount = abs(
            integrate(case.scalar, :gas) - case.initial.gas_amount
        ),
        reset = maximum_reset, partition = maximum_partition,
        projection_cycles = cycles, steps,
    )
end

function _advance_band!(case, maximum_dt)
    time = 0.0
    step = 0
    maximum_reset = 0.0
    maximum_partition = 0.0
    maximum_projection_cycles = 0
    while time < 1 - 16eps()
        dt = min(maximum_dt, 1 - time)
        _set_band_velocity!(case, cospi(time + dt / 2))
        advect!(
            case.interface, case.velocity, dt, case.workspace;
            scalars = (case.scalar, case.one_function),
            sweep_order = iseven(step) ? (1, 2) : (2, 1),
            scheme = Fromm(),
        )
        diagnostics = phase_flux_diagnostics(case.workspace)
        maximum_reset = max(
            maximum_reset,
            diagnostics.absolute_gas_reset_discrepancy,
        )
        maximum_partition = max(
            maximum_partition, diagnostics.maximum_partition_defect,
        )
        maximum_projection_cycles = max(
            maximum_projection_cycles, diagnostics.projection_iterations,
        )
        time += dt
        step += 1
    end
    return _band_errors(
        case, maximum_reset, maximum_partition,
        maximum_projection_cycles, step,
    )
end

function _observed_rates(values, resolutions)
    return [
        log(values[index] / values[index + 1]) /
            log(resolutions[index + 1] / resolutions[index])
            for index in 1:(length(values) - 1)
    ]
end

@testset "Two-velocity split-WY reversal convergence" begin
    resolutions = (16, 24, 32)
    methods = (
        projected = ProjectedLiquidVelocity(
            tolerance = 1e-12, max_cycles = 500,
        ),
        frozen = FrozenInterfaceFluxes(),
    )
    results = Dict{Symbol, Vector{NamedTuple}}(
        name => NamedTuple[] for name in keys(methods)
    )

    for resolution in resolutions
        cases = map(method -> _band_case(resolution, method), methods)
        limits = map(cases) do case
            _set_band_velocity!(case, 1.0)
            stable_timestep(case.velocity, case.workspace; cfl = 0.35)
        end
        common_dt = minimum(limits)
        for name in keys(methods)
            result = _advance_band!(cases[name], common_dt)
            push!(results[name], result)
            println(
                "two_velocity_reversal method=$name n=$resolution " *
                    join(("$key=$(getproperty(result, key))" for key in keys(result)), " "),
            )
        end
        @test last(results[:projected]).steps ==
            last(results[:frozen]).steps
    end

    for name in keys(methods)
        method_results = results[name]
        fraction_errors = getproperty.(method_results, :fraction)
        scalar_errors = getproperty.(method_results, :scalar)
        fraction_rates = _observed_rates(fraction_errors, resolutions)
        scalar_rates = _observed_rates(scalar_errors, resolutions)
        println(
            "two_velocity_convergence method=$name n=$resolutions " *
                "fraction=$fraction_errors fraction_rates=$fraction_rates " *
                "scalar=$scalar_errors scalar_rates=$scalar_rates",
        )
        @test all(diff(fraction_errors) .< 0)
        @test all(diff(scalar_errors) .< 0)
        @test fraction_rates[end] >= 1.8
        @test scalar_rates[end] >= 1.5
        @test all(iszero, getproperty.(method_results, :one))
        @test all(zip(method_results, resolutions)) do (result, resolution)
            result.volume <= resolution^2 * eps()
        end
    end

    projected = results[:projected]
    reset = getproperty.(projected, :reset)
    partition = getproperty.(projected, :partition)
    @test all(diff(reset) .< 0)
    @test all(diff(partition) .< 0)
    @test _observed_rates(reset, resolutions)[end] >= 0.8
    @test _observed_rates(partition, resolutions)[end] >= 1.8
    @test all(result -> result.projection_cycles == 0, projected)

    frozen = results[:frozen]
    @test all(result -> result.partition <= eps(), frozen)
end

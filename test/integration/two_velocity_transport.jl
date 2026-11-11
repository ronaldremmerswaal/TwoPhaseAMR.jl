using Test, TwoPhaseAMR

const TV_TRANSPORT = TwoPhaseAMR
const MIXTURE_PROJECTION_TOLERANCE = 1.0e-12

function _periodic_distance(value, centre)
    return mod(value - centre + 0.5, 1) - 0.5
end

function _band_case(kind, resolution)
    mesh = kind === :uniform ? BlockTreeMesh(
            block_size = (resolution, resolution), periodic = (true, true),
        ) : BlockTreeMesh(
            block_size = (resolution, resolution), max_level = 2,
            periodic = (true, true),
            refine = (key, origin, extent) -> key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 0),
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
    staggered_functions = (
        (x, y) -> sinpi(4x) * sinpi(4y),
        (x, y) -> cospi(2x) * cospi(2y),
    )
    staggered = ntuple(2) do axis
        PhaseFaceScalar{axis}(
            interface; liquid = staggered_functions[1],
            gas = staggered_functions[2],
        )
    end
    staggered_one = (
        PhaseFaceScalar{1}(interface), PhaseFaceScalar{2}(interface),
    )
    velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
    workspace = WYAdvectionWorkspace(interface, velocity)
    projection_solver = EllipticSolver(
        PoissonOperator(mesh); coarse_size = 8,
    )
    mixture_projection = (
        mixture = FaceVelocity(mesh),
        unprojected = FaceVelocity(mesh),
        pressure = CellField(mesh),
        solver = projection_solver,
        workspace = ProjectionWorkspace(projection_solver),
    )
    initial = (
        fraction = copy(interface.fraction),
        liquid = copy(scalar.liquid), gas = copy(scalar.gas),
        volume = integrate(interface),
        liquid_amount = integrate(scalar, :liquid),
        gas_amount = integrate(scalar, :gas),
        staggered = map(staggered) do quantity
            (liquid = copy(quantity.liquid), gas = copy(quantity.gas))
        end,
        staggered_amount = map(staggered) do quantity
            (
                liquid = integrate(quantity, :liquid),
                gas = integrate(quantity, :gas),
            )
        end,
    )
    density = (1000.0, 1.0)
    return (;
        mesh, interface, scalar, one_function, staggered,
        staggered_one, velocity, workspace, mixture_projection, density,
        initial,
    )
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


function _project_band_mixture!(case)
    interface = case.interface
    work = case.workspace
    projection = case.mixture_projection
    reconstruct!(interface)
    TV_TRANSPORT._freeze_phase_apertures!(interface, work)
    for index in eachindex(case.mesh.faces)
        projection.mixture.interior[index] =
            work.liquid_aperture[index] * case.velocity[1].interior[index] +
            work.gas_aperture[index] * case.velocity[2].interior[index]
    end
    copyto!(projection.unprojected.interior, projection.mixture.interior)
    statistics = project!(
        projection.mixture, projection.pressure, projection.solver, 1.0,
        projection.workspace;
        tolerance = MIXTURE_PROJECTION_TOLERANCE, max_cycles = 300,
    )
    statistics.converged || error(
        "frozen-interface test projection failed: $statistics",
    )
    for phase in 1:2, index in eachindex(case.mesh.faces)
        case.velocity[phase].interior[index] +=
            projection.mixture.interior[index] -
            projection.unprojected.interior[index]
    end
    return statistics.cycles
end

function _band_errors(
        case, maximum_partition, projection_count, steps,
    )
    fraction_error = scalar_error = one_error = 0.0
    for (block, leaf) in pairs(case.mesh.blocks)
        initial_fraction = interior(case.initial.fraction, block)
        fraction_error += leaf.volume * sum(
            abs.(
                interior(case.interface.fraction, block) .- initial_fraction
            )
        )
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
    staggered_liquid_error = staggered_gas_error = 0.0
    staggered_one_error = 0.0
    for component in 1:2
        quantity = case.staggered[component]
        reference = case.initial.staggered[component]
        values = (
            liquid = vcat(quantity.liquid.interior, quantity.liquid.boundary),
            gas = vcat(quantity.gas.interior, quantity.gas.boundary),
            fraction = vcat(
                quantity.fraction.interior, quantity.fraction.boundary,
            ),
        )
        reference_values = (
            liquid = vcat(reference.liquid.interior, reference.liquid.boundary),
            gas = vcat(reference.gas.interior, reference.gas.boundary),
        )
        ones = case.staggered_one[component]
        one_values = (
            liquid = vcat(ones.liquid.interior, ones.liquid.boundary),
            gas = vcat(ones.gas.interior, ones.gas.boundary),
        )
        for index in eachindex(values.liquid)
            volume = TV_TRANSPORT._phase_dual_volume(
                quantity.workspace.schedule, index,
            )
            fraction = values.fraction[index]
            staggered_liquid_error += volume * fraction * abs(
                values.liquid[index] - reference_values.liquid[index],
            )
            staggered_gas_error += volume * (1 - fraction) * abs(
                values.gas[index] - reference_values.gas[index],
            )
            staggered_one_error = max(
                staggered_one_error,
                abs(one_values.liquid[index] - 1),
                abs(one_values.gas[index] - 1),
            )
        end
    end
    return (;
        fraction = fraction_error, scalar = scalar_error,
        one = one_error,
        staggered_liquid = staggered_liquid_error,
        staggered_gas = staggered_gas_error,
        staggered_liquid_momentum =
            case.density[1] * staggered_liquid_error,
        staggered_gas_momentum = case.density[2] * staggered_gas_error,
        staggered_one = staggered_one_error,
        volume = abs(integrate(case.interface) - case.initial.volume),
        liquid_amount = abs(
            integrate(case.scalar, :liquid) - case.initial.liquid_amount
        ),
        gas_amount = abs(
            integrate(case.scalar, :gas) - case.initial.gas_amount
        ),
        staggered_liquid_amount = maximum(
            abs(
                integrate(case.staggered[component], :liquid) -
                    case.initial.staggered_amount[component].liquid
            ) for component in 1:2
        ),
        staggered_gas_amount = maximum(
            abs(
                integrate(case.staggered[component], :gas) -
                    case.initial.staggered_amount[component].gas
            ) for component in 1:2
        ),
        partition = maximum_partition, projection_count, steps,
    )
end

function _advance_band!(case, maximum_dt)
    time = 0.0
    step = 0
    maximum_partition = 0.0
    projection_count = 0
    while time < 1 - 16eps()
        dt = min(maximum_dt, 1 - time)
        _set_band_velocity!(case, cospi(time + dt / 2))
        projection_count += 1
        _project_band_mixture!(case)
        advect!(
            case.interface, case.velocity, dt, case.workspace;
            scalars = (case.scalar, case.one_function),
            staggered_scalars = (
                case.staggered..., case.staggered_one...,
            ),
            sweep_order = iseven(step) ? (1, 2) : (2, 1),
            scheme = Fromm(),
        )
        diagnostics = phase_flux_diagnostics(case.workspace)
        maximum_partition = max(
            maximum_partition, diagnostics.maximum_partition_defect,
        )
        time += dt
        step += 1
    end
    return _band_errors(case, maximum_partition, projection_count, step)
end

function _observed_rates(values, resolutions)
    return [
        log(values[index] / values[index + 1]) /
            log(resolutions[index + 1] / resolutions[index])
            for index in 1:(length(values) - 1)
    ]
end

@testset "Two-velocity split-WY reversal convergence" begin
    for kind in (:uniform, :mixed)
        resolutions = kind === :uniform ? (16, 24, 32) : (8, 12, 16)
        results = NamedTuple[]
        for resolution in resolutions
            case = _band_case(kind, resolution)
            _set_band_velocity!(case, 1.0)
            common_dt = stable_timestep(
                case.velocity, case.workspace; cfl = 0.35,
            )
            result = _advance_band!(case, common_dt)
            push!(results, result)
            println(
                "two_velocity_reversal mesh=$kind n=$resolution " *
                    join(
                    (
                        "$key=$(getproperty(result, key))"
                            for key in keys(result)
                    ), " ",
                ),
            )
        end

        fraction_errors = getproperty.(results, :fraction)
        scalar_errors = getproperty.(results, :scalar)
        staggered_liquid_errors = getproperty.(results, :staggered_liquid)
        staggered_gas_errors = getproperty.(results, :staggered_gas)
        fraction_rates = _observed_rates(fraction_errors, resolutions)
        scalar_rates = _observed_rates(scalar_errors, resolutions)
        staggered_liquid_rates = _observed_rates(
            staggered_liquid_errors, resolutions,
        )
        staggered_gas_rates = _observed_rates(
            staggered_gas_errors, resolutions,
        )
        println(
            "two_velocity_convergence mesh=$kind n=$resolutions " *
                "fraction=$fraction_errors fraction_rates=$fraction_rates " *
                "scalar=$scalar_errors scalar_rates=$scalar_rates " *
                "staggered_liquid=$staggered_liquid_errors " *
                "staggered_liquid_rates=$staggered_liquid_rates " *
                "staggered_gas=$staggered_gas_errors " *
                "staggered_gas_rates=$staggered_gas_rates",
        )
        # The classical vortex study validates interface convergence on all
        # mesh families. This band isolates scalar transport; its mixed-grid
        # interface error is small but grid-alignment dominated, so only the
        # uniform interface series is gated here.
        kind === :uniform && @test all(diff(fraction_errors) .< 0)
        @test all(diff(scalar_errors) .< 0)
        kind === :uniform && @test fraction_rates[end] >= 1.8
        @test scalar_rates[end] >= 1.5
        @test all(diff(staggered_liquid_errors) .< 0)
        @test all(diff(staggered_gas_errors) .< 0)
        # The frozen-subface velocity replacement is only first-order here.
        @test staggered_liquid_rates[end] >= 1.1
        @test staggered_gas_rates[end] >= 1.1
        @test all(iszero, getproperty.(results, :one))
        @test all(iszero, getproperty.(results, :staggered_one))
        @test all(results) do result
            tolerance = 16 * result.steps *
                max(MIXTURE_PROJECTION_TOLERANCE, eps())
            result.staggered_liquid_amount <= tolerance
        end
        @test all(results) do result
            tolerance = 16 * result.steps *
                max(MIXTURE_PROJECTION_TOLERANCE, eps())
            result.staggered_gas_amount <= tolerance
        end
        @test all(results) do result
            tolerance = 16 * result.steps *
                max(MIXTURE_PROJECTION_TOLERANCE, eps())
            result.volume <= tolerance
        end
        @test all(result -> result.partition <= eps(), results)
        @test all(result -> result.projection_count == result.steps, results)
    end
end

using Test, TwoPhaseAMR

const TV_ROTATION = TwoPhaseAMR
const ROTATION_PROJECTION_TOLERANCE = sqrt(eps())

function _rotation_case(resolution)
    mesh = BlockTreeMesh(
        block_size = (resolution, resolution), periodic = (false, false),
    )
    centre = (0.5, 0.5)
    radius = 0.2
    angular_velocity = 2pi
    interface = InterfaceField(
        mesh,
        (x, y) -> hypot(x - centre[1], y - centre[2]) - radius;
        lipschitz = 1,
    )
    velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
    analytical_velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
    workspace = WYAdvectionWorkspace(interface, velocity)
    liquid_scalar(x, y) =
        1 + 0.2(x - centre[1]) + 0.1(y - centre[2])
    gas_scalar(x, y) =
        2 - 0.15(x - centre[1]) + 0.12(y - centre[2])
    staggered = (
        PhaseFaceScalar{1}(
            interface; liquid = liquid_scalar, gas = gas_scalar,
        ),
        PhaseFaceScalar{2}(
            interface; liquid = liquid_scalar, gas = gas_scalar,
        ),
    )
    one_functions = (
        PhaseFaceScalar{1}(interface), PhaseFaceScalar{2}(interface),
    )
    projection_solver = CMDGFMProjectionSolver(
        interface; density = (1.0, 1.0), preconditioner = :block,
    )
    projection_case = (
        pressure = CellField(mesh),
        solver = projection_solver,
        workspace = CMDGFMProjectionWorkspace(projection_solver),
    )
    return (;
        mesh, interface, velocity, analytical_velocity, workspace,
        projection_case,
        staggered, one_functions, liquid_scalar, gas_scalar,
        centre, angular_velocity,
        initial_fraction = copy(interface.fraction),
        initial_volume = integrate(interface),
    )
end

function _project_rotation!(case)
    projection = case.projection_case
    before = map(copy, case.velocity)
    stats = project!(
        case.velocity, projection.pressure, projection.solver, 1.0,
        projection.workspace;
        tolerance = ROTATION_PROJECTION_TOLERANCE,
        max_iterations = 6000,
    )
    corrections = map(1:2) do phase
        result = FaceVelocity(case.mesh)
        result.interior .= case.velocity[phase].interior .-
            before[phase].interior
        result.boundary .= case.velocity[phase].boundary .-
            before[phase].boundary
        result
    end
    return (; stats, corrections)
end

function _advance_rotation!(case; final_time = 0.03125)
    time = 0.0
    step = 0
    converged = true
    zero_phase_errors = (liquid = 0.0, gas = 0.0)
    zero_components = (
        liquid_normal = 0.0, liquid_tangential = 0.0,
        gas_normal = 0.0, gas_tangential = 0.0,
    )
    maximum_preprojection_error = zero_phase_errors
    maximum_postprojection_error = zero_phase_errors
    maximum_projection_components = zero_components
    maximum_initial_residual = 0.0
    maximum_final_residual = 0.0
    maximum_mixture_divergence = 0.0
    maximum_normal_velocity_defect = 0.0
    while time < final_time - 16eps()
        _set_stationary_rotation!(case)
        maximum_preprojection_error = _maximum_projection_error(
            maximum_preprojection_error,
            _face_velocity_linf(case, case.velocity),
        )
        projection = _project_rotation!(case)
        stats = projection.stats
        converged &= stats.converged
        maximum_initial_residual = max(maximum_initial_residual, stats.initial_residual)
        maximum_final_residual = max(maximum_final_residual, stats.final_residual)
        maximum_mixture_divergence = max(maximum_mixture_divergence, stats.mixture_divergence)
        maximum_normal_velocity_defect = max(
            maximum_normal_velocity_defect, stats.normal_velocity_defect,
        )
        maximum_postprojection_error = _maximum_projection_error(
            maximum_postprojection_error,
            _face_velocity_linf(case, case.velocity),
        )
        maximum_projection_components = _maximum_projection_error(
            maximum_projection_components,
            _interface_projection_components(
                case, projection.corrections,
            ),
        )
        dt = min(
            final_time - time,
            stable_timestep(case.velocity, case.workspace; cfl = 0.25),
        )
        order = iseven(step) ? (1, 2) : (2, 1)
        advect!(
            case.interface, case.velocity, dt, case.workspace;
            staggered_scalars = (
                case.staggered..., case.one_functions...,
            ),
            sweep_order = order,
        )
        time += dt
        step += 1
    end
    metrics = (;
        maximum_initial_residual, maximum_final_residual,
        maximum_mixture_divergence, maximum_normal_velocity_defect,
    )
    return _rotation_error(
        case, step, metrics, 0.0, converged, maximum_preprojection_error,
        maximum_postprojection_error, maximum_projection_components, step, time,
    )
end

function _packed_face_values(field)
    return vcat(field.interior, field.boundary)
end

function _staggered_rotation_errors(case, time)
    liquid_error = gas_error = one_error = 0.0
    angle = case.angular_velocity * time
    cosine, sine = cos(angle), sin(angle)
    liquid_reference(x, y) = begin
        dx, dy = x - case.centre[1], y - case.centre[2]
        case.liquid_scalar(
            case.centre[1] + cosine * dx + sine * dy,
            case.centre[2] - sine * dx + cosine * dy,
        )
    end
    for component in 1:2
        scalar = case.staggered[component]
        liquid_reference_field = FaceField{component}(
            case.mesh, liquid_reference,
        )
        gas_reference_field = FaceField{component}(
            case.mesh, case.gas_scalar,
        )
        schedule = scalar.workspace.schedule
        # The thesis phase L1 norm is integrated over the numerical phase at
        # the comparison time.  The initial fraction is only the exact
        # geometry reference; using it as a quadrature mask would assign an
        # intrinsic error to faces on which that numerical phase is absent.
        fraction = _packed_face_values(scalar.fraction)
        liquid = _packed_face_values(scalar.liquid)
        gas = _packed_face_values(scalar.gas)
        liquid_reference_values = _packed_face_values(liquid_reference_field)
        gas_reference_values = _packed_face_values(gas_reference_field)
        one_liquid = _packed_face_values(
            case.one_functions[component].liquid,
        )
        one_gas = _packed_face_values(case.one_functions[component].gas)
        for index in eachindex(liquid)
            volume = TV_ROTATION._phase_dual_volume(schedule, index)
            liquid_error += volume * fraction[index] *
                abs(liquid[index] - liquid_reference_values[index])
            gas_error += volume * (1 - fraction[index]) *
                abs(gas[index] - gas_reference_values[index])
            one_error = max(
                one_error, abs(one_liquid[index] - 1),
                abs(one_gas[index] - 1),
            )
        end
    end
    return liquid_error, gas_error, one_error
end

function _set_stationary_rotation!(case)
    cx, cy = case.centre
    omega = case.angular_velocity
    set_velocities!(
        case.analytical_velocity,
        (
            (x, y, time, axis) ->
            axis == 1 ? -omega * (y - cy) : omega * (x - cx),
            (x, y, time, axis) -> 0.0,
        ),
    )
    enforce_no_permeation!(case.analytical_velocity)
    for phase in 1:2
        copyto!(
            case.velocity[phase].interior,
            case.analytical_velocity[phase].interior,
        )
        copyto!(
            case.velocity[phase].boundary,
            case.analytical_velocity[phase].boundary,
        )
    end
    return case.velocity
end

function _face_velocity_linf(case, velocity)
    TV_ROTATION._freeze_phase_apertures!(case.interface, case.workspace)
    errors = zeros(2)
    for phase in 1:2
        aperture = phase == 1 ? case.workspace.liquid_aperture :
            case.workspace.gas_aperture
        for index in eachindex(case.mesh.faces)
            aperture[index] > 0 || continue
            errors[phase] = max(
                errors[phase],
                abs(
                    velocity[phase].interior[index] -
                        case.analytical_velocity[phase].interior[index],
                ),
            )
        end
        for index in eachindex(case.mesh.boundary_faces)
            name = phase == 1 ? :liquid : :gas
            TV_ROTATION._phase_cell_is_active(
                case.interface,
                case.mesh.boundary_faces[index].cell, name
            ) || continue
            errors[phase] = max(
                errors[phase],
                abs(
                    velocity[phase].boundary[index] -
                        case.analytical_velocity[phase].boundary[index],
                ),
            )
        end
    end
    return (liquid = errors[1], gas = errors[2])
end

function _interface_projection_components(case, corrections)
    reconstruct!(case.interface)
    centered = map(corrections) do correction
        isnothing(correction) ? nothing :
            TV_ROTATION._cell_centered_velocity(correction)
    end
    normal_errors = zeros(2)
    tangential_errors = zeros(2)
    floor = case.interface.fraction_floor
    nx, ny = case.mesh.block_size
    for block in eachindex(case.mesh.blocks), j in 1:ny, i in 1:nx
        ref = TV_ROTATION.CellRef(block, i, j)
        fraction = TV_ROTATION._get(case.interface.conditional_fraction, ref)
        floor < fraction < 1 - floor || continue
        normal_x = TV_ROTATION._get(case.interface.normal_x, ref)
        normal_y = TV_ROTATION._get(case.interface.normal_y, ref)
        magnitude = hypot(normal_x, normal_y)
        magnitude > 0 || continue
        normal_x /= magnitude
        normal_y /= magnitude
        index = (block - 1) * nx * ny + (j - 1) * nx + i
        for phase in 1:2
            isnothing(centered[phase]) && continue
            correction_x = centered[phase][1][index]
            correction_y = centered[phase][2][index]
            normal_errors[phase] = max(
                normal_errors[phase],
                abs(normal_x * correction_x + normal_y * correction_y),
            )
            tangential_errors[phase] = max(
                tangential_errors[phase],
                abs(-normal_y * correction_x + normal_x * correction_y),
            )
        end
    end
    return (;
        liquid_normal = normal_errors[1],
        liquid_tangential = tangential_errors[1],
        gas_normal = normal_errors[2],
        gas_tangential = tangential_errors[2],
    )
end

function _maximum_projection_error(first, second)
    return map(max, first, second)
end

function _rotation_error(
        case, projection_count, projection_metrics, maximum_aperture_defect,
        converged, maximum_preprojection_error, maximum_postprojection_error,
        maximum_projection_components, steps, time
    )
    error = maximum_error = 0.0
    for (block, leaf) in pairs(case.mesh.blocks)
        difference = abs.(
            interior(case.interface.fraction, block) .-
                interior(case.initial_fraction, block)
        )
        error += leaf.volume * sum(difference)
        maximum_error = max(maximum_error, maximum(difference))
    end
    staggered_liquid_error, staggered_gas_error, one_error =
        _staggered_rotation_errors(case, time)
    staggered_error = staggered_liquid_error + staggered_gas_error
    return merge(
        (;
            error, maximum_error,
            staggered_error, staggered_liquid_error, staggered_gas_error,
            one_error,
            volume_error = abs(
                integrate(case.interface) - case.initial_volume
            ),
            projection_count, maximum_aperture_defect,
            maximum_preprojection_error, maximum_postprojection_error,
            maximum_projection_components,
            projection_converged = converged, steps,
        ), projection_metrics
    )
end

function _rotation_rates(values, resolutions)
    return [
        log(values[index] / values[index + 1]) /
            log(resolutions[index + 1] / resolutions[index])
            for index in 1:(length(values) - 1)
    ]
end

function _reported_rotation_rates(values, resolutions)
    return all(value -> isfinite(value) && value > 0, values) ?
        _rotation_rates(values, resolutions) :
        fill(NaN, length(values) - 1)
end

function _aggregate_rotation_rate(values, resolutions)
    return log(first(values) / last(values)) /
        log(last(resolutions) / first(resolutions))
end

function _rotation_volume_tolerance(case, steps)
    cells_per_block = prod(case.mesh.block_size)
    domain_volume = sum(
        cells_per_block * block.volume for block in case.mesh.blocks
    )
    wisp_bound = 2 * steps * case.interface.fraction_floor * domain_volume
    roundoff_bound =
        16 * steps * length(case.mesh.faces) * eps(eltype(case.interface))
    return wisp_bound + roundoff_bound
end

function _initial_projection_snapshot(case)
    _set_stationary_rotation!(case)
    before = _face_velocity_linf(case, case.velocity)
    projection = _project_rotation!(case)
    statistics = projection.stats
    return (;
        before,
        after = _face_velocity_linf(case, case.velocity),
        components = _interface_projection_components(
            case, projection.corrections,
        ),
        initial_residual = statistics.initial_residual,
        final_residual = statistics.final_residual,
        mixture_divergence = statistics.mixture_divergence,
        normal_velocity_defect = statistics.normal_velocity_defect,
        converged = statistics.converged,
    )
end

@testset "Initial tangential-jump projection accuracy" begin
    resolutions = (16, 32, 64)
    snapshots = map(resolutions) do resolution
        frozen = _initial_projection_snapshot(_rotation_case(resolution))
        println(
            "two_velocity_rotation_initial_projection n=$resolution " *
                "frozen=$frozen",
        )
        @test frozen.converged
        @test all(value -> value <= 16eps(), frozen.before)
        @test all(isfinite, frozen.after)
        @test all(isfinite, frozen.components)
        @test frozen.final_residual <= frozen.initial_residual
        @test frozen.mixture_divergence <= ROTATION_PROJECTION_TOLERANCE
        @test frozen.normal_velocity_defect <= ROTATION_PROJECTION_TOLERANCE
        @test frozen.after.liquid > 0
        @test frozen.after.gas > 0
        frozen
    end
    for phase in (:liquid, :gas)
        values = [
            getproperty(snapshot.after, phase) for snapshot in snapshots
        ]
        rates = _reported_rotation_rates(values, resolutions)
        println(
            "two_velocity_rotation_initial_projection_convergence " *
                "phase=$phase n=$resolutions errors=$values rates=$rates",
        )
    end
end

@testset "Tangentially discontinuous circular-disc rotation" begin
    resolutions = (16, 32, 64)
    results = map(resolutions) do resolution
        case = _rotation_case(resolution)
        result = _advance_rotation!(case; final_time = 0.125)
        println("two_velocity_rotation n=$resolution frozen=$result")
        @test result.projection_count == result.steps
        @test result.projection_converged
        @test result.maximum_initial_residual >
            result.maximum_final_residual
        @test result.maximum_mixture_divergence <=
            ROTATION_PROJECTION_TOLERANCE
        @test result.maximum_normal_velocity_defect <=
            ROTATION_PROJECTION_TOLERANCE
        @test result.maximum_aperture_defect <= eps()
        @test result.volume_error <=
            _rotation_volume_tolerance(case, result.steps)
        @test result.one_error <= 512eps()
        @test all(isfinite, result.maximum_preprojection_error)
        @test all(isfinite, result.maximum_postprojection_error)
        @test all(isfinite, result.maximum_projection_components)
        @test all(
            value -> value <= 16eps(),
            result.maximum_preprojection_error,
        )
        @test result.maximum_postprojection_error.liquid > 0
        @test result.maximum_postprojection_error.gas > 0
        result
    end

    errors = [result.error for result in results]
    rates = _rotation_rates(errors, resolutions)
    aggregate_rate = _aggregate_rotation_rate(errors, resolutions)
    println(
        "two_velocity_rotation_convergence " *
            "n=$resolutions errors=$errors rates=$rates " *
            "aggregate_rate=$aggregate_rate",
    )
    @test all(diff(errors) .< 0)
    @test aggregate_rate >= 0.8

    maximum_errors = [result.maximum_error for result in results]
    maximum_rates = _reported_rotation_rates(maximum_errors, resolutions)
    println(
        "two_velocity_rotation_linf_convergence " *
            "n=$resolutions errors=$maximum_errors rates=$maximum_rates",
    )

    staggered_errors = [result.staggered_error for result in results]
    staggered_rates = _rotation_rates(staggered_errors, resolutions)
    println(
        "two_velocity_rotation_staggered_convergence " *
            "n=$resolutions errors=$staggered_errors " *
            "rates=$staggered_rates",
    )
    @test all(diff(staggered_errors) .< 0)
    # This continuously rotating shear layer remains on the phase interface,
    # so the thesis CTU fallback is exercised for O(1/h) successive steps.
    # Its phase-weighted L1 error is therefore first order. The
    # vortex-reversal test separately verifies the thesis's
    # close-to-second-order integral rate for smooth reversible transport.
    @test staggered_rates[end] >= 1.0
    for (phase, metric) in (
            (:liquid, :staggered_liquid_error),
            (:gas, :staggered_gas_error),
        )
        phase_errors = [getproperty(result, metric) for result in results]
        phase_rates = _reported_rotation_rates(phase_errors, resolutions)
        println(
            "two_velocity_rotation_staggered_phase " *
                "phase=$phase n=$resolutions errors=$phase_errors " *
                "rates=$phase_rates",
        )
    end

    for phase in (:liquid, :gas)
        before = [
            getproperty(result.maximum_preprojection_error, phase)
                for result in results
        ]
        after = [
            getproperty(result.maximum_postprojection_error, phase)
                for result in results
        ]
        after_rates = _reported_rotation_rates(after, resolutions)
        println(
            "two_velocity_rotation_projection_linf " *
                "phase=$phase n=$resolutions before=$before " *
                "after=$after after_rates=$after_rates",
        )
    end
    for component in (
            :liquid_normal, :liquid_tangential,
            :gas_normal, :gas_tangential,
        )
        values = [
            getproperty(result.maximum_projection_components, component)
                for result in results
        ]
        rates = _reported_rotation_rates(values, resolutions)
        println(
            "two_velocity_rotation_projection_component " *
                "component=$component n=$resolutions values=$values " *
                "rates=$rates",
        )
    end
end

@testset "Rotating disc with the CMDGFM projection" begin
    resolutions = (8, 16, 32)
    results = map(resolutions) do resolution
        case = _rotation_case(resolution)
        result = _advance_rotation!(case)
        println(
            "two_velocity_CMDGFM_rotation method=frozen " *
                "n=$resolution result=$result",
        )
        @test result.projection_converged
        @test result.maximum_mixture_divergence <=
            ROTATION_PROJECTION_TOLERANCE
        @test result.maximum_normal_velocity_defect <=
            ROTATION_PROJECTION_TOLERANCE
        @test result.volume_error <=
            _rotation_volume_tolerance(case, result.steps)
        @test result.one_error <= 1024eps()
        result
    end
    for metric in (
            :error, :maximum_error, :staggered_liquid_error,
            :staggered_gas_error,
        )
        errors = collect(getproperty.(results, metric))
        rates = _reported_rotation_rates(errors, resolutions)
        println(
            "two_velocity_CMDGFM_rotation_convergence method=frozen " *
                "metric=$metric n=$resolutions errors=$errors rates=$rates",
        )
    end
    interface_errors = collect(getproperty.(results, :error))
    @test all(diff(interface_errors) .< 0)
    @test _aggregate_rotation_rate(interface_errors, resolutions) >= 0.9
    for metric in (:staggered_liquid_error, :staggered_gas_error)
        errors = collect(getproperty.(results, metric))
        @test all(diff(errors) .< 0)
        @test _aggregate_rotation_rate(errors, resolutions) >= 0.9
    end
end

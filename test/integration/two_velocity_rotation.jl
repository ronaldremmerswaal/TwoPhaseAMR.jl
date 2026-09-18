using Test, TwoPhaseAMR

const TV = TwoPhaseAMR
const ROTATION_PROJECTION_TOLERANCE = 1.0e-10

function _rotation_case(resolution, method)
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
    workspace = WYAdvectionWorkspace(
        interface, velocity; interface_method = method,
    )
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
    solver = EllipticSolver(PoissonOperator(mesh); coarse_size = 8)
    mixture_projection = (
        mixture = FaceVelocity(mesh),
        unprojected = FaceVelocity(mesh),
        correction = FaceVelocity(mesh),
        pressure = CellField(mesh),
        solver,
        workspace = ProjectionWorkspace(solver),
        divergence = CellField(mesh),
    )
    mimetic_solver = MimeticProjectionSolver(
        interface; density = (1.0, 1.0), preconditioner = :cholesky,
    )
    mimetic_projection = (
        pressure = CellField(mesh),
        solver = mimetic_solver,
        workspace = MimeticProjectionWorkspace(mimetic_solver),
    )
    return (;
        mesh, interface, velocity, analytical_velocity, workspace,
        mixture_projection, mimetic_projection,
        staggered, one_functions, liquid_scalar, gas_scalar,
        centre, angular_velocity,
        initial_fraction = copy(interface.fraction),
        initial_volume = integrate(interface),
    )
end

function _project_rotation_mimetic!(case)
    projection = case.mimetic_projection
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

function _advance_mimetic_rotation!(case; final_time = 0.03125)
    time = 0.0
    step = 0
    converged = true
    zero_phase_errors = (liquid = 0.0, gas = 0.0)
    zero_components = (
        liquid_normal = 0.0, liquid_tangential = 0.0,
        gas_normal = 0.0, gas_tangential = 0.0,
    )
    maximum_postprojection_error = zero_phase_errors
    maximum_projection_components = zero_components
    maximum_raw_divergence = 0.0
    maximum_projected_divergence = 0.0
    while time < final_time - 16eps()
        _set_stationary_rotation!(case)
        projection = _project_rotation_mimetic!(case)
        stats = projection.stats
        converged &= stats.converged
        maximum_raw_divergence = max(
            maximum_raw_divergence, stats.initial_residual,
        )
        maximum_projected_divergence = max(
            maximum_projected_divergence,
            max(stats.mixture_divergence, stats.normal_jump_divergence),
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
    return _rotation_error(
        case, step, maximum_raw_divergence,
        maximum_projected_divergence, 0.0, converged,
        zero_phase_errors, maximum_postprojection_error,
        maximum_projection_components, step, time,
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
            volume = TV._phase_dual_volume(schedule, index)
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
    TV._freeze_phase_apertures!(case.interface, case.workspace)
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
            TV._cell_centered_velocity(correction)
    end
    normal_errors = zeros(2)
    tangential_errors = zeros(2)
    floor = case.interface.fraction_floor
    nx, ny = case.mesh.block_size
    for block in eachindex(case.mesh.blocks), j in 1:ny, i in 1:nx
        ref = TV.CellRef(block, i, j)
        fraction = TV._get(case.interface.conditional_fraction, ref)
        floor < fraction < 1 - floor || continue
        normal_x = TV._get(case.interface.normal_x, ref)
        normal_y = TV._get(case.interface.normal_y, ref)
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

function _divergence_linf!(destination, velocity)
    divergence!(destination, velocity)
    maximum_value = 0.0
    for block in eachindex(velocity.mesh.blocks)
        maximum_value = max(
            maximum_value, maximum(abs, interior(destination, block)),
        )
    end
    return maximum_value
end

"""Project the current aperture-weighted mixture flux for the passive test.

This is deliberately test-side preparation rather than the physical
two-velocity pressure projection. Applying the same correction to both phase
velocities preserves their jump, while aperture partitioning makes the
corrected mixture flux equal to the ordinary projected face field.
"""
function _project_rotation_mixture!(case)
    interface = case.interface
    work = case.workspace
    projection = case.mixture_projection
    reconstruct!(interface)
    TV._freeze_phase_apertures!(interface, work)
    aperture_defect = 0.0
    for index in eachindex(case.mesh.faces)
        liquid = work.liquid_aperture[index]
        gas = work.gas_aperture[index]
        aperture_defect = max(aperture_defect, abs(liquid + gas - 1))
        projection.mixture.interior[index] =
            liquid * case.velocity[1].interior[index] +
            gas * case.velocity[2].interior[index]
    end
    fill!(projection.mixture.boundary, 0)
    copyto!(projection.unprojected.interior, projection.mixture.interior)
    copyto!(projection.unprojected.boundary, projection.mixture.boundary)
    initial_divergence = _divergence_linf!(
        projection.divergence, projection.mixture,
    )
    statistics = project!(
        projection.mixture, projection.pressure, projection.solver, 1.0,
        projection.workspace;
        tolerance = ROTATION_PROJECTION_TOLERANCE, max_cycles = 200,
    )
    for index in eachindex(case.mesh.faces)
        projection.correction.interior[index] =
            projection.mixture.interior[index] -
            projection.unprojected.interior[index]
    end
    fill!(projection.correction.boundary, 0)
    for phase in 1:2, index in eachindex(case.mesh.faces)
        case.velocity[phase].interior[index] +=
            projection.correction.interior[index]
    end
    final_divergence = _divergence_linf!(
        projection.divergence, projection.mixture,
    )
    return (;
        statistics, initial_divergence, final_divergence,
        aperture_defect, correction = projection.correction,
    )
end

function _rotation_error(
        case, projection_count, maximum_raw_divergence,
        maximum_projected_divergence, maximum_aperture_defect, converged,
        maximum_preprojection_error, maximum_postprojection_error,
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
    return (;
        error, maximum_error,
        staggered_error, staggered_liquid_error, staggered_gas_error,
        one_error,
        volume_error = abs(
            integrate(case.interface) - case.initial_volume
        ),
        projection_count, maximum_raw_divergence,
        maximum_projected_divergence, maximum_aperture_defect,
        maximum_preprojection_error, maximum_postprojection_error,
        maximum_projection_components,
        projection_converged = converged, steps,
    )
end

function _advance_rotation_pair!(projected, frozen; final_time = 0.125)
    time = 0.0
    step = 0
    projection_count = 0
    maximum_raw_divergence = 0.0
    maximum_projected_divergence = 0.0
    maximum_aperture_defect = 0.0
    zero_phase_errors = (liquid = 0.0, gas = 0.0)
    zero_components = (
        liquid_normal = 0.0, liquid_tangential = 0.0,
        gas_normal = 0.0, gas_tangential = 0.0,
    )
    projected_preprojection_error = zero_phase_errors
    projected_postprojection_error = zero_phase_errors
    projected_projection_components = zero_components
    frozen_preprojection_error = zero_phase_errors
    frozen_postprojection_error = zero_phase_errors
    frozen_projection_components = zero_components
    projected_projection_count = 0
    projected_maximum_raw_divergence = 0.0
    projected_maximum_final_divergence = 0.0
    projected_projection_converged = true
    projection_converged = true
    while time < final_time - 16eps()
        _set_stationary_rotation!(projected)
        _set_stationary_rotation!(frozen)
        projected_preprojection_error = _maximum_projection_error(
            projected_preprojection_error,
            _face_velocity_linf(projected, projected.velocity),
        )
        frozen_preprojection_error = _maximum_projection_error(
            frozen_preprojection_error,
            _face_velocity_linf(frozen, frozen.velocity),
        )
        projection = _project_rotation_mixture!(frozen)
        projection_count += 1
        maximum_raw_divergence = max(
            maximum_raw_divergence, projection.initial_divergence,
        )
        maximum_projected_divergence = max(
            maximum_projected_divergence, projection.final_divergence,
        )
        maximum_aperture_defect = max(
            maximum_aperture_defect, projection.aperture_defect,
        )
        projection_converged &= projection.statistics.converged

        frozen_postprojection_error = _maximum_projection_error(
            frozen_postprojection_error,
            _face_velocity_linf(frozen, frozen.velocity),
        )
        frozen_projection_components = _maximum_projection_error(
            frozen_projection_components,
            _interface_projection_components(
                frozen, (projection.correction, projection.correction),
            ),
        )

        # Project exactly once from the freshly initialized analytical field.
        # The returned carrier is separate storage; neither analytical phase
        # field is overwritten by the projected-liquid policy.
        TV._reset!(projected.workspace.diagnostics)
        projected_liquid = TV._project_liquid_velocity!(
            projected.workspace, projected.velocity,
        )
        projected_projection_count += 1
        projected_diagnostics = phase_flux_diagnostics(projected.workspace)
        projected_maximum_raw_divergence = max(
            projected_maximum_raw_divergence,
            projected_diagnostics.projection_initial_divergence,
        )
        projected_maximum_final_divergence = max(
            projected_maximum_final_divergence,
            projected_diagnostics.projection_final_divergence,
        )
        projected_state = projected.workspace.method_workspace[]
        projected_advecting_velocity =
            (projected_liquid, projected.velocity[2])
        projected_postprojection_error = _maximum_projection_error(
            projected_postprojection_error,
            _face_velocity_linf(
                projected,
                projected_advecting_velocity,
            ),
        )
        projected_limit = min(
            stable_timestep(
                projected_liquid, projected.workspace; cfl = 0.25,
            ),
            stable_timestep(
                projected.velocity[2], projected.workspace; cfl = 0.25,
            ),
        )
        projected_projection_components = _maximum_projection_error(
            projected_projection_components,
            _interface_projection_components(
                projected, (projected_state.correction, nothing),
            ),
        )
        dt = min(
            final_time - time, projected_limit,
            stable_timestep(
                frozen.velocity, frozen.workspace; cfl = 0.25,
            ),
        )
        order = iseven(step) ? (1, 2) : (2, 1)
        TV._advect_wy!(
            projected.interface, projected_advecting_velocity, dt,
            projected.workspace;
            staggered_scalars = (
                projected.staggered..., projected.one_functions...,
            ),
            sweep_order = order, reset_diagnostics = false,
            check_timestep = false,
        )
        advect!(
            frozen.interface, frozen.velocity, dt,
            frozen.workspace;
            staggered_scalars = (
                frozen.staggered..., frozen.one_functions...,
            ),
            sweep_order = order,
        )
        time += dt
        step += 1
    end
    return (
        projected = _rotation_error(
            projected, projected_projection_count,
            projected_maximum_raw_divergence,
            projected_maximum_final_divergence, 0.0,
            projected_projection_converged,
            projected_preprojection_error,
            projected_postprojection_error,
            projected_projection_components, step, time,
        ),
        frozen = _rotation_error(
            frozen, projection_count, maximum_raw_divergence,
            maximum_projected_divergence, maximum_aperture_defect,
            projection_converged, frozen_preprojection_error,
            frozen_postprojection_error, frozen_projection_components,
            step, time,
        ),
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
    if case.workspace.interface_method isa ProjectedLiquidVelocity
        projected = TV._project_liquid_velocity!(
            case.workspace, case.velocity,
        )
        state = case.workspace.method_workspace[]
        diagnostics = phase_flux_diagnostics(case.workspace)
        return (;
            before,
            after = _face_velocity_linf(
                case, (projected, case.velocity[2]),
            ),
            components = _interface_projection_components(
                case, (state.correction, nothing),
            ),
            initial_divergence = diagnostics.projection_initial_divergence,
            final_divergence = diagnostics.projection_final_divergence,
            converged = true,
        )
    end
    projection = _project_rotation_mixture!(case)
    return (;
        before,
        after = _face_velocity_linf(case, case.velocity),
        components = _interface_projection_components(
            case, (projection.correction, projection.correction),
        ),
        initial_divergence = projection.initial_divergence,
        final_divergence = projection.final_divergence,
        converged = projection.statistics.converged,
    )
end

@testset "Initial tangential-jump projection accuracy" begin
    resolutions = (16, 32, 64)
    snapshots = map(resolutions) do resolution
        projected = _initial_projection_snapshot(
            _rotation_case(
                resolution,
                ProjectedLiquidVelocity(
                    tolerance = 1.0e-12, max_cycles = 500,
                ),
            ),
        )
        frozen = _initial_projection_snapshot(
            _rotation_case(resolution, FrozenInterfaceFluxes()),
        )
        println(
            "two_velocity_rotation_initial_projection n=$resolution " *
                "projected=$projected frozen=$frozen",
        )
        for snapshot in (projected, frozen)
            @test snapshot.converged
            @test all(value -> value <= 16eps(), snapshot.before)
            @test all(isfinite, snapshot.after)
            @test all(isfinite, snapshot.components)
            @test snapshot.final_divergence <=
                ROTATION_PROJECTION_TOLERANCE
        end
        @test projected.after.liquid == 0
        @test projected.components.liquid_normal == 0
        @test projected.components.liquid_tangential == 0
        @test frozen.after.liquid > 0
        @test frozen.after.gas > 0
        (; projected, frozen)
    end
    for method in (:projected, :frozen), phase in (:liquid, :gas)
        values = [
            getproperty(getproperty(snapshot, method).after, phase)
                for snapshot in snapshots
        ]
        rates = _reported_rotation_rates(values, resolutions)
        println(
            "two_velocity_rotation_initial_projection_convergence " *
                "method=$method phase=$phase n=$resolutions " *
                "errors=$values rates=$rates",
        )
    end
end

@testset "Tangentially discontinuous circular-disc rotation" begin
    resolutions = (16, 32, 64)
    results = map(resolutions) do resolution
        projected = _rotation_case(
            resolution,
            ProjectedLiquidVelocity(tolerance = 1.0e-12, max_cycles = 500),
        )
        frozen = _rotation_case(resolution, FrozenInterfaceFluxes())
        result = _advance_rotation_pair!(projected, frozen)
        println(
            "two_velocity_rotation n=$resolution " *
                "projected=$(result.projected) frozen=$(result.frozen)",
        )
        @test result.projected.steps == result.frozen.steps
        @test result.projected.projection_count == result.projected.steps
        @test result.frozen.projection_count == result.frozen.steps
        @test result.frozen.projection_converged
        @test result.frozen.maximum_raw_divergence >
            result.frozen.maximum_projected_divergence
        @test result.frozen.maximum_projected_divergence <=
            ROTATION_PROJECTION_TOLERANCE
        @test result.frozen.maximum_aperture_defect <= eps()
        for (case, method_result) in zip((projected, frozen), result)
            @test method_result.volume_error <=
                _rotation_volume_tolerance(case, method_result.steps)
            @test method_result.one_error <= 512eps()
            @test all(isfinite, method_result.maximum_preprojection_error)
            @test all(isfinite, method_result.maximum_postprojection_error)
            @test all(isfinite, method_result.maximum_projection_components)
            @test all(
                value -> value <= 16eps(),
                method_result.maximum_preprojection_error,
            )
        end
        @test result.projected.projection_converged
        @test result.projected.maximum_raw_divergence == 0
        @test result.projected.maximum_projected_divergence == 0
        @test result.projected.maximum_projected_divergence <=
            ROTATION_PROJECTION_TOLERANCE
        @test result.projected.maximum_postprojection_error.liquid == 0
        @test result.projected.maximum_postprojection_error.gas == 0
        @test result.projected.maximum_projection_components.liquid_normal == 0
        @test result.projected.maximum_projection_components.liquid_tangential == 0
        @test result.frozen.maximum_postprojection_error.liquid > 0
        @test result.frozen.maximum_postprojection_error.gas > 0
        result
    end

    for method in (:projected, :frozen)
        errors = [getproperty(result, method).error for result in results]
        rates = _rotation_rates(errors, resolutions)
        aggregate_rate = _aggregate_rotation_rate(errors, resolutions)
        minimum_aggregate_rate = method === :projected ? 2.0 : 0.8
        println(
            "two_velocity_rotation_convergence method=$method " *
                "n=$resolutions errors=$errors rates=$rates " *
                "aggregate_rate=$aggregate_rate",
        )
        @test all(diff(errors) .< 0)
        @test aggregate_rate >= minimum_aggregate_rate

        maximum_errors = [
            getproperty(result, method).maximum_error for result in results
        ]
        maximum_rates = _reported_rotation_rates(
            maximum_errors, resolutions,
        )
        println(
            "two_velocity_rotation_linf_convergence method=$method " *
                "n=$resolutions errors=$maximum_errors " *
                "rates=$maximum_rates",
        )

        staggered_errors = [
            getproperty(result, method).staggered_error
                for result in results
        ]
        staggered_rates = _rotation_rates(
            staggered_errors, resolutions,
        )
        println(
            "two_velocity_rotation_staggered_convergence method=$method " *
                "n=$resolutions errors=$staggered_errors " *
                "rates=$staggered_rates",
        )
        @test all(diff(staggered_errors) .< 0)
        # This continuously rotating shear layer remains on the phase
        # interface, so the thesis CTU fallback is exercised for O(1/h)
        # successive steps. Its phase-weighted L1 error is therefore first
        # order. The vortex-reversal test separately verifies the thesis's
        # close-to-second-order integral rate for smooth reversible transport.
        minimum_staggered_rate = method === :projected ? 0.9 : 1.0
        @test staggered_rates[end] >= minimum_staggered_rate
        for (phase, metric) in (
                (:liquid, :staggered_liquid_error),
                (:gas, :staggered_gas_error),
            )
            phase_errors = [
                getproperty(getproperty(result, method), metric)
                    for result in results
            ]
            phase_rates = _reported_rotation_rates(
                phase_errors, resolutions,
            )
            println(
                "two_velocity_rotation_staggered_phase method=$method " *
                    "phase=$phase n=$resolutions errors=$phase_errors " *
                    "rates=$phase_rates",
            )
        end
        if method === :projected
            @test all(
                iszero,
                getproperty.(
                    getproperty.(results, method), :staggered_gas_error,
                ),
            )
        end

        for phase in (:liquid, :gas)
            before = [
                getproperty(
                    getproperty(result, method).maximum_preprojection_error,
                    phase,
                ) for result in results
            ]
            after = [
                getproperty(
                    getproperty(result, method).maximum_postprojection_error,
                    phase,
                ) for result in results
            ]
            after_rates = _reported_rotation_rates(after, resolutions)
            println(
                "two_velocity_rotation_projection_linf method=$method " *
                    "phase=$phase n=$resolutions before=$before " *
                    "after=$after after_rates=$after_rates",
            )
        end
        for component in (
                :liquid_normal, :liquid_tangential,
                :gas_normal, :gas_tangential,
            )
            values = [
                getproperty(
                    getproperty(result, method).maximum_projection_components,
                    component,
                ) for result in results
            ]
            rates = _reported_rotation_rates(values, resolutions)
            println(
                "two_velocity_rotation_projection_component method=$method " *
                    "component=$component n=$resolutions values=$values " *
                    "rates=$rates",
            )
        end
    end
end

@testset "Rotating disc with the physical mimetic projection" begin
    resolutions = (8, 16, 32)
    for (method_name, method) in pairs((;
            projected = ProjectedLiquidVelocity(
                tolerance = 1.0e-12, max_cycles = 500,
            ),
            frozen = FrozenInterfaceFluxes(),
        ))
        results = map(resolutions) do resolution
            case = _rotation_case(resolution, method)
            result = _advance_mimetic_rotation!(case)
            println(
                "two_velocity_mimetic_rotation method=$method_name " *
                    "n=$resolution result=$result",
            )
            @test result.projection_converged
            @test result.maximum_projected_divergence <= 1.0e-8
            @test result.volume_error <=
                _rotation_volume_tolerance(case, result.steps)
            @test result.one_error <= 1024eps()
            result
        end
        for metric in (
                :error, :maximum_error, :staggered_liquid_error,
                :staggered_gas_error,
            )
            errors = [getproperty(result, metric) for result in results]
            rates = _reported_rotation_rates(errors, resolutions)
            println(
                "two_velocity_mimetic_rotation_convergence " *
                    "method=$method_name metric=$metric n=$resolutions " *
                    "errors=$errors rates=$rates",
            )
        end
        interface_errors = getproperty.(results, :error)
        @test all(diff(interface_errors) .< 0)
        interface_rate = _aggregate_rotation_rate(
            interface_errors, resolutions,
        )
        if method_name === :projected
            @test interface_rate >= 1.2
        end
        for metric in (:staggered_liquid_error, :staggered_gas_error)
            errors = getproperty.(results, metric)
            @test all(diff(errors) .< 0)
            @test _aggregate_rotation_rate(errors, resolutions) >= 0.9
        end
    end
end

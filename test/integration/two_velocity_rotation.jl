using Test, TwoPhaseAMR

const TV = TwoPhaseAMR
const ROTATION_PROJECTION_TOLERANCE = 1.0e-12

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
    workspace = WYAdvectionWorkspace(
        interface, velocity; interface_method = method,
    )
    solver = EllipticSolver(PoissonOperator(mesh); coarse_size = 8)
    mixture_projection = (
        mixture = FaceVelocity(mesh),
        unprojected = FaceVelocity(mesh),
        pressure = CellField(mesh),
        solver,
        workspace = ProjectionWorkspace(solver),
        divergence = CellField(mesh),
    )
    return (;
        mesh, interface, velocity, workspace, mixture_projection,
        centre, angular_velocity,
        initial_fraction = copy(interface.fraction),
        initial_volume = integrate(interface),
    )
end

function _set_stationary_rotation!(case)
    cx, cy = case.centre
    omega = case.angular_velocity
    set_velocities!(
        case.velocity,
        (
            (x, y, time, axis) ->
                axis == 1 ? -omega * (y - cy) : omega * (x - cx),
            (x, y, time, axis) -> 0.0,
        ),
    )
    enforce_no_permeation!(case.velocity)
    return case.velocity
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
    for phase in 1:2, index in eachindex(case.mesh.faces)
        case.velocity[phase].interior[index] +=
            projection.mixture.interior[index] -
            projection.unprojected.interior[index]
    end
    final_divergence = _divergence_linf!(
        projection.divergence, projection.mixture,
    )
    return (; statistics, initial_divergence, final_divergence,
        aperture_defect)
end

function _rotation_error(case, projection_count, maximum_raw_divergence,
        maximum_projected_divergence, maximum_aperture_defect, converged,
        steps)
    error = maximum_error = 0.0
    for (block, leaf) in pairs(case.mesh.blocks)
        difference = abs.(
            interior(case.interface.fraction, block) .-
                interior(case.initial_fraction, block)
        )
        error += leaf.volume * sum(difference)
        maximum_error = max(maximum_error, maximum(difference))
    end
    return (;
        error, maximum_error,
        volume_error = abs(
            integrate(case.interface) - case.initial_volume
        ),
        projection_count, maximum_raw_divergence,
        maximum_projected_divergence, maximum_aperture_defect,
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
    projection_converged = true
    while time < final_time - 16eps()
        _set_stationary_rotation!(projected)
        _set_stationary_rotation!(frozen)
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

        dt = min(
            final_time - time,
            stable_timestep(
                projected.velocity, projected.workspace; cfl = 0.25,
            ),
            stable_timestep(
                frozen.velocity, frozen.workspace; cfl = 0.25,
            ),
        )
        order = iseven(step) ? (1, 2) : (2, 1)
        advect!(
            projected.interface, projected.velocity, dt,
            projected.workspace; sweep_order = order,
        )
        advect!(
            frozen.interface, frozen.velocity, dt,
            frozen.workspace; sweep_order = order,
        )
        time += dt
        step += 1
    end
    return (
        projected = _rotation_error(
            projected, 0, 0.0, 0.0, 0.0, true, step,
        ),
        frozen = _rotation_error(
            frozen, projection_count, maximum_raw_divergence,
            maximum_projected_divergence, maximum_aperture_defect,
            projection_converged, step,
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

@testset "Tangentially discontinuous circular-disc rotation" begin
    resolutions = (16, 24, 32)
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
        @test result.frozen.projection_count == result.frozen.steps
        @test result.frozen.projection_converged
        @test result.frozen.maximum_raw_divergence >
            result.frozen.maximum_projected_divergence
        @test result.frozen.maximum_projected_divergence <=
            ROTATION_PROJECTION_TOLERANCE
        @test result.frozen.maximum_aperture_defect <= eps()
        for method_result in result
            @test method_result.volume_error <= resolution^2 * eps()
        end
        result
    end

    for method in (:projected, :frozen)
        errors = [getproperty(result, method).error for result in results]
        rates = _rotation_rates(errors, resolutions)
        minimum_finest_rate = method === :projected ? 1.6 : 2.5
        println(
            "two_velocity_rotation_convergence method=$method " *
                "n=$resolutions errors=$errors rates=$rates",
        )
        @test all(diff(errors) .< 0)
        @test rates[end] >= minimum_finest_rate
    end
end

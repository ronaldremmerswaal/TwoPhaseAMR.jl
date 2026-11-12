using Test, TwoPhaseAMR
using StaticArrays: SVector
include("embedded_operator_reference_helpers.jl")
isdefined(@__MODULE__, :test_allocated_call) ||
    include(joinpath(@__DIR__, "test_utils.jl"))

function _warmed_diagnostic_allocations(diagnostic, state)
    diagnostic_value(diagnostic, state)
    return @allocated diagnostic_value(diagnostic, state)
end

# Independent cell-integrated face balance, using the projection's physical
# phase apertures but not the diagnostic's face or divergence scratch.
function _reference_two_velocity_divergence(interface, velocity)
    mesh = interface.mesh
    geometry = interface.geometry
    nx, ny = mesh.block_size
    liquid_aperture, gas_aperture =
        TwoPhaseAMR._current_phase_face_apertures!(interface)
    cell_flux = zeros(eltype(interface), ncells(mesh))
    for (index, face) in pairs(mesh.faces)
        liquid = iszero(liquid_aperture[index]) ? 0.0 :
            liquid_aperture[index] * velocity[1].interior[index]
        gas = iszero(gas_aperture[index]) ? 0.0 :
            gas_aperture[index] * velocity[2].interior[index]
        flux = face.area * (liquid + gas)
        cell_flux[TwoPhaseAMR._cell_index(face.minus, nx, ny)] += flux
        cell_flux[TwoPhaseAMR._cell_index(face.plus, nx, ny)] -= flux
    end
    for (index, face) in pairs(mesh.boundary_faces)
        liquid_aperture, gas_aperture =
            TwoPhaseAMR._phase_boundary_face_apertures(interface, index)
        liquid = iszero(liquid_aperture) ? 0.0 :
            liquid_aperture * velocity[1].boundary[index]
        gas = iszero(gas_aperture) ? 0.0 :
            gas_aperture * velocity[2].boundary[index]
        cell_flux[TwoPhaseAMR._cell_index(face.cell, nx, ny)] +=
            face.side * face.area * (liquid + gas)
    end
    norm_squared = zero(eltype(interface))
    for (block, leaf) in pairs(mesh.blocks), j in 1:ny, i in 1:nx
        ref = CellRef(block, i, j)
        open = isnothing(geometry) ? 1.0 :
            TwoPhaseAMR._get(geometry.open_fraction, ref)
        open > 0 || continue
        flux = cell_flux[TwoPhaseAMR._cell_index(ref, nx, ny)]
        norm_squared += abs2(flux) / (leaf.volume * open)
    end
    return sqrt(norm_squared)
end

# The former diagnostic used this mass-weighted mixture and ignored boundary
# fluxes. It is deliberately different from the projection's volume flux.
function _reference_mass_weighted_divergence(flow)
    interface = flow.interface
    mesh = interface.mesh
    geometry = interface.geometry
    liquid_density, gas_density = flow.system.two_phase.density
    mixture = FaceVelocity(mesh)
    for (index, face) in pairs(mesh.faces)
        !isnothing(geometry) && iszero(geometry.face_dual_volumes[index]) && continue
        fraction = TwoPhaseAMR._staggered_liquid_fraction(interface, geometry, face)
        liquid_mass = liquid_density * fraction
        gas_mass = gas_density * (1 - fraction)
        liquid = iszero(liquid_mass) ? 0.0 :
            liquid_mass * flow.velocity[1].interior[index]
        gas = iszero(gas_mass) ? 0.0 :
            gas_mass * flow.velocity[2].interior[index]
        mixture.interior[index] = (liquid + gas) / (liquid_mass + gas_mass)
    end
    divergence = CellField(mesh)
    isnothing(geometry) ? divergence!(divergence, mixture) :
        divergence!(divergence, mixture, geometry)
    return sqrt(
        sum(eachindex(mesh.blocks)) do block
            values = interior(divergence, block)
            open = isnothing(geometry) ? ones(size(values)) :
                interior(geometry.open_fraction, block)
            sum(abs2.(values) .* open) * mesh.blocks[block].volume
        end
    )
end

@testset "equation-system-independent simulation state" begin
    state = SimulationState(nothing, nothing, (;), 0.0, 0)
    @test isnothing(state.interface)
    @test isnothing(state.geometry)
    @test isnothing(state.workspace)
    @test_throws ArgumentError Diagnostic(:name, :measure)
end

@testset "simulation-owned diagnostics" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    physics = SinglePhase(density = 2.0, dynamic_viscosity = 0.1)
    evaluations = Ref(0)
    flow = FlowSimulation(
        mesh, physics;
        velocity = (2.0, 3.0), dt = 1.0e-3,
        diagnostics = (
            KineticEnergy(), PotentialEnergy(), SurfaceEnergy(), TotalEnergy(),
            LinearMomentum(), DivergenceNorm(),
            MaximumVelocity(), CFLTimeStep(), LeafBlockCount(), CellCount(),
            Diagnostic(:reported_step, state -> state.step) => EveryNSteps(2),
            Diagnostic(:complex_value, state -> complex(state.time, state.step)),
            Diagnostic(
                :vector_value,
                state -> SVector(state.time, typeof(state.time)(state.step)),
            ),
            Diagnostic(
                :evaluation_count, state -> begin
                    evaluations[] += 1
                    state.time
                end
            ),
        ),
    )

    @test diagnostic_names(flow) == [
        :kinetic_energy, :potential_energy, :surface_energy, :total_energy,
        :linear_momentum, :divergence_norm,
        :maximum_velocity, :cfl_timestep, :leaf_blocks, :cells,
        :reported_step, :complex_value, :vector_value, :evaluation_count,
    ]
    @test only(diagnostic_history(flow, :kinetic_energy).values) ≈ 13.0
    @test only(diagnostic_history(flow, :potential_energy).values) ≈ 0.0
    @test only(diagnostic_history(flow, :surface_energy).values) ≈ 0.0
    @test only(diagnostic_history(flow, :total_energy).values) ≈ 13.0
    @test only(diagnostic_history(flow, :linear_momentum).values) ≈ [4.0, 6.0]
    @test only(diagnostic_history(flow, :divergence_norm).values) ≈ 0.0
    @test only(diagnostic_history(flow, :maximum_velocity).values) ≈ 3.0
    @test only(diagnostic_history(flow, :cfl_timestep).values) ≈
        stable_timestep(flow)
    @test only(diagnostic_history(flow, :leaf_blocks).values) == nblocks(mesh)
    @test only(diagnostic_history(flow, :cells).values) == ncells(mesh)
    @test eltype(diagnostic_history(flow, :complex_value).values) ==
        ComplexF64
    @test eltype(diagnostic_history(flow, :vector_value).values) ==
        SVector{2, Float64}
    @test simulation_state(flow).step == 0

    fields = VisualizationFieldRegistry()
    register_field!(fields, :pressure, flow.pressure)
    recorder = Recorder(
        fields; diagnostics = flow, keyframes = nothing, preview = nothing,
    )
    result = simulate!(flow, 3.0e-3; observer = recorder)
    @test result.total_steps == 3
    @test evaluations[] == 4
    @test diagnostic_history(flow, :reported_step).steps == [0, 2, 3]
    @test diagnostic_history(flow, :kinetic_energy).steps == [0, 1, 2, 3]
    # The recorder mirrors simulation-owned samples without re-evaluating them.
    store = snapshot_store(recorder)
    @test all(≈(13.0), diagnostic_history(store, :kinetic_energy).values)
    @test all(≈(4.0), diagnostic_history(store, :linear_momentum).values[:, 1])
    @test diagnostic_history(store, :evaluation_count).steps == [0, 1, 2, 3]

    report = IOBuffer()
    @test isnothing(print_simulation_summary(report, flow, result))

    add_diagnostic!(flow, :time, state -> state.time)
    @test only(diagnostic_history(flow, :time).values) == flow.time
    @test_throws ArgumentError add_diagnostic!(flow, KineticEnergy())
end

@testset "one- and two-velocity physical diagnostics" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), periodic = (false, false),
    )
    interface = InterfaceField(mesh, (x, y) -> x - 0.5)
    physics = TwoPhase(
        interface; density = (4.0, 1.0),
        dynamic_viscosity = (0.1, 0.1),
        surface_tension = 0.25, gravity = (-2.0, 0.0),
    )
    shared = (x, y, time, axis) -> axis == 1 ? 2.0 : -1.0
    requested = (
        KineticEnergy(), PotentialEnergy(), SurfaceEnergy(), TotalEnergy(),
        LinearMomentum(), PhaseVolume(), PhaseMass(), InterfaceMeasure(),
        MaximumVelocity(),
    )
    one = FlowSimulation(
        mesh, physics; velocity_model = OneVelocityModel(),
        velocity = shared, diagnostics = requested,
    )
    two = FlowSimulation(
        mesh, physics; velocity_model = TwoVelocityModel(),
        velocity = shared, diagnostics = requested,
    )
    for flow in (one, two)
        state = simulation_state(flow)
        for diagnostic in (
                KineticEnergy(), LinearMomentum(), PhaseVolume(),
                DivergenceNorm(), LeafBlockCount(),
            )
            @test _warmed_diagnostic_allocations(diagnostic, state) == 0
        end
    end

    @test only(diagnostic_history(one, :kinetic_energy).values) ≈
        only(diagnostic_history(two, :kinetic_energy).values)
    @test only(diagnostic_history(one, :linear_momentum).values) ≈
        only(diagnostic_history(two, :linear_momentum).values)
    @test only(diagnostic_history(one, :potential_energy).values) ≈ 1.75
    @test only(diagnostic_history(two, :potential_energy).values) ≈ 1.75
    @test only(diagnostic_history(one, :surface_energy).values) ≈ 0.25
    @test only(diagnostic_history(two, :surface_energy).values) ≈ 0.25
    @test only(diagnostic_history(one, :total_energy).values) ≈
        only(diagnostic_history(one, :kinetic_energy).values) + 2.0
    @test only(diagnostic_history(two, :total_energy).values) ≈
        only(diagnostic_history(two, :kinetic_energy).values) + 2.0
    volumes = only(diagnostic_history(two, :phase_volume).values)
    @test volumes[1] ≈ 0.5
    @test volumes[2] ≈ 0.5
    @test only(diagnostic_history(two, :phase_mass).values) ≈ [2.0, 0.5]
    @test only(diagnostic_history(two, :interface_measure).values) ≈ 1.0
    @test only(diagnostic_history(two, :maximum_velocity).values) ≈ 2.0

    distinct = FlowSimulation(
        mesh, physics; velocity_model = TwoVelocityModel(),
        velocity = (
            (x, y, time, axis) -> axis == 1 ? 2.0 : -1.0,
            (x, y, time, axis) -> axis == 1 ? -1.0 : 3.0,
        ),
        diagnostics = (KineticEnergy(), LinearMomentum(), MaximumVelocity()),
    )
    # The discrete momentum inner product owns interior staggered unknowns;
    # prescribed physical-boundary faces are not degrees of freedom. On this
    # four-cell nonperiodic grid their dual volumes sum to 3/4 per component.
    @test only(diagnostic_history(distinct, :kinetic_energy).values) ≈ 5.625
    @test only(diagnostic_history(distinct, :linear_momentum).values) ≈
        [2.625, -0.375]
    @test only(diagnostic_history(distinct, :maximum_velocity).values) ≈ 3.0

    mixed_interface = InterfaceField(mesh, (x, y) -> x - 0.3)
    mixed_flow = FlowSimulation(
        mesh,
        TwoPhase(
            mixed_interface; density = (4.0, 1.0),
            dynamic_viscosity = (0.1, 0.1),
            surface_tension = 0.25, gravity = (-2.0, 0.0),
        );
        diagnostics = (PotentialEnergy(), SurfaceEnergy(), TotalEnergy()),
    )
    # The gravitational diagnostic integrates the first moment of each PLIC
    # phase polygon, rather than placing a mixed cell's mass at its centre.
    @test only(diagnostic_history(mixed_flow, :potential_energy).values) ≈ 1.27
    @test only(diagnostic_history(mixed_flow, :surface_energy).values) ≈ 0.25
    @test only(diagnostic_history(mixed_flow, :total_energy).values) ≈ 1.52
end

@testset "two-velocity divergence uses the projection volume flux" begin
    liquid = (x, y, time, axis) ->
    axis == 1 ? 0.2 + 0.3x - 0.1y : -0.4 + 0.2x + 0.25y
    gas = (x, y, time, axis) ->
    axis == 1 ? -0.3 + 0.4y : 0.1 + 0.2x
    for cut in (false, true), adaptive in (false, true)
        mesh = BlockTreeMesh(
            block_size = (4, 4), max_level = 2, periodic = (false, false),
            refine = (key, origin, extent) -> adaptive &&
                (key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0)),
        )
        geometry = cut ? GeometryField(
                mesh, (x, y) -> y - 0.16 - 0.1x; lipschitz = hypot(1, 0.1),
            ) : nothing
        interface = InterfaceField(
            mesh, (x, y) -> x + 0.23y - 0.43;
            geometry, lipschitz = hypot(1, 0.23),
        )
        cut && @test any(aperture -> 0 < aperture < 1, geometry.aperture)
        values = Float64[]
        for density in ((1.0, 1.0), (1000.0, 1.0))
            flow = FlowSimulation(
                mesh,
                TwoPhase(interface; density, dynamic_viscosity = (0.0, 0.0));
                geometry, velocity_model = TwoVelocityModel(),
                velocity = (liquid, gas), diagnostics = (DivergenceNorm(),),
            )
            @test any(isnan, flow.velocity[1].interior)
            @test any(isnan, flow.velocity[2].interior)
            expected = _reference_two_velocity_divergence(interface, flow.velocity)
            actual = only(diagnostic_history(flow, :divergence_norm).values)
            @test expected > 1.0e-3
            @test actual ≈ expected rtol = 2.0e-13 atol = 2.0e-13
            # Manually constructed states must use the same flux without a
            # simulation-owned workspace.
            state = SimulationState(flow.velocity, flow.pressure, flow.system, 0.0, 0)
            @test diagnostic_value(DivergenceNorm(), state) ≈ expected rtol = 2.0e-13 atol = 2.0e-13
            @test _warmed_diagnostic_allocations(
                DivergenceNorm(), simulation_state(flow),
            ) == 0
            push!(values, actual)
        end
        @test values[1] ≈ values[2] rtol = 2.0e-13 atol = 2.0e-13
    end

    # A uniform through-flow has zero divergence only when the prescribed
    # physical-boundary fluxes are included, even with absent-phase NaNs.
    mesh = BlockTreeMesh(block_size = (4, 4), periodic = (false, false))
    interface = InterfaceField(mesh, (x, y) -> x + 0.23y - 0.43)
    shared = (x, y, time, axis) -> axis == 1 ? 0.7 : -0.4
    flow = FlowSimulation(
        mesh,
        TwoPhase(
            interface; density = (1000.0, 1.0), dynamic_viscosity = (0.0, 0.0),
        );
        velocity_model = TwoVelocityModel(), velocity = (shared, shared),
        diagnostics = (DivergenceNorm(),),
    )
    @test only(diagnostic_history(flow, :divergence_norm).values) <= 128eps()
end

@testset "two-velocity projected-step divergence with slip" begin
    dt = 1.0e-3
    tolerance = 1.0e-9
    for cut in (false, true)
        mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, false))
        geometry = cut ? GeometryField(mesh, (x, y) -> y - 0.1; lipschitz = 1) : nothing
        interface = InterfaceField(
            mesh, (x, y) -> y - 0.47 - 0.03cospi(2x);
            geometry, lipschitz = 1.02,
        )
        liquid = (x, y, time, axis) -> axis == 1 ?
            0.18 + 0.04cospi(2y) : 0.02sinpi(2x) * sinpi(y)
        gas = (x, y, time, axis) -> axis == 1 ?
            -0.08 + 0.03cospi(2y) : 0.02sinpi(2x) * sinpi(y)
        flow = FlowSimulation(
            mesh,
            TwoPhase(
                interface; density = (1000.0, 1.0),
                dynamic_viscosity = (0.0, 0.0),
            );
            geometry, velocity_model = TwoVelocityModel(),
            velocity = (liquid, gas), boundary = FreeSlip(),
            diagnostics = (DivergenceNorm(),),
        )
        initial_liquid_aperture = copy(interface.phase_liquid_aperture)
        stats = step!(
            flow.velocity, flow.pressure, flow.system, dt, flow.workspace;
            tolerance, max_cycles = 4000,
        )
        @test stats.converged
        flow.time += dt
        flow.steps += 1
        record_diagnostics!(flow)
        values = diagnostic_history(flow, :divergence_norm).values
        @test length(values) == 2
        actual = last(values)
        reference = _reference_two_velocity_divergence(interface, flow.velocity)
        @test actual ≈ reference rtol = 2.0e-10 atol = 1.0e-11
        open_volume = isnothing(geometry) ? prod(mesh.extent) :
            integrate(geometry.open_fraction)
        # Projection controls dt*max(abs(div)); convert its bound to an L2
        # norm, with a roundoff allowance for the independent face sum.
        @test stats.mixture_volume_defect <= tolerance
        @test actual <= sqrt(open_volume) * stats.mixture_divergence + 1.0e-11
        @test actual <= sqrt(open_volume) * tolerance / dt + 1.0e-11

        liquid_aperture, gas_aperture =
            TwoPhaseAMR._current_phase_face_apertures!(interface)
        @test maximum(abs, liquid_aperture .- initial_liquid_aperture) > 128eps()
        mixed = filter(eachindex(mesh.faces)) do index
            liquid_aperture[index] > 0 && gas_aperture[index] > 0
        end
        @test !isempty(mixed)
        max_slip = maximum(mixed) do index
            abs(flow.velocity[1].interior[index] - flow.velocity[2].interior[index])
        end
        @test max_slip > 1.0e-2
        # Confirm this is a discriminating slip state: the previous
        # mass-weighted diagnostic is far above the projection tolerance.
        @test _reference_mass_weighted_divergence(flow) > 100tolerance / dt
    end
end

@testset "embedded and adaptive diagnostic measures" begin
    mesh = BlockTreeMesh(block_size = (8, 8), max_level = 1)
    geometry = GeometryField(mesh, (x, y) -> x - 0.5; lipschitz = 1)
    @test geometry.face_dual_volumes ==
        _component_volumes_by_global_face(
        mesh, geometry.dual_volumes, eltype(mesh.origin);
        covered_value = 0.0,
    )
    flow = FlowSimulation(
        mesh, SinglePhase(density = 2.0, dynamic_viscosity = 0.1);
        geometry, velocity = (2.0, 3.0),
        diagnostics = (
            KineticEnergy(), LinearMomentum(), DivergenceNorm(),
        ),
    )
    @test only(diagnostic_history(flow, :kinetic_energy).values) ≈ 6.5
    @test only(diagnostic_history(flow, :linear_momentum).values) ≈ [2.0, 3.0]
    divergence = CellField(mesh)
    divergence!(divergence, flow.velocity, geometry)
    expected_divergence = sqrt(
        sum(eachindex(mesh.blocks)) do block
            sum(
                abs2.(interior(divergence, block)) .*
                    interior(geometry.open_fraction, block),
            ) * mesh.blocks[block].volume
        end
    )
    @test only(diagnostic_history(flow, :divergence_norm).values) ≈
        expected_divergence

    cut_interface = InterfaceField(
        mesh, (x, y) -> y - 0.5; geometry, lipschitz = 1,
    )
    cut_flow = FlowSimulation(
        mesh,
        TwoPhase(
            cut_interface; density = (4.0, 1.0),
            dynamic_viscosity = (0.1, 0.1),
        );
        geometry,
        diagnostics = (PhaseVolume(), PhaseMass(), InterfaceMeasure()),
    )
    @test only(diagnostic_history(cut_flow, :phase_volume).values) ≈
        [0.25, 0.25]
    @test only(diagnostic_history(cut_flow, :phase_mass).values) ≈ [1.0, 0.25]
    @test only(diagnostic_history(cut_flow, :interface_measure).values) ≈ 1.0
    stable_timestep(cut_flow)
    @test test_allocated_call(() -> stable_timestep(cut_flow)) == 0
    if VERSION >= v"1.12"
        @test @allocated(stable_timestep(cut_flow)) == 0
    end
    cut_two = FlowSimulation(
        mesh,
        TwoPhase(
            cut_interface; density = (4.0, 1.0),
            dynamic_viscosity = (0.1, 0.1),
        );
        geometry,
        velocity_model = TwoVelocityModel(
            pressure_projection = CMDGFMProjection(),
        ),
        diagnostics = (),
    )
    stable_timestep(cut_two)
    @test test_allocated_call(() -> stable_timestep(cut_two)) == 0
    if VERSION >= v"1.12"
        @test @allocated(stable_timestep(cut_two)) == 0
    end

    energy_mesh = BlockTreeMesh(
        block_size = (4, 4), periodic = (false, false),
    )
    energy_geometry = GeometryField(
        energy_mesh, (x, y) -> x - 0.5; lipschitz = 1,
    )
    energy_interface = InterfaceField(
        energy_mesh, (x, y) -> y - 0.5;
        geometry = energy_geometry, lipschitz = 1,
    )
    energy_flow = FlowSimulation(
        energy_mesh,
        TwoPhase(
            energy_interface; density = (4.0, 1.0),
            dynamic_viscosity = (0.1, 0.1),
            surface_tension = 0.25, gravity = (0.0, -2.0),
        );
        geometry = energy_geometry,
        diagnostics = (PotentialEnergy(), SurfaceEnergy(), TotalEnergy()),
    )
    @test only(diagnostic_history(energy_flow, :potential_energy).values) ≈
        0.875
    @test only(diagnostic_history(energy_flow, :surface_energy).values) ≈ 0.125
    @test only(diagnostic_history(energy_flow, :total_energy).values) ≈ 1.0

    coarse_mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 1,
        periodic = (true, true),
    )
    coarse_interface = InterfaceField(
        coarse_mesh, (x, y) -> x - 0.5; lipschitz = 1,
    )
    refined_mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 1,
        periodic = (true, true),
    )
    refined_interface = InterfaceField(
        refined_mesh, (x, y) -> x - 0.5; lipschitz = 1,
    )
    adaptation = AMRWorkspace(refined_mesh, refined_interface)
    @test adapt!(adaptation, [:refine]; step = 1, force = true).changed
    phase(mesh, interface) = FlowSimulation(
        mesh,
        TwoPhase(
            interface; density = (4.0, 1.0),
            dynamic_viscosity = (0.1, 0.1),
        );
        velocity = (2.0, -1.0),
        diagnostics = (
            KineticEnergy(), LinearMomentum(), PhaseVolume(), PhaseMass(),
            InterfaceMeasure(),
        ),
    )
    coarse = phase(coarse_mesh, coarse_interface)
    refined = phase(refined_mesh, refined_interface)
    for name in (
            :kinetic_energy, :linear_momentum, :phase_volume, :phase_mass,
            :interface_measure,
        )
        @test only(diagnostic_history(refined, name).values) ≈
            only(diagnostic_history(coarse, name).values) atol = 2.0e-13
    end
end

@testset "diagnostic time schedules resume without duplicate boundaries" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    flow = FlowSimulation(
        mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.1);
        dt = 1.0e-3,
        diagnostics = (
            Diagnostic(:scheduled_time, state -> state.time) =>
                EveryTimeInterval(1.5e-3),
        ),
    )
    simulate!(flow, 3.0e-3)
    simulate!(flow, 5.0e-3)
    history = diagnostic_history(flow, :scheduled_time)
    @test count(==(3.0e-3), history.times) == 1
    @test length(unique(collect(zip(history.times, history.steps)))) ==
        length(history)
end

@testset "run summary reports the projection work of the whole run" begin
    # A developed flow usually starts a projection solve within its absolute
    # target `tolerance / dt^2`: that solve reports no iterations and leaves
    # its residual unchanged, so the final step alone says nothing about the
    # projection work of the run (#158).
    viscosity = inv(50.0)
    mesh = BlockTreeMesh(
        block_size = (12, 12), extent = (2pi, 2pi), max_level = 1,
        refine = (key, origin, extent) ->
        key.level == 0 || (key.level < 1 && key.i == 0 && key.j == 0),
    )
    velocity = (x, y, time, axis) -> exp(-2viscosity * time) *
        (axis == 1 ? sin(x) * cos(y) : -cos(x) * sin(y))
    flow = FlowSimulation(
        mesh, SinglePhase(density = 1.0, dynamic_viscosity = viscosity);
        velocity, boundary = FreeSlip(),
    )
    result = simulate!(flow, 0.01; dt = 1.0e-3)
    @test result.steps == 10
    # The last solve needed no iteration and reported no residual change.
    @test TwoPhaseAMR.iterations(result.pressure) == 0
    @test TwoPhaseAMR.final_residual(result.pressure) ==
        TwoPhaseAMR.initial_residual(result.pressure)
    # The run as a whole projected, so its accumulated statistics report work.
    @test result.projection.solves == result.steps
    @test result.projection.iterations > 0
    @test result.projection.without_iterations > 0
    @test result.projection.without_iterations < result.projection.solves
    @test result.projection.max_iterations >= result.projection.last_iterations
    @test result.projection.last_iterations > 0
    @test result.projection.last_final_residual <
        result.projection.last_initial_residual
    @test result.projection.converged

    report = IOBuffer()
    @test isnothing(print_simulation_summary(report, flow, result))
    @test occursin(
        "iterations this run: $(result.projection.iterations)",
        String(take!(report)),
    )

    # Every call accumulates its own run, not the flow's history.
    continued = simulate!(flow, 0.012; dt = 1.0e-3)
    @test continued.projection.solves == continued.steps == 2

    # A flow that starts every solve within its target reports no residual
    # reduction at all, and the summary still describes it.
    resting = FlowSimulation(
        BlockTreeMesh(block_size = (4, 4)),
        SinglePhase(density = 1.0, dynamic_viscosity = 0.01);
        boundary = FreeSlip(),
    )
    resting_result = simulate!(resting, 3.0e-3; dt = 1.0e-3)
    @test resting_result.projection.solves == resting_result.steps == 3
    @test resting_result.projection.iterations == 0
    @test resting_result.projection.without_iterations == resting_result.steps
    @test resting_result.projection.last_iterations == 0
    @test resting_result.projection.last_initial_residual ==
        resting_result.projection.last_final_residual == 0
    @test isnothing(
        print_simulation_summary(IOBuffer(), resting, resting_result),
    )
end

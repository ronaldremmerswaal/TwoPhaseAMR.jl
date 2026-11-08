using Test, TwoPhaseAMR

@testset "Diagnostic metadata of built-in diagnostics" begin
    energy = PhysicalDimension(1, 1, -2)
    for diagnostic in (KineticEnergy(), PotentialEnergy(), SurfaceEnergy(), TotalEnergy())
        info = diagnostic_info(diagnostic)
        @test info.dimension == energy
        @test info.kind === :physical
        @test info.scale === :linear
        @test isempty(info.components)
        @test endswith(info.label, "energy")
    end
    @test diagnostic_info(LinearMomentum()).components == ["x", "y"]
    @test diagnostic_info(LinearMomentum()).dimension == PhysicalDimension(1, 0, -1)
    @test diagnostic_info(PhaseVolume()).components == ["liquid", "gas"]
    @test diagnostic_info(PhaseVolume()).dimension == PhysicalDimension(0, 2, 0)
    @test diagnostic_info(PhaseMass()).dimension == PhysicalDimension(1, -1, 0)
    @test diagnostic_info(InterfaceMeasure()).dimension == PhysicalDimension(0, 1, 0)
    @test diagnostic_info(MaximumVelocity()).dimension == PhysicalDimension(0, 1, -1)
    @test diagnostic_info(CFLTimeStep()).kind === :solver
    divergence = diagnostic_info(DivergenceNorm())
    @test divergence.dimension == diagnostic_info(MaximumVelocity()).dimension
    @test (divergence.kind, divergence.scale) == (:defect, :log)
    @test diagnostic_info(LeafBlockCount()).kind === :count
    @test diagnostic_info(CellCount()).dimension == PhysicalDimension(0, 0, 0)

    @test string(energy) == "M L T⁻²"
    @test string(PhysicalDimension(0, 0, 0)) == "1"
    @test string(PhysicalDimension(-1, 12, 1)) == "M⁻¹ L¹² T"
    @test repr(energy) == "PhysicalDimension(1, 1, -2)"
end

@testset "Custom diagnostic metadata" begin
    plain = Diagnostic(:wave_amplitude, state -> 0.0)
    @test diagnostic_info(plain) == DiagnosticInfo(; label = "Wave amplitude")
    described = Diagnostic(
        :amplitude, state -> 0.0; label = "Amplitude", units = "m",
        dimension = PhysicalDimension(0, 1, 0),
    )
    @test diagnostic_info(described).label == "Amplitude"
    @test diagnostic_info(described).units == "m"
    @test Diagnostic(state -> 0.0, :reversed; kind = :solver).info.kind === :solver
    @test_throws ArgumentError Diagnostic(:name, :measure)
    @test_throws ArgumentError DiagnosticInfo(; label = "x", kind = :unknown)
    @test_throws ArgumentError DiagnosticInfo(; label = "x", scale = :sqrt)
    @test DiagnosticInfo(; label = "x", components = [:a, :b]).components == ["a", "b"]
    @test hash(DiagnosticInfo(; label = "x")) == hash(DiagnosticInfo(; label = "x"))

    mesh = BlockTreeMesh(block_size = (4, 4))
    flow = FlowSimulation(
        mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.01);
        boundary = FreeSlip(), dt = 0.01,
    )
    add_diagnostic!(
        flow, :momentum_pair, state -> (1.0, 2.0);
        label = "Momentum pair", components = ["first", "second"],
        dimension = PhysicalDimension(1, 0, -1),
    )
    add_diagnostic!(flow, :status, state -> "ok")
    add_diagnostic!(flow, KineticEnergy())
    recorder = Recorder(; diagnostics = flow, keyframes = nothing, preview = nothing)
    simulate!(flow, 0.02; observer = recorder)
    store = snapshot_store(recorder)
    pair = diagnostic_history(store, :momentum_pair)
    @test pair.components == ["first", "second"]
    @test pair.values == repeat([1.0 2.0], 3)
    @test pair.info.label == "Momentum pair"
    # Values that are not real scalars or vectors are not mirrored.
    @test length(diagnostic_history(store, :status)) == 0
    @test isempty(filter(event -> event.kind === :failure, run_events(store)))
    groups = diagnostic_groups(store)
    @test [group.names for group in groups] ==
        [[:momentum_pair], [:status], [:kinetic_energy]]
end

@testset "Diagnostics group by kind, dimension, and scale" begin
    infos = [
        :kinetic_energy => diagnostic_info(KineticEnergy()),
        :surface_energy => diagnostic_info(SurfaceEnergy()),
        :maximum_velocity => diagnostic_info(MaximumVelocity()),
        :divergence_norm => diagnostic_info(DivergenceNorm()),
        :cells => diagnostic_info(CellCount()),
        :leaf_blocks => diagnostic_info(LeafBlockCount()),
        :amplitude => DiagnosticInfo(; label = "Amplitude", units = "m"),
        :reference => DiagnosticInfo(; label = "Reference", units = "m"),
        :ratio => DiagnosticInfo(; label = "Ratio"),
        :other => DiagnosticInfo(; label = "Other"),
    ]
    groups = diagnostic_groups(infos)
    @test [group.names for group in groups] == [
        [:kinetic_energy, :surface_energy], [:maximum_velocity],
        [:divergence_norm], [:cells, :leaf_blocks], [:amplitude, :reference],
        [:ratio], [:other],
    ]
    @test groups[1].title == "Energy (M L T⁻²)"
    @test groups[3].scale === :log
    @test groups[4].title == "Count"
    @test groups[5].title == "m"
    @test groups[6].title == "Ratio"

    collection = DiagnosticCollection(Float64)
    @test isempty(diagnostic_groups(collection))
end

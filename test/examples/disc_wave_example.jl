using Test, TwoPhaseAMR

# The contract of the disc wave example's `run_example`; its accuracy is
# tested in test/integration/disc_wave.jl.
@testset "Standalone disc wave example" begin
    example = load_example(:disc_wave)
    states = []
    run = call_example(
        example, :run_example; resolution = 32, steps = 64,
        on_step = state -> push!(states, (state.step, state.time)),
    )
    @test run.steps == 64
    @test run.time ≈ 2pi / Base.invokelatest(getproperty, example, :bessel_zero)(0, 1)
    @test run.dt ≈ run.time / 64
    @test first.(states) == 0:64
    @test last(states)[2] == run.time
    @test length(run.energy) == 65
    # Each solve changes the energy by at most about its relative tolerance.
    @test maximum(abs, run.energy ./ first(run.energy) .- 1) <= run.steps * 1.0e-10
    @test all(isfinite, run.displacement.interior) && all(isfinite, run.rate.interior)
    @test run.iterations > 0
    @test isnothing(run.app)
    # One keyframe per step over one period of the (0,1) mode.
    @test length(keyframes(snapshot_store(run.recorder))) == 65

    # The composite mesh refines the blocks along the circle once.
    composite = call_example(
        example, :run_example; resolution = 32, composite = true, steps = 4, keyframes = nothing,
    )
    @test sort!(unique(block.key.level for block in composite.mesh.blocks)) == [0, 1]
    @test maximum(abs, composite.energy ./ first(composite.energy) .- 1) <= 4 * 1.0e-10

    # Studio: the driver is attached before the server starts, and the run
    # completes through it.
    live = call_example(
        example, :run_example; resolution = 16, steps = 4, keyframes = nothing,
        start_studio = example_start_probe,
    )
    @test isopen(live.app)
    close(live.app)
    @test live.steps == 4
    @test run_state(live.recorder) === :completed
end

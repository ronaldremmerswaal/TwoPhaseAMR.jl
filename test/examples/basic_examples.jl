using Test, TwoPhaseAMR

@testset "Standalone single-phase examples" begin
    for name in (:taylor_green, :lid_driven_cavity, :decaying_turbulence)
        example = load_example(name)
        run = call_example(
            example, :run_example; max_level = 0, final_time = 1.0e-6,
            start_studio = example_start_probe,
        )
        @test run.flow.velocity isa FaceVelocity
        @test isopen(run.app)
        close(run.app)
        @test run.result.time == 1.0e-6
        # A cavity at rest has an infinite stability bound, so simulate! caps
        # its first step at 1/128 of the interval; the others take one step.
        @test run.result.steps == (name == :lid_driven_cavity ? 128 : 1)
        @test run.result.pressure.converged
        if name == :lid_driven_cavity
            @test length(keyframes(snapshot_store(run.recorder))) >= 2
        else
            @test length(keyframes(snapshot_store(run.recorder))) == 2
        end
        @test :kinetic_energy in diagnostic_names(run.flow)
    end
end

@testset "Standalone VOF transport example" begin
    example = load_example(:vortex_reverse)
    for reconstruction_method in (:lhf, :lhf_ghf, :wy)
        run = call_example(
            example, :run_example; max_level = 0, final_time = 1.0e-3,
            reconstruction_method, start_studio = example_start_probe,
        )
        @test run.interface.reconstruction_method == reconstruction_method
        @test isopen(run.app)
        close(run.app)
        @test run.result.time ≈ 1.0e-3
        @test run.result.steps == 1
        @test !isempty(keyframes(snapshot_store(run.recorder)))
        @test run.result.volume_error <= 128eps()
    end
end

@testset "Example recorder and Studio lifecycle" begin
    example = load_example(:taylor_green)
    headless = call_example(
        example, :run_example; max_level = 0, final_time = 4.0e-6, dt = 1.0e-6,
    )
    @test isnothing(headless.app)
    @test all(info -> info.kind !== :preview, frames(snapshot_store(headless.recorder)))

    clock_time = Ref(0.0)
    clock = () -> (clock_time[] += 1.0)
    started = Ref{Any}(nothing)
    start_studio = recorder -> begin
        started[] = (recorder, run_state(recorder))
        IOBuffer()
    end
    live = call_example(
        example, :run_example; max_level = 0, final_time = 4.0e-6, dt = 1.0e-6,
        start_studio, recorder_options = (; keyframes = nothing, clock),
    )
    @test started[][1] === live.recorder
    @test started[][2] === :created
    @test run_state(live.recorder) === :completed
    @test isopen(live.app)
    @test any(info -> info.kind === :preview, frames(snapshot_store(live.recorder)))
    close(live.app)

    overridden = call_example(
        example, :run_example; max_level = 0, final_time = 4.0e-6, dt = 1.0e-6,
        start_studio, recorder_options = (; keyframes = nothing, preview = nothing, clock),
    )
    @test all(info -> info.kind !== :preview, frames(snapshot_store(overridden.recorder)))
    close(overridden.app)

    app = IOBuffer()
    @test_throws ArgumentError call_example(
        example, :run_example; max_level = 0, final_time = -1.0,
        start_studio = _ -> app,
    )
    @test !isopen(app)

    reversal = load_example(:vortex_reverse)
    stopped = call_example(
        reversal, :run_example; max_level = 0, final_time = 0.1,
        start_studio = recorder -> begin
            request_stop!(recorder)
            IOBuffer()
        end,
    )
    @test stopped.result.time == 0.0
    @test stopped.result.steps == 0
    @test run_state(stopped.recorder) === :completed
    close(stopped.app)
end

@testset "Vortex-reversal failure reporting and cleanup" begin
    example = load_example(:vortex_reverse)
    for fail_reporting in (false, true)
        app = IOBuffer()
        recorder = Ref{Any}()
        failure = ErrorException("clock failed")
        reporting_failure = ErrorException("failure-event clock failed")
        failed_clock_calls = Ref(0)
        clock = () -> begin
            if run_state(recorder[]) === :completed
                failed_clock_calls[] += 1
                failed_clock_calls[] == 1 && throw(failure)
                fail_reporting && throw(reporting_failure)
            end
            0.0
        end
        caught = try
            call_example(
                example, :run_example; max_level = 0, final_time = 0.1,
                recorder_options = (; keyframes = nothing, preview = nothing, clock),
                start_studio = observed -> begin
                    recorder[] = observed
                    request_stop!(observed)
                    app
                end,
            )
            nothing
        catch exception
            exception
        end
        @test caught === failure
        @test failed_clock_calls[] == 2
        @test !isopen(app)
        if !fail_reporting
            @test run_state(recorder[]) === :failed
            event = only(run_events(snapshot_store(recorder[])))
            @test event.kind === :failure
            @test event.time == 0.0 && event.step == 0
        end
    end
end

@testset "Adaptive decaying turbulence example" begin
    example = load_example(:decaying_turbulence)
    # Levels through 2 are protected by the example's minimum_level, so use
    # level 3 to exercise its pressure-driven coarsening criterion.
    run = call_example(
        example, :run_example;
        max_level = 3, final_time = 8.0e-4, dt = 1.0e-4,
    )
    @test run.result.time ≈ 8.0e-4
    @test run.result.steps == 8
    @test run.result.pressure.converged
    @test run.result.mesh_updates > 0
    test_finite_example_fields(run.flow)
end

@testset "Adaptive vortex-reversal example" begin
    example = load_example(:vortex_reverse)
    # The smaller grid needs this much transport to cross a refinement
    # boundary and exercise an actual mesh update.
    run = call_example(
        example, :run_example;
        max_level = 2, final_time = 0.6,
        recorder_options = (; keyframes = EveryStep()),
    )
    @test run.result.time ≈ 0.6
    @test run.result.steps >= 5
    @test run.result.mesh_updates > 0
    store = snapshot_store(run.recorder)
    recorded = keyframes(store)
    log = step_log(store)
    @test [info.step for info in recorded] == collect(0:run.result.steps)
    @test log.step == collect(1:run.result.steps)
    @test count(log.topology_changed) == run.result.mesh_updates
    @test any(!, log.topology_changed)
    @test log.topology_changed == [
        recorded[i].topology != recorded[i + 1].topology
            for i in eachindex(log.step)
    ]
    @test all(isfinite, run.velocity.interior)
    @test all(isfinite, run.velocity.boundary)
    @test all(
        isfinite(value)
            for block in eachindex(run.interface.mesh.blocks)
            for value in interior(run.interface, block)
    )
    @test run.result.volume_error <=
        8 * run.interface.fraction_floor * prod(run.interface.mesh.extent)
end

@testset "Manufactured Poisson example" begin
    example = load_example(:manufactured_poisson)
    for method in (:mimetic, :joint_face, :cmdgfm)
        run = call_example(
            example, :run_example; resolution = 8, method,
        )
        @test run.method == method
        @test run.density[1] / run.density[2] == 1000
        @test 0.8 <= run.analytical_jump_components.rms_ratio <= 1.25
        @test run.stats.converged
        @test field_names(run.registry)[1] == :pressure
        @test all(isfinite, values(run.errors.pressure))
        @test all(isfinite, values(run.errors.gradient))
        @test all(isfinite, values(run.errors.xi.x))
        @test all(isfinite, values(run.errors.xi.y))
        @test run.errors.pressure.l2 <= run.errors.pressure.linf
        @test run.errors.gradient.l2 <= run.errors.gradient.linf
        @test run.errors.xi.x.l2 <= run.errors.xi.x.linf
        @test run.errors.xi.y.l2 <= run.errors.xi.y.linf
    end
end

@testset "Vortex-reversal Studio driver continues its prescribed flow" begin
    example = load_example(:vortex_reverse)
    run = call_example(
        example, :run_example; max_level = 0, final_time = 0.1,
        start_studio = recorder -> begin
            request_stop!(recorder)
            IOBuffer()
        end,
    )
    try
        @test run.result.stopped && run.result.time == 0
        driver = run.recorder.driver
        @test driver isa TwoPhaseAMR.StudioDriver
        store = snapshot_store(run.recorder)
        initial_id = first(frames(store)).id
        short = simulate!(driver, 0.05)
        @test !short.stopped && short.time ≈ 0.05 && short.steps > 0
        short_steps = short.steps
        continued = simulate!(driver, 0.15)
        @test !continued.stopped && continued.time ≈ 0.15
        @test continued.steps > short_steps
        @test step_log(store).step == collect(1:continued.steps)
        @test first(frames(store)).id == initial_id
        @test count(info -> info.time == 0.0, frames(store)) == 1
        @test count(info -> isapprox(info.time, 0.05; atol = 1.0e-12, rtol = 0), frames(store)) == 1
        @test any(info -> any(t -> isapprox(t, 0.1; atol = 1.0e-12, rtol = 0),
            info.requested_times), frames(store))

        # The carrier's reversal period is the original example final_time,
        # not the newly requested end of either continuation.
        log = step_log(store)
        midpoint = continued.time - last(log.dt) / 2
        streamfunction(x, y) = sinpi(x)^2 * sinpi(y)^2 / pi
        expected = map(run.interface.mesh.faces) do face
            x, y = face.center
            half_width = face.area / 2
            value = if face.axis == 1
                (streamfunction(x, y + half_width) - streamfunction(x, y - half_width)) / face.area
            else
                -(streamfunction(x + half_width, y) - streamfunction(x - half_width, y)) / face.area
            end
            cospi(midpoint / 0.1) * value
        end
        @test run.velocity.interior ≈ expected atol = 1.0e-12 rtol = 1.0e-12
    finally
        close(run.app)
    end
end

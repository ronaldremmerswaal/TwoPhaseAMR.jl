mutable struct FrameRecorder <: AbstractVisualization
    frames::Vector{VisualizationFrame}
    gate::Union{Nothing,Channel{Nothing}}
    session::Any
    state_updates::Int
end

FrameRecorder(; gate=nothing) =
    FrameRecorder(VisualizationFrame[], gate, nothing, 0)

function TwoPhaseAMR.update_visualization!(viewer::FrameRecorder,
                                           frame::VisualizationFrame)
    isnothing(viewer.gate) || take!(viewer.gate)
    push!(viewer.frames, frame)
    return viewer
end

function TwoPhaseAMR._bind_visualization_session!(
        viewer::FrameRecorder, session::VisualizationSession)
    viewer.session = session
    return viewer
end

function TwoPhaseAMR._update_visualization_state!(
        viewer::FrameRecorder, ::VisualizationSession)
    viewer.state_updates += 1
    return viewer
end

struct FailingVisualization <: AbstractVisualization end

TwoPhaseAMR.update_visualization!(::FailingVisualization,
                                  ::VisualizationFrame) =
    error("test viewer failure")

function _publish_checkpoint_test_frame!(session, field, time, step;
                                         event=:step, final_time=1.0)
    initialize!(field, (x, y) -> time)
    return update_visualization!(
        session; event, time, step, final_time,
        velocity=nothing, pressure=field, solver=nothing)
end

@testset "Visualization checkpoint schedules" begin
    mesh = BlockTreeMesh(block_size=(4, 4))
    field = CellField(mesh)
    registry = VisualizationFieldRegistry()
    register_field!(registry, :state, field)

    count_session = VisualizationSession(registry; checkpoints=5)
    for (step, time) in enumerate((0.0, 0.18, 0.44, 0.68, 1.0))
        event = step == 1 ? :initial : :step
        _publish_checkpoint_test_frame!(
            count_session, field, time, step - 1; event)
    end
    count_checkpoints = visualization_checkpoints(count_session)
    @test count_session.checkpoint_times == [0.0, 0.25, 0.5, 0.75, 1.0]
    @test [checkpoint.requested_time for checkpoint in count_checkpoints] ==
        count_session.checkpoint_times
    @test [checkpoint.frame.time for checkpoint in count_checkpoints] ==
        [0.0, 0.18, 0.44, 0.68, 1.0]
    @test all(all(==(checkpoint.frame.time), checkpoint.frame[:state])
              for checkpoint in count_checkpoints)

    # Retained arrays do not alias a subsequently mutated solver field.
    fill!(interior(field, 1), 99.0)
    @test all(iszero, count_checkpoints[1].frame[:state])

    interval_session = VisualizationSession(registry; checkpoint_dt=0.3)
    _publish_checkpoint_test_frame!(
        interval_session, field, 0.0, 0; event=:initial)
    @test interval_session.checkpoint_times ≈ [0.0, 0.3, 0.6, 0.9, 1.0]

    shorthand_interval = VisualizationSession(registry; checkpoints=0.25)
    _publish_checkpoint_test_frame!(
        shorthand_interval, field, 0.0, 0; event=:initial)
    @test shorthand_interval.checkpoint_times == [0.0, 0.25, 0.5, 0.75, 1.0]

    explicit_session = VisualizationSession(
        registry; checkpoint_times=[0.7, 0.2, 0.7])
    for (step, time) in enumerate((0.0, 0.3, 0.6, 0.9, 1.0))
        event = step == 1 ? :initial : :step
        _publish_checkpoint_test_frame!(
            explicit_session, field, time, step - 1; event)
    end
    @test explicit_session.checkpoint_times == [0.2, 0.7]
    @test [checkpoint.frame.time for checkpoint in
           visualization_checkpoints(explicit_session)] == [0.3, 0.6]
    @test latest_frame(explicit_session).time == 1.0

    select_checkpoint!(explicit_session, 1)
    @test explicit_session.mode == :checkpoint
    @test explicit_session.selected_checkpoint == 1
    live_mode!(explicit_session)
    @test explicit_session.mode == :live
    @test isnothing(explicit_session.selected_checkpoint)

    one_session = VisualizationSession(registry; checkpoints=1)
    _publish_checkpoint_test_frame!(
        one_session, field, 0.0, 0; event=:initial, final_time=2.0)
    @test one_session.checkpoint_times == [2.0]

    @test_throws ArgumentError VisualizationSession(
        registry; checkpoints=4, checkpoint_dt=0.1)
    @test_throws ArgumentError VisualizationSession(
        registry; checkpoint_dt=0.0)
    outside_session = VisualizationSession(
        registry; checkpoint_times=[-0.1, 0.5])
    @test_throws ArgumentError _publish_checkpoint_test_frame!(
        outside_session, field, 0.0, 0; event=:initial)

    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? 3.0 : 4.0)
    velocity_registry = VisualizationFieldRegistry()
    register_field!(velocity_registry, :velocity_x, velocity, Face{1}())
    register_field!(velocity_registry, :velocity_y, velocity, Face{2}())
    register_field!(
        velocity_registry, :velocity_magnitude,
        velocity, CenteredMagnitude())
    register_field!(
        velocity_registry, :vorticity, velocity, Vorticity())
    velocity_session = VisualizationSession(velocity_registry)
    update_visualization!(
        velocity_session; event=:initial, time=0.0, step=0,
        final_time=1.0, velocity, pressure=field, solver=nothing)
    frame = latest_frame(velocity_session)
    @test length(frame.primitives) == 1
    @test all(==(5.0), frame[:velocity_magnitude])
    @test all(iszero, frame[:vorticity])

    xfield, yfield = FaceField{1}(velocity), FaceField{2}(velocity)
    pair_registry = VisualizationFieldRegistry()
    register_field!(pair_registry, :x, xfield)
    register_field!(pair_registry, :y, yfield)
    register_field!(
        pair_registry, :magnitude, (xfield, yfield), CenteredMagnitude())
    register_field!(pair_registry, :curl, (xfield, yfield), Vorticity())
    pair_session = VisualizationSession(pair_registry)
    update_visualization!(
        pair_session; event=:initial, time=0.0, step=0,
        final_time=1.0, velocity, pressure=field, solver=nothing)
    pair_frame = latest_frame(pair_session)
    @test length(pair_frame.primitives) == 2
    @test all(==(5.0), pair_frame[:magnitude])
    @test all(iszero, pair_frame[:curl])
end

@testset "Live-frame sampling and latest-wins publication" begin
    mesh = BlockTreeMesh(block_size=(4, 4))
    field = CellField(mesh)
    registry = VisualizationFieldRegistry()
    register_field!(registry, :state, field)

    step_viewer = FrameRecorder()
    step_session = VisualizationSession(
        registry; viewer=step_viewer, live_every=2)
    _publish_checkpoint_test_frame!(
        step_session, field, 0.0, 0; event=:initial, final_time=0.5)
    for step in 1:5
        _publish_checkpoint_test_frame!(
            step_session, field, step / 10, step; final_time=0.5)
    end
    _publish_checkpoint_test_frame!(
        step_session, field, 0.5, 5; event=:completed, final_time=0.5)
    @test [frame.step for frame in step_viewer.frames] == [0, 2, 4, 5]
    @test [frame.time for frame in step_viewer.frames] == [0.0, 0.2, 0.4, 0.5]
    @test latest_frame(step_session).time == 0.5
    @test step_viewer.session === step_session

    time_viewer = FrameRecorder()
    time_session = VisualizationSession(
        registry; viewer=time_viewer, live_dt=0.25)
    for (step, time) in enumerate((0.0, 0.1, 0.3, 0.4, 0.7))
        event = step == 1 ? :initial : :step
        _publish_checkpoint_test_frame!(
            time_session, field, time, step - 1;
            event, final_time=0.7)
    end
    _publish_checkpoint_test_frame!(
        time_session, field, 0.7, 4; event=:completed, final_time=0.7)
    @test [frame.time for frame in time_viewer.frames] == [0.0, 0.3, 0.7]

    historical_viewer = FrameRecorder()
    historical_session = VisualizationSession(
        registry; viewer=historical_viewer, checkpoint_times=[0.0])
    _publish_checkpoint_test_frame!(
        historical_session, field, 0.0, 0;
        event=:initial, final_time=0.2)
    select_checkpoint!(historical_session, 1)
    _publish_checkpoint_test_frame!(
        historical_session, field, 0.1, 1; final_time=0.2)
    @test historical_viewer.state_updates == 1
    @test last(historical_viewer.frames).time == 0.0
    @test latest_frame(historical_session).time == 0.1
    live_mode!(historical_session)
    @test last(historical_viewer.frames).time == 0.1

    gate = Channel{Nothing}(2)
    slow_viewer = FrameRecorder(; gate)
    asynchronous_session = VisualizationSession(
        registry; viewer=slow_viewer,
        checkpoint_times=[0.0, 0.1, 0.2, 0.3], asynchronous=true)
    _publish_checkpoint_test_frame!(
        asynchronous_session, field, 0.0, 0;
        event=:initial, final_time=0.3)
    yield()
    @test isempty(slow_viewer.frames)
    for step in 1:3
        _publish_checkpoint_test_frame!(
            asynchronous_session, field, step / 10, step;
            final_time=0.3)
    end
    _publish_checkpoint_test_frame!(
        asynchronous_session, field, 0.3, 3;
        event=:completed, final_time=0.3)
    @test length(visualization_checkpoints(asynchronous_session)) == 4
    @test dropped_visualization_frames(asynchronous_session) == 2
    put!(gate, nothing)
    put!(gate, nothing)
    @test wait_visualization!(asynchronous_session) === asynchronous_session
    @test [frame.time for frame in slow_viewer.frames] == [0.0, 0.3]
    @test all(==(0.0), slow_viewer.frames[1][:state])
    @test all(==(0.3), slow_viewer.frames[2][:state])

    running_session = VisualizationSession(registry; asynchronous=true)
    _publish_checkpoint_test_frame!(
        running_session, field, 0.0, 0; event=:initial)
    @test_throws ArgumentError wait_visualization!(running_session)
    update_visualization!(
        running_session; event=:failed, time=0.0, step=0,
        final_time=1.0, velocity=nothing, pressure=field, solver=nothing)
    wait_visualization!(running_session)

    failing_session = VisualizationSession(
        registry; viewer=FailingVisualization(), asynchronous=true)
    _publish_checkpoint_test_frame!(
        failing_session, field, 0.0, 0; event=:initial)
    yield()
    @test failing_session.state == :failed
    @test_throws TaskFailedException wait_visualization!(failing_session)

    @test_throws ArgumentError VisualizationSession(
        registry; live_every=2, live_dt=0.1)
    @test_throws ArgumentError VisualizationSession(registry; live_every=0)
    @test_throws ArgumentError VisualizationSession(registry; live_dt=0.0)
end

@testset "Cooperative simulation pause and resume" begin
    mesh = BlockTreeMesh(block_size=(4, 4))
    pressure = CellField(mesh)
    registry = VisualizationFieldRegistry()
    register_field!(registry, :pressure, pressure)

    created = VisualizationSession(registry)
    @test simulation_state(created) == :created
    request_pause!(created)
    @test simulation_state(created) == :pause_requested
    resume_simulation!(created)
    @test simulation_state(created) == :created
    @test_throws ArgumentError resume_simulation!(created)
    @test_throws ArgumentError wait_until_paused(created)
    request_pause!(created)
    initial_update = @async _publish_checkpoint_test_frame!(
        created, pressure, 0.0, 0; event=:initial, final_time=0.2)
    wait_until_paused(created)
    @test simulation_state(created) == :paused
    resume_simulation!(created)
    fetch(initial_update)
    _publish_checkpoint_test_frame!(
        created, pressure, 0.0, 0; event=:completed, final_time=0.2)

    viewer = FrameRecorder()
    session = VisualizationSession(registry; viewer)
    _publish_checkpoint_test_frame!(
        session, pressure, 0.0, 0; event=:initial, final_time=0.2)
    @test simulation_state(session) == :running
    @test request_pause!(session) === session
    @test simulation_state(session) == :pause_requested

    boundary_update = @async _publish_checkpoint_test_frame!(
        session, pressure, 0.1, 1; final_time=0.2)
    @test wait_until_paused(session) === session
    @test simulation_state(session) == :paused
    @test !istaskdone(boundary_update)
    @test resume_simulation!(session) === session
    fetch(boundary_update)
    @test simulation_state(session) == :running

    # Resuming a pending request before the next boundary cancels it.
    request_pause!(session)
    @test simulation_state(session) == :pause_requested
    resume_simulation!(session)
    @test simulation_state(session) == :running
    _publish_checkpoint_test_frame!(
        session, pressure, 0.2, 2; event=:completed, final_time=0.2)
    @test simulation_state(session) == :completed
    @test_throws ArgumentError request_pause!(session)
    @test_throws ArgumentError resume_simulation!(session)

    failed_session = VisualizationSession(registry)
    _publish_checkpoint_test_frame!(
        failed_session, pressure, 0.0, 0;
        event=:initial, final_time=0.2)
    request_pause!(failed_session)
    pause_waiter = @async try
        wait_until_paused(failed_session)
    catch exception
        exception
    end
    yield()
    _publish_checkpoint_test_frame!(
        failed_session, pressure, 0.0, 0;
        event=:failed, final_time=0.2)
    @test fetch(pause_waiter) isa ErrorException
    @test simulation_state(failed_session) == :failed

    # Exercise the same observer-controlled boundary wait through simulate!.
    velocity = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    solver = NavierStokesSolver(
        mesh; viscosity=0.01, boundary=FreeSlip(), coarse_size=8)
    workspace = NavierStokesWorkspace(solver)
    simulation_registry = VisualizationFieldRegistry()
    register_field!(simulation_registry, :pressure, pressure)
    simulation_session = VisualizationSession(simulation_registry)
    entered_step = Channel{Nothing}(1)
    continue_step = Channel{Nothing}(1)
    first_force_call = Ref(true)
    force = function (x, y, time, axis)
        if first_force_call[]
            first_force_call[] = false
            put!(entered_step, nothing)
            take!(continue_step)
        end
        return 0.0
    end
    simulation = Threads.@spawn simulate!(
        velocity, pressure, solver, 0.2, workspace;
        dt=0.1, force, visualization=simulation_session,
        rtol=1e-10, max_cycles=20)
    take!(entered_step)
    request_pause!(simulation_session)
    @test simulation_state(simulation_session) == :pause_requested
    put!(continue_step, nothing)
    wait_until_paused(simulation_session)
    @test simulation_state(simulation_session) == :paused
    @test latest_frame(simulation_session).time == 0.1
    @test !istaskdone(simulation)
    resume_simulation!(simulation_session)
    result = fetch(simulation)
    @test result.time == 0.2
    @test result.steps == 2
    @test simulation_state(simulation_session) == :completed
end

@testset "Exact visualization checkpoints use independent side steps" begin
    mesh = BlockTreeMesh(block_size=(4, 4))
    force = (x, y, time, axis) -> axis == 1 ? 0.1sinpi(2y) : 0.0

    baseline_velocity = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    baseline_pressure = CellField(mesh)
    baseline_solver = NavierStokesSolver(
        mesh; viscosity=0.01, boundary=FreeSlip(), coarse_size=8)
    baseline_workspace = NavierStokesWorkspace(baseline_solver)
    baseline = simulate!(
        baseline_velocity, baseline_pressure, baseline_solver, 1.0,
        baseline_workspace; dt=0.3, force, rtol=1e-10, max_cycles=20)

    velocity = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    pressure = CellField(mesh)
    solver = NavierStokesSolver(
        mesh; viscosity=0.01, boundary=FreeSlip(), coarse_size=8)
    workspace = NavierStokesWorkspace(solver)
    registry = VisualizationFieldRegistry()
    register_field!(registry, :pressure, pressure)
    register_field!(registry, :velocity_x, velocity, Face{1}())
    session = VisualizationSession(
        registry; checkpoint_times=[0.2, 0.65], exact_checkpoints=true)

    result = simulate!(
        velocity, pressure, solver, 1.0, workspace;
        dt=0.3, force, visualization=session, rtol=1e-10, max_cycles=20)

    @test result.steps == baseline.steps == 4
    @test result.time == 1.0
    @test velocity.interior == baseline_velocity.interior
    @test velocity.boundary == baseline_velocity.boundary
    @test visualization_values(pressure) ==
        visualization_values(baseline_pressure)
    @test any(!iszero, velocity.interior)
    @test session.state == :completed
    @test session.step_procedure isa Function
    @test session.checkpoint_state_time == 1.0
    @test [checkpoint.requested_time for checkpoint in
           visualization_checkpoints(session)] == [0.2, 0.65]
    @test [checkpoint.frame.time for checkpoint in
           visualization_checkpoints(session)] == [0.2, 0.65]
    @test any(!iszero,
              first(visualization_checkpoints(session)).frame[:velocity_x])
end

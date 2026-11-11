using Test, TwoPhaseAMR

# Simulation times of an AMR run: large steps early, clustered steps where the
# adaptive time step shrank, a repeated time, and a late jump.
const _IRREGULAR_TIMES = [
    0.0, 0.004, 0.007, 0.019, 0.020, 0.021, 0.050, 0.310, 0.620, 1.0,
]

function _local_half_gap(times, target)
    index = searchsortedfirst(times, target)
    index > length(times) && return (target - last(times), lastindex(times))
    index == 1 && return (first(times) - target, 1)
    return (times[index] - times[index - 1]) / 2, index
end

# Every output frame shows a stored frame within half the surrounding gap,
# and output times are uniform and monotone.
function _check_linear(times, schedule)
    @test issorted(schedule.indices)
    @test length(schedule.indices) == length(schedule.times)
    if length(schedule) > 1
        steps = diff(schedule.times)
        @test all(>(0), steps)
        @test maximum(steps) - minimum(steps) <= 64 * eps(maximum(abs, schedule.times))
    end
    for (index, target) in zip(schedule.indices, schedule.times)
        half_gap, _ = _local_half_gap(times, target)
        @test abs(times[index] - target) <= half_gap + 64 * eps(1.0)
    end
    return nothing
end

@testset "video frame count" begin
    @test video_frame_count(20, 2.0) == 41
    @test video_frame_count(10, 0.15) == 3
    @test video_frame_count(1, 0.1) == 2
    @test video_frame_count(240, 10.0) == 2401
    @test_throws ArgumentError video_frame_count(0, 1.0)
    @test_throws ArgumentError video_frame_count(20, 0.0)
    @test_throws ArgumentError video_frame_count(Inf, 1.0)
    @test_throws ArgumentError video_frame_count(2, NaN)
end

@testset "uniform output times on irregular stored frames" begin
    times = _IRREGULAR_TIMES
    schedule = video_frame_schedule(times, 0.0, 1.0, 11)
    @test length(schedule) == 11
    @test schedule.times ≈ collect(range(0.0, 1.0; length = 11))
    _check_linear(times, schedule)
    # Irregular and clustered stored times still produce a uniform output
    # time line; the mapping skips the cluster and holds the far frames.
    @test schedule.indices == [1, 7, 8, 8, 8, 9, 9, 9, 9, 10, 10]

    # Dense recording: fewer output frames than stored frames, so frames are
    # skipped and the mapping stays linear.
    dense = collect(range(0.0, 1.0; length = 101))
    dense_schedule = video_frame_schedule(dense, 0.0, 1.0, 11)
    @test dense_schedule.indices == collect(1:10:101)
    _check_linear(dense, dense_schedule)

    # Sparse recording: only two stored frames, so output frames repeat.
    sparse = [0.0, 1.0]
    sparse_schedule = video_frame_schedule(sparse, 0.0, 1.0, 11)
    @test sparse_schedule.indices == [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    @test length(unique(sparse_schedule.indices)) == 2
    _check_linear(sparse, sparse_schedule)
end

@testset "window boundaries" begin
    times = _IRREGULAR_TIMES
    schedule = video_frame_schedule(times, 0.019, 0.31, 5)
    @test first(schedule.times) == 0.019 && last(schedule.times) == 0.31
    @test schedule.indices == [4, 7, 7, 8, 8]
    _check_linear(times, schedule)

    # A window wider than the recording holds the boundary frames.
    clamped = video_frame_schedule([0.2, 0.7], 0.1, 0.7, 3)
    @test clamped.times ≈ [0.1, 0.4, 0.7]
    @test clamped.indices == [1, 1, 2]

    # A window inside a dense run only selects from stored frames but the
    # video time line still spans the requested window.
    dense = collect(range(0.0, 1.0; length = 101))
    narrow = video_frame_schedule(dense, 0.219, 0.281, 7)
    @test narrow.times ≈ collect(range(0.219, 0.281; length = 7))
    @test narrow.indices == [23, 24, 25, 26, 27, 28, 29]
    _check_linear(dense, narrow)
end

@testset "ties choose the earlier frame" begin
    @test video_frame_schedule([0.0, 0.5, 1.0], 0.25, 0.75, 2).indices == [1, 2]
    @test video_frame_schedule([0.0, 1.0], 0.0, 1.0, 3).indices == [1, 1, 2]
    # A target at a repeated stored time resolves to its first occurrence;
    # off that time the nearest of the two neighbours wins.
    @test video_frame_schedule([0.2, 0.2, 0.5], 0.2, 0.5, 2).indices == [1, 3]
    @test video_frame_schedule([0.2, 0.2, 0.5], 0.2, 0.5, 4).indices == [1, 2, 3, 3]
end

@testset "single-frame recordings and single output frames" begin
    one = video_frame_schedule([0.4], 0.4, 0.4, 5)
    @test one.indices == fill(1, 5)
    @test all(==(0.4), one.times)
    one_window = video_frame_schedule([0.4], 0.2, 0.6, 3)
    @test one_window.indices == fill(1, 3)
    @test one_window.times == [0.2, 0.4, 0.6]
    single = video_frame_schedule(_IRREGULAR_TIMES, 0.31, 0.9, 1)
    @test single.times == [0.31] && single.indices == [8]
    empty_like = video_frame_schedule([0.2, 0.4], 0.3, 0.3, 4)
    @test empty_like.indices == fill(1, 4) # every target is the same midpoint tie
end

@testset "fps and duration" begin
    times = _IRREGULAR_TIMES
    schedule = video_frame_schedule(times, 0.0, 1.0; fps = 10, duration = 1.0)
    @test length(schedule) == 11
    @test schedule.times ≈ collect(range(0.0, 1.0; length = 11))
    _check_linear(times, schedule)

    exact = video_frame_schedule(times, 0.0, 1.0; fps = 12, nframes = 25)
    @test length(exact) == 25
    @test exact.times ≈ collect(range(0.0, 1.0; length = 25))

    @test_throws ArgumentError video_frame_schedule(times, 0.0, 1.0; fps = 10)
    @test_throws ArgumentError video_frame_schedule(
        times, 0.0, 1.0; fps = 10, duration = 1.0, nframes = 11,
    )
    @test_throws ArgumentError video_frame_schedule(times, 0.0, 1.0; fps = 0, duration = 1.0)
end

@testset "schedule validation" begin
    @test_throws ArgumentError video_frame_schedule(Float64[], 0.0, 1.0, 3)
    @test_throws ArgumentError video_frame_schedule([0.0, 0.5, 0.4], 0.0, 0.5, 3)
    @test_throws ArgumentError video_frame_schedule([0.0, NaN], 0.0, 0.5, 3)
    @test_throws ArgumentError video_frame_schedule([0.0, 0.5], 0.6, 0.5, 3)
    @test_throws ArgumentError video_frame_schedule([0.0, 0.5], 0.0, Inf, 3)
    @test_throws ArgumentError video_frame_schedule([0.0, 0.5], 0.0, 0.5, 0)
    # Repeated times are allowed and the earlier occurrence is used.
    @test video_frame_schedule([0.0, 0.0, 0.5], 0.0, 0.5, 2).indices == [1, 3]
end

struct TestFrame
    id::Int
    time::Float64
end

@testset "live sampler matches a stored schedule" begin
    # A live export samples as frames arrive; over the same window it must
    # choose the same frames as a stored export when none were lost.
    times = _IRREGULAR_TIMES
    frames = [TestFrame(index, time) for (index, time) in enumerate(times)]
    step = 0.1
    sampler = LiveVideoSampler{TestFrame}(first(times), step)
    shown = TestFrame[]
    for frame in frames
        append!(shown, push_live_frame!(sampler, frame, frame.time))
    end
    append!(shown, flush_live_frames!(sampler))
    stored = video_frame_schedule(times, first(times), last(times), 11)
    @test [frame.id for frame in shown] == stored.indices
    @test [frame.time for frame in shown] == times[stored.indices]
    @test issorted([frame.time for frame in shown])
    # Pushing more frames after a flush would continue the same time line.
    @test sampler.next_index == 12

    # Sparse and dense streams repeat or skip, exactly like the stored rule.
    sparse_frames = [TestFrame(1, 0.0), TestFrame(2, 1.0)]
    sparse_sampler = LiveVideoSampler{TestFrame}(0.0, 0.1)
    sparse_shown = TestFrame[]
    for frame in sparse_frames
        append!(sparse_shown, push_live_frame!(sparse_sampler, frame, frame.time))
    end
    append!(sparse_shown, flush_live_frames!(sparse_sampler))
    @test [frame.id for frame in sparse_shown] == vcat(fill(1, 6), fill(2, 5))
    @test length(sparse_shown) == 11

    dense_times = collect(range(0.0, 1.0; length = 101))
    dense_frames = [TestFrame(index, time) for (index, time) in enumerate(dense_times)]
    dense_sampler = LiveVideoSampler{TestFrame}(0.0, 0.1)
    dense_shown = TestFrame[]
    for frame in dense_frames
        append!(dense_shown, push_live_frame!(dense_sampler, frame, frame.time))
    end
    append!(dense_shown, flush_live_frames!(dense_sampler))
    @test [frame.id for frame in dense_shown] == collect(1:10:101)

    # Stopping early keeps only output times up to the last arrived frame.
    early = LiveVideoSampler{TestFrame}(0.0, 0.1)
    early_shown = TestFrame[]
    for frame in frames
        frame.time <= 0.65 || break
        append!(early_shown, push_live_frame!(early, frame, frame.time))
    end
    append!(early_shown, flush_live_frames!(early))
    @test length(early_shown) == 7
    @test [frame.id for frame in early_shown] == first(stored.indices, 7)

    # The first frame also decides output times before it.
    late = LiveVideoSampler{TestFrame}(0.0, 0.1)
    first_late = push_live_frame!(late, TestFrame(7, 0.02), 0.02)
    @test [frame.id for frame in first_late] == [7]

    @test_throws ArgumentError LiveVideoSampler{TestFrame}(0.0, 0.0)
    @test_throws ArgumentError push_live_frame!(late, TestFrame(8, 0.01), 0.01)
    @test_throws ArgumentError push_live_frame!(late, TestFrame(8, NaN), NaN)
end

@testset "report" begin
    text = sprint(show, video_frame_schedule([0.0, 1.0], 0.0, 1.0, 3))
    @test occursin("3 output frames", text) && occursin("2 stored frames", text)
end

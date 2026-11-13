# Refining the step of a consistent scheme makes its solution converge to the
# time-continuous one, so a coarse run of a series must approach the finest
# run of that series. The error against the exact solution is not a refinement
# gate: at a fixed mesh it converges to the spatial error, and it increases
# while a temporal error of the opposite sign stops cancelling it, which the
# correct one-velocity capillary wave shows (#178). The checks in this suite
# therefore gate the temporal self-convergence, the difference to the
# finest-step run of the same series.
if !isdefined(@__MODULE__, :_linear_sample)
    """Sample `values`, given at the increasing `times`, linearly at the increasing `target_times`."""
    function _linear_sample(times, values, target_times)
        length(times) == length(values) || throw(
            DimensionMismatch(
                "sample times and values must have equal length",
            ),
        )
        length(times) >= 2 || throw(
            ArgumentError(
                "at least two samples are required for interpolation",
            ),
        )
        all(diff(times) .> 0) ||
            throw(ArgumentError("sample times must increase"))
        all(diff(target_times) .> 0) ||
            throw(ArgumentError("target times must increase"))
        samples = similar(values, length(target_times))
        interval = 1
        for (position, target) in enumerate(target_times)
            # The runs of a series share their final time, but a coarse run can
            # stop within roundoff of it; clamp instead of failing there.
            time = clamp(target, first(times), last(times))
            while interval < length(times) - 1 && times[interval + 1] < time
                interval += 1
            end
            lower = times[interval]
            upper = times[interval + 1]
            samples[position] = values[interval] + (
                values[interval + 1] - values[interval]
            ) * ((time - lower) / (upper - lower))
        end
        return samples
    end

    """
        _temporal_self_convergence(measure, runs)

    Temporal self-convergence of a refinement series at a fixed mesh.

    `runs` holds `(times, amplitudes)` pairs ordered from the largest to the
    smallest step, so its last entry is the finest run. The runs take
    different steps and therefore sample different times: the finest
    amplitudes are sampled linearly at the sample times of every coarse run,
    and `measure(times, coarse, finest)` returns that run's difference, the
    same normalized norm the series uses for its error against the exact
    solution. The result holds one difference per coarse run, in the order of
    `runs`.
    """
    function _temporal_self_convergence(measure, runs)
        length(runs) >= 2 || throw(
            ArgumentError("a refinement series needs at least two runs"),
        )
        finest_times, finest_amplitudes = last(runs)
        return [
            measure(
                times, amplitudes,
                _linear_sample(finest_times, finest_amplitudes, times),
            )
                for (times, amplitudes) in runs[1:(end - 1)]
        ]
    end
end

# Frozen cross-process comparison budget for short deterministic trajectories.
const ROUNDOFF_UNITS = 512

function compare_number(reference::Real, candidate::Real, path::String)
    if reference isa Integer || candidate isa Integer
        isequal(reference, candidate) ||
            error("integer mismatch at $path: $reference versus $candidate")
        return 0.0
    end
    if isequal(reference, candidate)
        return 0.0
    end
    isfinite(reference) && isfinite(candidate) ||
        error("non-finite mismatch at $path: $reference versus $candidate")
    scale = max(1.0, abs(Float64(reference)), abs(Float64(candidate)))
    tolerance = ROUNDOFF_UNITS * eps(Float64) * scale
    difference = abs(Float64(reference) - Float64(candidate))
    difference <= tolerance ||
        error("value mismatch at $path: difference $difference > $tolerance")
    return difference
end

function compare_value(reference, candidate, path::String)
    if reference isa Real && candidate isa Real
        return compare_number(reference, candidate, path)
    elseif reference isa AbstractVector && candidate isa AbstractVector
        length(reference) == length(candidate) ||
            error("length mismatch at $path")
        largest = 0.0
        for index in eachindex(reference)
            largest = max(largest, compare_value(
                reference[index], candidate[index], "$path[$index]",
            ))
        end
        return largest
    elseif reference isa AbstractDict && candidate isa AbstractDict
        keys(reference) == keys(candidate) || error("keys mismatch at $path")
        largest = 0.0
        for key in keys(reference)
            largest = max(largest, compare_value(
                reference[key], candidate[key], "$path.$key",
            ))
        end
        return largest
    end
    isequal(reference, candidate) ||
        error("discrete mismatch at $path: $reference versus $candidate")
    return 0.0
end

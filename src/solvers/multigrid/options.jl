"""
    MultigridOptions(; coarse_size=64, pre_sweeps=2, post_sweeps=2,
                     max_levels=32)

Common geometric-multigrid hierarchy and smoothing configuration.
"""
struct MultigridOptions
    coarse_size::Int
    pre_sweeps::Int
    post_sweeps::Int
    max_levels::Int
end

function MultigridOptions(;
        coarse_size = 64, pre_sweeps = 2, post_sweeps = 2,
        max_levels = 32,
    )
    values = (coarse_size, pre_sweeps, post_sweeps, max_levels)
    all(value -> value isa Integer && !(value isa Bool), values) || throw(
        ArgumentError("multigrid sizes and sweep counts must be integers"),
    )
    coarse_size > 0 || throw(
        ArgumentError("coarse_size must be positive"),
    )
    pre_sweeps >= 0 || throw(
        ArgumentError("pre_sweeps must be nonnegative"),
    )
    post_sweeps >= 0 || throw(
        ArgumentError("post_sweeps must be nonnegative"),
    )
    max_levels > 0 || throw(
        ArgumentError("max_levels must be positive"),
    )
    return MultigridOptions(
        Int(coarse_size), Int(pre_sweeps), Int(post_sweeps), Int(max_levels),
    )
end

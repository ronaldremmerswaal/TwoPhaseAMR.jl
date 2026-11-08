# Private row storage for the per-projection joint-fit scratch.
mutable struct _MimeticJointFitRow{T}
    support::Vector{Int}
    base_t::Vector{T}
    base_n::Vector{T}
    moments::Matrix{T}
    pressure::Matrix{T}
    pressure_target::Vector{T}
    tangent_values::Vector{T}
    normal_values::Vector{T}
    tangent_success::Bool
    normal_success::Bool
    geometry_fallback::Bool
end

function _MimeticJointFitRow(::Type{T}) where {T}
    return _MimeticJointFitRow{T}(
        Int[], T[], T[], Matrix{T}(undef, 2, 0), Matrix{T}(undef, 4, 0),
        zeros(T, 4), T[], T[], true, true, false,
    )
end

"""Reusable storage for the local mimetic joint fit of one projection workspace.

The buffers are private to one fit at a time. Their dimensions and contents are
refreshed for each operator construction; returned operators own their arrays.
"""
mutable struct MimeticJointFitScratch{T}
    rows::Vector{_MimeticJointFitRow{T}}
    graph_supports::Vector{Vector{Int}}
    graph_rows_at_column::Vector{Vector{Int}}
    matrix_rows_at_column::Vector{Vector{Int}}
    selected::BitSet
    frontier::Vector{Int}
    next_frontier::Vector{Int}
    positions::Vector{Int}
    weights::Vector{T}
    constant_samples::Matrix{T}
    common_pressure::Matrix{T}
    moment_target::Vector{T}
    unit_normal::Vector{T}
    ones_kinetic::Vector{T}
    overlapping::Vector{Int}
    overlap_lookup::Vector{Int}
    inverse_kinetic::Vector{T}
    overlap_buffers::Dict{Tuple{Int, Int}, Matrix{T}}
    matrix_rows::Vector{Int}
    matrix_columns::Vector{Int}
    matrix_coefficients::Vector{T}
    observation::Vector{T}
    observation_coefficients::Vector{T}
    observation_rhs::Vector{T}
    fit_matrices::Dict{Tuple{Int, Int}, Matrix{T}}
    fit_rhs::Vector{T}
    pressure_errors::Vector{T}
    empty_pressure::Matrix{T}
    empty_target::Vector{T}
end

function MimeticJointFitScratch(::Type{T}) where {T}
    return MimeticJointFitScratch{T}(
        _MimeticJointFitRow{T}[], Vector{Int}[], Vector{Int}[], Vector{Int}[],
        BitSet(), Int[], Int[], Int[], T[], zeros(T, 0, 2), zeros(T, 0, 2),
        zeros(T, 2), zeros(T, 2), T[], Int[], Int[], T[],
        Dict{Tuple{Int, Int}, Matrix{T}}(), Int[], Int[], T[], T[], T[], T[],
        Dict{Tuple{Int, Int}, Matrix{T}}(), T[], T[], zeros(T, 0, 0), T[],
    )
end

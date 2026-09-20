"""
    LinearSolverOptions(; krylov=nothing, krylov_options=(;),
                        preconditioner=:auto,
                        preconditioner_options=(;))

Configuration shared by pressure and diffusion linear solves. This object
contains only algebraic solver choices; equation-system tolerances and time
integration options belong to their respective equation system.
"""
struct LinearSolverOptions{K, KO <: NamedTuple, P, PO <: NamedTuple}
    krylov::K
    krylov_options::KO
    preconditioner::P
    preconditioner_options::PO
end

function LinearSolverOptions(;
        krylov = nothing, krylov_options = (;),
        preconditioner = :auto, preconditioner_options = (;),
    )
    krylov_options isa NamedTuple || throw(
        ArgumentError("krylov_options must be a named tuple"),
    )
    preconditioner_options isa NamedTuple || throw(
        ArgumentError("preconditioner_options must be a named tuple"),
    )
    return LinearSolverOptions(
        krylov, krylov_options, preconditioner, preconditioner_options,
    )
end

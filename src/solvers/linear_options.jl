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

function _validate_linear_solver_symbols(krylov, preconditioner)
    krylov isa Symbol && throw(
        ArgumentError(
            "krylov must be nothing, a Krylov workspace, or a workspace constructor (CG, BiCGSTAB, GMRES, FGMRES); symbols are not supported",
        )
    )
    preconditioner isa Symbol &&
        !(preconditioner in (:auto, :geometric, :tree_geometric, :amg, :ilu, :ichol)) &&
        throw(
        ArgumentError(
            "preconditioner must be :auto, :geometric, :tree_geometric, :amg, :ilu, :ichol, nothing, a factory, or a pre-built preconditioner",
        )
    )
    return nothing
end

function LinearSolverOptions(;
        krylov = nothing, krylov_options = (;),
        preconditioner = :auto, preconditioner_options = (;),
    )
    _validate_linear_solver_symbols(krylov, preconditioner)
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

module TwoPhaseAMRLimitedLDLFactorizationsExt

using LinearAlgebra: Symmetric
using LimitedLDLFactorizations
using TwoPhaseAMR

function build_incomplete_cholesky_preconditioner(
        matrix; memory=0, drop_tolerance=0,
        shift=zero(eltype(matrix)), kwargs...)
    memory >= 0 || throw(ArgumentError("memory must be nonnegative"))
    drop_tolerance >= 0 ||
        throw(ArgumentError("drop_tolerance must be nonnegative"))
    shift >= 0 || throw(ArgumentError("shift must be nonnegative"))
    return LimitedLDLFactorizations.lldl(
        Symmetric(matrix, :L); memory, droptol=drop_tolerance,
        α=shift, kwargs...)
end

end

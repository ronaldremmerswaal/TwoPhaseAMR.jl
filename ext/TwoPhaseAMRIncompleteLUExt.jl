module TwoPhaseAMRIncompleteLUExt

using IncompleteLU
using TwoPhaseAMR

function build_incomplete_lu_preconditioner(
        matrix; drop_tolerance=1e-3)
    drop_tolerance >= 0 ||
        throw(ArgumentError("drop_tolerance must be nonnegative"))
    return IncompleteLU.ilu(matrix; τ=drop_tolerance)
end

end

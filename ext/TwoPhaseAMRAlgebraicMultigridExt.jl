module TwoPhaseAMRAlgebraicMultigridExt

using LinearAlgebra: Symmetric
using AlgebraicMultigrid
using TwoPhaseAMR

function build_algebraic_multigrid_preconditioner(
        matrix; symmetric=true, singular=false,
        algorithm=:smoothed_aggregation, kwargs...)
    algorithm in (:smoothed_aggregation, :ruge_stuben) ||
        throw(ArgumentError(
            "AMG algorithm must be :smoothed_aggregation or :ruge_stuben"))
    operator = symmetric ? Symmetric(matrix) : matrix
    settings = singular && !haskey(kwargs, :coarse_solver) ?
        merge((; coarse_solver=AlgebraicMultigrid.Pinv), kwargs) : kwargs
    hierarchy = if algorithm === :smoothed_aggregation
        AlgebraicMultigrid.smoothed_aggregation(operator; settings...)
    else
        AlgebraicMultigrid.ruge_stuben(operator; settings...)
    end
    return AlgebraicMultigrid.aspreconditioner(hierarchy)
end

end

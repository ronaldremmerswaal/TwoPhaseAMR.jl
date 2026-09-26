module MetricFeasibility

using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "jump_inference_feasibility.jl"))

"""Cheap necessary tests for changing the jump kinetic metric.

For a prescribed pressure response Y, write R=-B'P. If a K-orthogonal
projection is to return Y, then Y'K(Y-K^-1 R)=0, hence Y'K Y=Y'R.
The left side is symmetric positive semidefinite for every SPD K. This
work test is independent of the normal interpolant and interface topology.
"""
function work_test(problem)
    sample = samples(problem)
    k0 = sample.k
    y = sample.yp
    r = k0 .* sample.xp
    work = transpose(y) * r
    sym_work = Symmetric((work + transpose(work)) / 2)
    asymmetry = norm(work - transpose(work)) /
        max(norm(work), eps())
    eigenvalues = eigvals(sym_work)
    return (; k0, y, r, work, asymmetry,
        minimum_eigenvalue = minimum(eigenvalues))
end

"""Illustrative bounded diagonal fit of K to the necessary work moments.

This is deliberately *not* a proposed kinetic Hodge: it tests how much
pointwise reweighting the moment equations request. The ridge fit is clipped
to the stated factor bounds, so its residual is descriptive rather than a
certified optimum over the box. No production matrix is changed.
"""
function fit_diagonal_metric(test; lower = 0.5, upper = 2.0,
        ridge_fraction = 1e-6)
    (; k0, y, r, work) = test
    m = size(y, 2)
    rows = div(m * (m + 1), 2)
    a = Matrix{Float64}(undef, rows, length(k0))
    b = Vector{Float64}(undef, rows)
    index = 0
    for j in 1:m, i in 1:j
        index += 1
        @views a[index, :] .= k0 .* y[:, i] .* y[:, j]
        b[index] = (work[i, j] + work[j, i]) / 2
        scale = max(sqrt(abs(work[i, i] * work[j, j])), eps())
        @views a[index, :] ./= scale
        b[index] /= scale
    end
    baseline = norm(a * ones(length(k0)) - b) / max(norm(b), eps())
    lambda = ridge_fraction * sum(abs2, a) / length(k0)
    theta = (transpose(a) * a + lambda * I) \
        (transpose(a) * b + lambda * ones(length(k0)))
    theta = clamp.(theta, lower, upper)
    residual = a * theta - b
    # Exact coordinate minimisation of the box-constrained convex ridge
    # problem. Clipping an unconstrained fit is not itself a box optimum.
    for _ in 1:2000
        largest_change = 0.0
        for j in eachindex(theta)
            column = view(a, :, j)
            old = theta[j]
            gradient = dot(column, residual) + lambda * (old - 1)
            curvature = dot(column, column) + lambda
            new = clamp(old - gradient / curvature, lower, upper)
            difference = new - old
            if difference != 0
                theta[j] = new
                residual .+= difference .* column
                largest_change = max(largest_change, abs(difference))
            end
        end
        largest_change < 1e-10 && break
    end
    fitted = norm(a * theta - b) / max(norm(b), eps())
    k = k0 .* theta
    x = r ./ k
    bound = maximum(projector_error_lower_bound(x, y, k))
    return (; lower, upper, baseline, fitted,
        theta_min = minimum(theta), theta_max = maximum(theta),
        bound, theta)
end

function study(; densities = ((1000.0, 1.0), (1.0, 1000.0)),
        resolutions = (16, 32, 64), boxes = ((0.5, 2.0), (0.1, 10.0)))
    println("density\tn\twork_asym\tmin_work_eigenvalue\tK_range\tmoment_residual_before\tmoment_residual_after\tsphere_RMS_bound\tscale_min\tscale_max")
    for density in densities, n in resolutions
        problem = build_problem(n; density)
        test = work_test(problem)
        for (lower, upper) in boxes
            fit = fit_diagonal_metric(test; lower, upper)
            @printf("%s\t%d\t%.6g\t%.6g\t[%g,%g]\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\n",
                string(density), n, test.asymmetry,
                test.minimum_eigenvalue, lower, upper,
                fit.baseline, fit.fitted, fit.bound,
                fit.theta_min, fit.theta_max,
            )
            flush(stdout)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    study()
end

end # module

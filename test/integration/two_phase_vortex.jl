using Test, TwoPhaseAMR

# The thesis vortex figure reports O(h) in L∞(volume fraction). PLIC geometric
# error is O(h²); volume-weighted L¹ additionally measures cancellation within
# cut cells and is reported separately from geometric error.
@testset "Classical T=1 vortex refinement study" begin
    for kind in (:uniform, :mixed, :amr)
        resolutions = kind == :uniform ? (16, 32, 64) : (8, 16, 32)
        errors = [phase_vortex(kind, n) for n in resolutions]
        for metric in (
                :geometry, :fraction, :maximum_fraction,
                :centered, :staggered,
            )
            values = [getproperty(e, metric) for e in errors]
            rates = log2.(values[1:(end - 1)] ./ values[2:end])
            println(
                "convergence mesh=$kind metric=$metric n=$resolutions " *
                    "errors=$values rates=$rates",
            )
        end
        # The thesis reports second-order symmetric-difference error for PLIC
        # reconstruction and close to second-order L1 error for the modified
        # high-order phase-scalar flux interpolant. L∞ fraction error is
        # first-order and is reported above, but is too alignment-sensitive for
        # a three-grid pass/fail estimate.
        for metric in (:geometry, :fraction, :centered, :staggered)
            values = [getproperty(e, metric) for e in errors]
            rates = log2.(values[1:(end - 1)] ./ values[2:end])
            observed_order = log2(values[1] / values[end]) /
                log2(resolutions[end] / resolutions[1])
            # The coarsest fixed-mixed grid is visibly pre-asymptotic. Require
            # both the full series and finest pair to reach that regime.
            @test observed_order >= 1.5
            @test rates[end] >= 1.5
        end
    end
end

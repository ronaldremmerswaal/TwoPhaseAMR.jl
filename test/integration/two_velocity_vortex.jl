using Test, TwoPhaseAMR

function observed_order(values, resolutions)
    return log(values[1] / values[end]) /
        log(resolutions[end] / resolutions[1])
end

@testset "Continuous two-velocity vortex reversal" begin
    for kind in (:uniform, :mixed, :amr)
        resolutions = kind == :uniform ? (16, 32, 64) : (8, 16, 32)
        results = [
            phase_vortex(kind, resolution; two_velocity = true)
                for resolution in resolutions
        ]
        for metric in (
                :geometry, :fraction, :maximum_fraction, :centered,
                :staggered, :staggered_liquid, :staggered_gas,
                :liquid_momentum, :gas_momentum,
            )
            values = [getproperty(result, metric) for result in results]
            rates = log2.(values[1:(end - 1)] ./ values[2:end])
            println(
                "two_velocity_vortex mesh=$kind metric=$metric " *
                    "n=$resolutions errors=$values rates=$rates",
            )
        end
        # The phase fields are the discontinuous staggered fields from
        # pt1_validation.tex. The thesis reports close to second order in
        # the phase-weighted L1 norm for each phase separately.
        for metric in (
                :geometry, :fraction,
                :staggered_liquid, :staggered_gas,
            )
            values = [getproperty(result, metric) for result in results]
            rates = log2.(values[1:(end - 1)] ./ values[2:end])
            @test observed_order(values, resolutions) >= 1.5
            @test rates[end] >= 1.5
        end
        gas_amount = [result.gas_amount for result in results]
        println(
            "two_velocity_vortex_conservation mesh=$kind " *
                "volume=$(getproperty.(results, :volume)) " *
                "liquid=$(getproperty.(results, :liquid_amount)) " *
                "gas=$gas_amount",
        )
        @test all(isfinite, gas_amount)
    end
end

using Test, TwoPhaseAMR

include("two_phase_vortex.jl")

function observed_order(values, resolutions)
    return log(values[1] / values[end]) /
        log(resolutions[end] / resolutions[1])
end

@testset "Continuous two-velocity vortex reversal" begin
    methods = (
        projected = ProjectedLiquidVelocity(
            tolerance = 1.0e-12, max_cycles = 500,
        ),
        frozen = FrozenInterfaceFluxes(),
    )
    for kind in (:uniform, :mixed, :amr)
        resolutions = kind == :uniform ? (16, 32, 64) : (8, 16, 32)
        for (method_name, method) in pairs(methods)
            results = [
                phase_vortex(
                    kind, resolution; interface_transport = method,
                ) for resolution in resolutions
            ]
            for metric in (
                    :geometry, :fraction, :maximum_fraction, :centered,
                    :staggered, :staggered_liquid, :staggered_gas,
                    :liquid_momentum, :gas_momentum,
                )
                values = [getproperty(result, metric) for result in results]
                rates = log2.(values[1:(end - 1)] ./ values[2:end])
                println(
                    "two_velocity_vortex mesh=$kind method=$method_name " *
                        "metric=$metric n=$resolutions errors=$values " *
                        "rates=$rates",
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
                # The projected method resets its auxiliary gas carrier after
                # each step. The thesis predicts the same asymptotic phase
                # accuracy, but this reset makes the finest gas pair approach
                # that regime more slowly in the measured vortex series.
                minimum_finest_rate =
                    method isa ProjectedLiquidVelocity &&
                    metric === :staggered_gas ? 1.2 : 1.5
                @test rates[end] >= minimum_finest_rate
            end
            gas_amount = [result.gas_amount for result in results]
            println(
                "two_velocity_vortex_conservation mesh=$kind " *
                    "method=$method_name volume=$(getproperty.(results, :volume)) " *
                    "liquid=$(getproperty.(results, :liquid_amount)) " *
                    "gas=$gas_amount",
            )
            if method isa FrozenInterfaceFluxes
                @test all(isfinite, gas_amount)
            else
                @test all(diff(gas_amount) .< 0)
                @test observed_order(gas_amount, resolutions) >= 0.9
            end
        end
    end
end

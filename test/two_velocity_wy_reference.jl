using Test

"""Independent periodic 1-D form of one directional WY phase/scalar sweep."""
function reference_wy_sweep(
        phase, scalar, indicator, carrier_flux, phase_flux;
        volumes = ones(eltype(phase), length(phase)),
    )
    n = length(phase)
    all(length(array) == n for array in (
        scalar, indicator, carrier_flux, phase_flux, volumes,
    )) || throw(DimensionMismatch("all reference arrays must have equal length"))
    amount = phase .* scalar
    next_phase = similar(phase)
    next_amount = similar(amount)
    for cell in 1:n
        left = cell == 1 ? n : cell - 1
        right = cell
        qleft, qright = carrier_flux[left], carrier_flux[right]
        fleft, fright = phase_flux[left], phase_flux[right]
        sleft = fleft >= 0 ? scalar[left] : scalar[cell]
        next = cell == n ? 1 : cell + 1
        sright = fright >= 0 ? scalar[cell] : scalar[next]
        dilation = indicator[cell] * (qright - qleft)
        next_phase[cell] = phase[cell] + (
            dilation - (fright - fleft)
        ) / volumes[cell]
        next_amount[cell] = amount[cell] + (
            indicator[cell] * scalar[cell] * (qright - qleft) -
                (fright * sright - fleft * sleft)
        ) / volumes[cell]
    end
    next_scalar = similar(scalar)
    for cell in 1:n
        next_scalar[cell] = iszero(next_phase[cell]) ? scalar[cell] :
            next_amount[cell] / next_phase[cell]
    end
    return next_phase, next_scalar, next_amount
end

"""Independent conservative auxiliary carrier used by projected Method A."""
function reference_conservative_sweep(
        phase, scalar, phase_flux;
        volumes = ones(eltype(phase), length(phase)),
    )
    n = length(phase)
    amount = phase .* scalar
    next_phase = similar(phase)
    next_amount = similar(amount)
    for cell in 1:n
        left = cell == 1 ? n : cell - 1
        right = cell
        fleft, fright = phase_flux[left], phase_flux[right]
        next = cell == n ? 1 : cell + 1
        sleft = fleft >= 0 ? scalar[left] : scalar[cell]
        sright = fright >= 0 ? scalar[cell] : scalar[next]
        next_phase[cell] = phase[cell] -
            (fright - fleft) / volumes[cell]
        next_amount[cell] = amount[cell] -
            (fright * sright - fleft * sleft) / volumes[cell]
    end
    next_scalar = similar(scalar)
    for cell in 1:n
        next_scalar[cell] = iszero(next_phase[cell]) ? scalar[cell] :
            next_amount[cell] / next_phase[cell]
    end
    return next_phase, next_scalar, next_amount
end

@testset "Two-velocity directional WY reference equations" begin
    liquid = [0.0, 0.2, 0.8, 1.0, 0.65, 0.1]
    gas = 1 .- liquid
    indicator = Float64.(liquid .> 0.5)
    gas_indicator = 1 .- indicator

    # These signed arrays stand for geometrically integrated partial donating
    # regions. Their partition is the only assumption used by the balance.
    carrier = [0.015, -0.008, 0.012, 0.004, -0.006, -0.003]
    liquid_flux = [0.0, -0.002, 0.009, 0.004, -0.004, 0.0]
    gas_flux = carrier .- liquid_flux

    new_liquid, liquid_one, liquid_amount = reference_wy_sweep(
        liquid, ones(6), indicator, carrier, liquid_flux,
    )
    new_gas, gas_one, gas_amount = reference_wy_sweep(
        gas, ones(6), gas_indicator, carrier, gas_flux,
    )
    @test new_liquid + new_gas ≈ ones(6) atol = 8eps()
    @test liquid_one ≈ ones(6) atol = 8eps()
    @test gas_one ≈ ones(6) atol = 8eps()
    @test liquid_amount == new_liquid
    @test gas_amount == new_gas
    # A non-solenoidal directional carrier can exchange the two phase volumes
    # through WY dilation, but their sum remains the open volume. With a
    # discretely solenoidal (constant, in 1-D) carrier, each phase is globally
    # conservative as well.
    solenoidal_carrier = fill(0.01, 6)
    solenoidal_liquid_flux = [0.0, 0.002, 0.009, 0.01, 0.006, 0.0]
    solenoidal_gas_flux = solenoidal_carrier .- solenoidal_liquid_flux
    conservative_liquid = reference_wy_sweep(
        liquid, ones(6), indicator, solenoidal_carrier,
        solenoidal_liquid_flux,
    )[1]
    conservative_gas = reference_wy_sweep(
        gas, ones(6), gas_indicator, solenoidal_carrier,
        solenoidal_gas_flux,
    )[1]
    @test sum(conservative_liquid) ≈ sum(liquid) atol = 8eps()
    @test sum(conservative_gas) ≈ sum(gas) atol = 8eps()

    liquid_scalar = 2 .+ 0.2 .* (1:6)
    gas_scalar = 5 .- 0.1 .* (1:6)
    _, transported_liquid, _ = reference_wy_sweep(
        liquid, liquid_scalar, indicator, carrier, liquid_flux,
    )
    _, transported_gas, _ = reference_wy_sweep(
        gas, gas_scalar, gas_indicator, carrier, gas_flux,
    )
    @test all(isfinite, transported_liquid)
    @test all(isfinite, transported_gas)

    # The projected method's gas carrier is conservative and deliberately
    # need not equal the complement of the physical liquid fraction.
    auxiliary_gas, auxiliary_one, auxiliary_amount =
        reference_conservative_sweep(gas, ones(6), gas_flux)
    @test sum(auxiliary_gas) ≈ sum(gas) atol = 8eps()
    @test auxiliary_one ≈ ones(6) atol = 8eps()
    @test auxiliary_amount == auxiliary_gas
    @test maximum(abs, auxiliary_gas - new_gas) > 0

    # Equal-velocity reduction: splitting a common carrier by phase and using
    # the same WY equation is exactly the ordinary one-velocity algebra.
    common_liquid = liquid_flux
    common_gas = carrier .- common_liquid
    reduced_liquid = reference_wy_sweep(
        liquid, ones(6), indicator, carrier, common_liquid,
    )[1]
    reduced_gas = reference_wy_sweep(
        gas, ones(6), gas_indicator, carrier, common_gas,
    )[1]
    @test reduced_liquid == new_liquid
    @test reduced_gas == new_gas
end

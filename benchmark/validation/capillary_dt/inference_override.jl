# In-session copy of the CMDGFM jump-row assembly with a switch for the
# inference term. src/ is NOT modified; this file redefines two methods in the
# loaded module for the experiment only.
#   :baseline      identical to src/two_phase/projection/cmdgfm_projection.jl:721-853
#   :same_location V3 of the #171 study (cmdgfm_jump/FINDINGS.md). The inference
#                  term (β-1)(u_g(h_g) - u_l(h_l)) compares gas and liquid
#                  values 2h apart; V3 replaces it by the jumps at those donor
#                  faces: (β-1)(γ_lg [u]_{h_g} + γ_gl [u]_{h_l}), which vanishes
#                  for a continuous field. Same ξ columns (β-1)γ; the
#                  (β-1)(Gp(h_g) - Gp(h_l)) pressure columns are dropped.
@eval TwoPhaseAMR begin
    const STUDY178_MODE = Ref(:baseline)

    @inline function _cmdgfm_normal_velocity_row(velocity::TwoPhaseVelocity, stencil)
        T = eltype(velocity[1])
        liquid, gas = velocity
        face = stencil.face
        value = @inbounds gas.interior[face] - liquid.interior[face]
        stencil.resolved || return value
        factor = stencil.beta * stencil.ratio
        for (candidate, weight) in zip(stencil.transverse_faces, stencil.transverse_weights)
            @inbounds value += factor * weight *
                (gas.interior[candidate] - liquid.interior[candidate])
        end
        if stencil.gas_inference_face > 0
            hg, hl = stencil.gas_inference_face, stencil.liquid_inference_face
            if STUDY178_MODE[] === :same_location
                # γ from the current phase masses at the donors.
                @inbounds value += (stencil.beta - one(T)) * (
                    _phase_product(_STUDY178_GAMMA[hg][1], gas.interior[hg] - liquid.interior[hg]) +
                        _phase_product(_STUDY178_GAMMA[hl][2], gas.interior[hl] - liquid.interior[hl])
                )
            else
                @inbounds value += (stencil.beta - one(T)) * (
                    gas.interior[hg] - liquid.interior[hl]
                )
            end
        end
        return value
    end

    # γ_lg / γ_gl per face, refreshed by every assembly (needed by the row check).
    const _STUDY178_GAMMA = Dict{Int, Tuple{Float64, Float64}}()

    function _cmdgfm_jump_blocks(
            stencils, mixed_faces, data, common_gradient, pressure_indices,
            velocity::TwoPhaseVelocity, dt, ::Type{T}; pressure_lookup = nothing,
            storage = nothing,
        ) where {T}
        mode = STUDY178_MODE[]
        empty!(_STUDY178_GAMMA)
        jump_index = if isnothing(storage)
            Dict(face => index for (index, face) in pairs(mixed_faces))
        else
            resize!(storage.jump_lookup, length(data.liquid_fraction))
            fill!(storage.jump_lookup, 0)
            for (index, face) in pairs(mixed_faces)
                storage.jump_lookup[face] = index
            end
            storage.jump_lookup
        end
        rows, columns, values = isnothing(storage) ? (Int[], Int[], T[]) :
            _cmdgfm_clear_triplets!(storage.jump_triplets)
        pressure_rows, pressure_columns, pressure_values = isnothing(storage) ?
            (Int[], Int[], T[]) : _cmdgfm_clear_triplets!(storage.pressure_triplets)
        right_hand_side = isnothing(storage) ? zeros(T, length(stencils)) : storage.jump_rhs
        resize!(right_hand_side, length(stencils))
        underresolved = 0
        liquid = velocity[1].interior
        gas = velocity[2].interior
        isnothing(pressure_lookup) && (
            pressure_lookup = Dict(
                cell => local_index for (local_index, cell) in pairs(pressure_indices)
            )
        )
        gradient_rows = _cmdgfm_gradient_rows(common_gradient)
        for (row, stencil) in pairs(stencils)
            face = stencil.face
            push!(rows, row)
            push!(columns, jump_index[face])
            push!(values, one(T))
            if !stencil.resolved
                underresolved += 1
                @inbounds right_hand_side[row] = (gas[face] - liquid[face]) / T(dt)
                continue
            end
            rhs = @inbounds(gas[face] - liquid[face]) / T(dt)
            transverse_factor = stencil.beta * stencil.ratio
            for (candidate, weight) in zip(
                    stencil.transverse_faces, stencil.transverse_weights,
                )
                coefficient = transverse_factor * weight
                local_index = _cmdgfm_pressure_index(jump_index, candidate)
                iszero(local_index) || begin
                    push!(rows, row)
                    push!(columns, local_index)
                    push!(values, coefficient)
                end
                @inbounds rhs += coefficient * (gas[candidate] - liquid[candidate]) / T(dt)
            end
            if stencil.gas_inference_face > 0
                gas_face = stencil.gas_inference_face
                liquid_face = stencil.liquid_inference_face
                inference_factor = stencil.beta - one(T)
                gamma_liquid_gas = data.liquid_mass[gas_face] / data.total_mass[gas_face]
                gamma_gas_liquid = data.gas_mass[liquid_face] / data.total_mass[liquid_face]
                _STUDY178_GAMMA[gas_face] = (gamma_liquid_gas, get(_STUDY178_GAMMA, gas_face, (0.0, 0.0))[2])
                _STUDY178_GAMMA[liquid_face] = (get(_STUDY178_GAMMA, liquid_face, (0.0, 0.0))[1], gamma_gas_liquid)
                for (candidate, gamma) in (
                        (gas_face, gamma_liquid_gas),
                        (liquid_face, gamma_gas_liquid),
                    )
                    local_index = _cmdgfm_pressure_index(jump_index, candidate)
                    iszero(local_index) || begin
                        push!(rows, row)
                        push!(columns, local_index)
                        push!(values, inference_factor * gamma)
                    end
                end
                if mode === :same_location
                    @inbounds rhs += inference_factor * (
                        _phase_product(gamma_liquid_gas, gas[gas_face] - liquid[gas_face]) +
                            _phase_product(gamma_gas_liquid, gas[liquid_face] - liquid[liquid_face])
                    ) / T(dt)
                else
                    for (gradient_face, sign) in ((gas_face, one(T)), (liquid_face, -one(T)))
                        _cmdgfm_append_gradient_row!(
                            pressure_rows, pressure_columns,
                            pressure_values, gradient_rows, pressure_lookup, row,
                            gradient_face, inference_factor * sign
                        )
                    end
                    @inbounds rhs += inference_factor *
                        (gas[gas_face] - liquid[liquid_face]) / T(dt)
                end
            end
            @inbounds right_hand_side[row] = rhs
        end
        jump_block, pressure_block = if isnothing(storage)
            (
                sparse(rows, columns, values, length(stencils), length(mixed_faces)),
                sparse(
                    pressure_rows, pressure_columns, pressure_values,
                    length(stencils), length(pressure_indices)
                ),
            )
        else
            storage.jump_block = _cmdgfm_sorted_rows_csc!(
                storage.jump_block,
                storage.jump_triplets, length(stencils), length(mixed_faces)
            )
            storage.jump_pressure = _cmdgfm_sorted_rows_csc!(
                storage.jump_pressure,
                storage.pressure_triplets, length(stencils), length(pressure_indices)
            )
            (storage.jump_block, storage.jump_pressure)
        end
        return pressure_block, jump_block, right_hand_side, underresolved
    end
end

function _staggered_wave_rhs!(
        displacement_rate, velocity_rate,
        displacement, velocity, operator
    )
    copyto!(displacement_rate.interior, velocity.interior)
    copyto!(displacement_rate.boundary, velocity.boundary)
    diffusion!(velocity_rate, displacement, operator)
    return nothing
end

function _staggered_wave_stage!(stage, state, scale, rate)
    for index in eachindex(stage.interior)
        @inbounds stage.interior[index] =
            state.interior[index] + scale * rate.interior[index]
    end
    for index in eachindex(stage.boundary)
        @inbounds stage.boundary[index] =
            state.boundary[index] + scale * rate.boundary[index]
    end
    return stage
end

function _staggered_wave_rk4_step!(
        displacement, velocity, dt, operator,
        stage_displacement, stage_velocity,
        displacement_rates, velocity_rates
    )
    _staggered_wave_rhs!(
        displacement_rates[1], velocity_rates[1],
        displacement, velocity, operator
    )

    _staggered_wave_stage!(
        stage_displacement, displacement, dt / 2, displacement_rates[1]
    )
    _staggered_wave_stage!(
        stage_velocity, velocity, dt / 2, velocity_rates[1]
    )
    _staggered_wave_rhs!(
        displacement_rates[2], velocity_rates[2],
        stage_displacement, stage_velocity, operator
    )

    _staggered_wave_stage!(
        stage_displacement, displacement, dt / 2, displacement_rates[2]
    )
    _staggered_wave_stage!(
        stage_velocity, velocity, dt / 2, velocity_rates[2]
    )
    _staggered_wave_rhs!(
        displacement_rates[3], velocity_rates[3],
        stage_displacement, stage_velocity, operator
    )

    _staggered_wave_stage!(
        stage_displacement, displacement, dt, displacement_rates[3]
    )
    _staggered_wave_stage!(
        stage_velocity, velocity, dt, velocity_rates[3]
    )
    _staggered_wave_rhs!(
        displacement_rates[4], velocity_rates[4],
        stage_displacement, stage_velocity, operator
    )

    for index in eachindex(displacement.interior)
        @inbounds begin
            displacement.interior[index] += dt * (
                displacement_rates[1].interior[index] +
                    2displacement_rates[2].interior[index] +
                    2displacement_rates[3].interior[index] +
                    displacement_rates[4].interior[index]
            ) / 6
            velocity.interior[index] += dt * (
                velocity_rates[1].interior[index] +
                    2velocity_rates[2].interior[index] +
                    2velocity_rates[3].interior[index] +
                    velocity_rates[4].interior[index]
            ) / 6
        end
    end
    for index in eachindex(displacement.boundary)
        @inbounds begin
            displacement.boundary[index] += dt * (
                displacement_rates[1].boundary[index] +
                    2displacement_rates[2].boundary[index] +
                    2displacement_rates[3].boundary[index] +
                    displacement_rates[4].boundary[index]
            ) / 6
            velocity.boundary[index] += dt * (
                velocity_rates[1].boundary[index] +
                    2velocity_rates[2].boundary[index] +
                    2velocity_rates[3].boundary[index] +
                    velocity_rates[4].boundary[index]
            ) / 6
        end
    end
    return nothing
end

function _manufactured_staggered_wave_error(
        block_size; mixed = false,
        final_time = 0.1
    )
    mesh = mixed ? BlockTreeMesh(
            block_size = (block_size, block_size), max_level = 2,
            refine = (key, origin, extent) ->
            key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
        ) : BlockTreeMesh(block_size = (block_size, block_size))

    wave_shape = (x, y, axis) -> axis == 1 ?
        sinpi(2 * (x + 0.13)) * cospi(2 * (y - 0.07)) :
        cospi(2 * (x + 0.13)) * sinpi(2 * (y - 0.07))
    displacement = FaceVelocity(
        mesh, (x, y, time, axis) -> wave_shape(x, y, axis)
    )
    velocity = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    operator = StaggeredDiffusionOperator(
        mesh; viscosity = 1.0, boundary = FreeSlip()
    )

    zero_face_field() =
        FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    stage_displacement = zero_face_field()
    stage_velocity = zero_face_field()
    displacement_rates = ntuple(_ -> zero_face_field(), 4)
    velocity_rates = ntuple(_ -> zero_face_field(), 4)

    minimum_spacing = minimum(minimum(block.spacing) for block in mesh.blocks)
    number_of_steps = ceil(Int, final_time / (0.1 * minimum_spacing))
    dt = final_time / number_of_steps
    for _ in 1:number_of_steps
        _staggered_wave_rk4_step!(
            displacement, velocity, dt, operator,
            stage_displacement, stage_velocity,
            displacement_rates, velocity_rates
        )
    end

    frequency = 2pi * sqrt(2)
    displacement_amplitude = cos(frequency * final_time)
    velocity_amplitude = -frequency * sin(frequency * final_time)
    displacement_squared_error = 0.0
    displacement_maximum_error = 0.0
    velocity_squared_error = 0.0
    velocity_maximum_error = 0.0
    total_volume = 0.0
    for component in (operator.x, operator.y)
        for local_index in eachindex(component.local_to_global)
            global_index = component.local_to_global[local_index]
            face = mesh.faces[global_index]
            shape = wave_shape(face.center[1], face.center[2], face.axis)
            displacement_error = abs(
                displacement.interior[global_index] -
                    displacement_amplitude * shape
            )
            velocity_error = abs(
                velocity.interior[global_index] - velocity_amplitude * shape
            )
            volume = component.volumes[local_index]
            displacement_squared_error += volume * displacement_error^2
            velocity_squared_error += volume * velocity_error^2
            displacement_maximum_error =
                max(displacement_maximum_error, displacement_error)
            velocity_maximum_error = max(velocity_maximum_error, velocity_error)
            total_volume += volume
        end
    end

    return (;
        displacement_l2 = sqrt(displacement_squared_error / total_volume),
        displacement_linf = displacement_maximum_error,
        velocity_l2 = sqrt(velocity_squared_error / total_volume),
        velocity_linf = velocity_maximum_error,
        levels = Tuple(sort!(unique(block.key.level for block in mesh.blocks))),
        has_refinement_interface = any(
            face -> mesh.blocks[face.minus.block].key.level !=
                mesh.blocks[face.plus.block].key.level,
            mesh.faces
        ),
        dt,
        number_of_steps,
    )
end

@testset "manufactured staggered wave equation" begin
    for mixed in (false, true)
        results = [
            _manufactured_staggered_wave_error(n; mixed)
                for n in (8, 16, 32)
        ]
        expected_levels = mixed ? (1, 2) : (0,)
        @test all(result -> result.levels == expected_levels, results)
        @test all(result -> result.has_refinement_interface == mixed, results)

        for field in (:displacement, :velocity)
            l2_errors = collect(getproperty.(results, Symbol(field, :_l2)))
            linf_errors = collect(getproperty.(results, Symbol(field, :_linf)))
            l2_rates = log2.(l2_errors[1:2] ./ l2_errors[2:3])
            linf_rates = log2.(linf_errors[1:2] ./ linf_errors[2:3])

            @test all(diff(l2_errors) .< 0)
            @test all(diff(linf_errors) .< 0)
            @test minimum(l2_rates) > (mixed ? 1.5 : 1.9)
            @test minimum(linf_rates) > (mixed ? 0.9 : 1.9)
        end
    end
end

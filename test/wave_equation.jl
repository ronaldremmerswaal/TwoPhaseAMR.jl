function _wave_cell_axpy!(destination::CellField, scale, increment::CellField)
    nx, ny = destination.mesh.block_size
    for block in eachindex(destination.blocks), j in 1:ny, i in 1:nx
        destination_ref = TwoPhaseAMR.CellRef(block, i, j)
        TwoPhaseAMR._set!(
            destination,
            TwoPhaseAMR._get(destination, destination_ref) +
                scale * TwoPhaseAMR._get(increment, destination_ref),
            destination_ref,
        )
    end
    return destination
end

function _wave_face_axpy!(
        destination::FaceVelocity, scale,
        increment::FaceVelocity
    )
    for index in eachindex(destination.interior)
        @inbounds destination.interior[index] +=
            scale * increment.interior[index]
    end
    for index in eachindex(destination.boundary)
        @inbounds destination.boundary[index] +=
            scale * increment.boundary[index]
    end
    return destination
end

function _stormer_verlet_wave_step!(
        pressure, velocity, dt,
        gradient_operator,
        divergence_work, gradient_work
    )
    # Kick-drift-kick form for p_t = -D*u, u_t = -G*p. Unlike explicit
    # two-stage RK2, this has a nonempty stability interval on the imaginary
    # axis and keeps the Hamiltonian error bounded over long integrations.
    gradient!(gradient_work, pressure, gradient_operator)
    _wave_face_axpy!(velocity, -dt / 2, gradient_work)
    divergence!(divergence_work, velocity)
    _wave_cell_axpy!(pressure, -dt, divergence_work)
    gradient!(gradient_work, pressure, gradient_operator)
    _wave_face_axpy!(velocity, -dt / 2, gradient_work)
    return nothing
end

function _manufactured_wave_error(block_size; final_time = 0.1)
    mesh = BlockTreeMesh(
        block_size = (block_size, block_size),
        max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
    )
    wave_shape = (x, y) ->
    sinpi(2 * (x + 0.13)) * cospi(2 * (y - 0.07))
    pressure = CellField(mesh)
    initialize!(pressure, wave_shape)
    velocity = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)

    divergence_work = CellField(mesh)
    gradient_work = copy(velocity)
    gradient_operator = GradientOperator(pressure)

    minimum_spacing = minimum(minimum(block.spacing) for block in mesh.blocks)
    number_of_steps = ceil(Int, final_time / (0.1 * minimum_spacing))
    dt = final_time / number_of_steps
    for _ in 1:number_of_steps
        _stormer_verlet_wave_step!(
            pressure, velocity, dt, gradient_operator,
            divergence_work, gradient_work,
        )
    end

    frequency = 2pi * sqrt(2)
    amplitude = cos(frequency * final_time)
    pressure_squared_error = 0.0
    pressure_maximum_error = 0.0
    total_volume = 0.0
    nx, ny = mesh.block_size
    for (block_index, block) in pairs(mesh.blocks), j in 1:ny, i in 1:nx
        x, y = cellcenter(mesh, block_index, i, j)
        exact = amplitude * wave_shape(x, y)
        reference = TwoPhaseAMR.CellRef(block_index, i, j)
        error = abs(TwoPhaseAMR._get(pressure, reference) - exact)
        pressure_squared_error += block.volume * error^2
        pressure_maximum_error = max(pressure_maximum_error, error)
        total_volume += block.volume
    end

    velocity_squared_error = 0.0
    velocity_maximum_error = 0.0
    velocity_amplitude = sin(frequency * final_time) / sqrt(2)
    for (index, face) in pairs(mesh.faces)
        x, y = face.center
        exact = face.axis == 1 ?
            -velocity_amplitude * cospi(2 * (x + 0.13)) *
            cospi(2 * (y - 0.07)) :
            velocity_amplitude * sinpi(2 * (x + 0.13)) *
            sinpi(2 * (y - 0.07))
        error = abs(velocity.interior[index] - exact)
        minus = cellcenter(mesh, face.minus.block, face.minus.i, face.minus.j)
        plus = cellcenter(mesh, face.plus.block, face.plus.i, face.plus.j)
        normal_distance = plus[face.axis] - minus[face.axis]
        normal_distance -= round(normal_distance / mesh.extent[face.axis]) *
            mesh.extent[face.axis]
        dual_volume = face.area * abs(normal_distance)
        velocity_squared_error += dual_volume * error^2
        velocity_maximum_error = max(velocity_maximum_error, error)
    end

    return (;
        pressure_l2 = sqrt(pressure_squared_error / total_volume),
        pressure_linf = pressure_maximum_error,
        velocity_l2 = sqrt(velocity_squared_error / total_volume),
        velocity_linf = velocity_maximum_error,
        levels = Tuple(sort!(unique(block.key.level for block in mesh.blocks))),
        dt,
        number_of_steps,
    )
end

@testset "mixed-level manufactured wave equation" begin
    results = _manufactured_wave_error.((8, 16, 32))
    @test all(result -> result.levels == (1, 2), results)
    for field in (:pressure, :velocity)
        l2_errors = collect(getproperty.(results, Symbol(field, :_l2)))
        linf_errors = collect(getproperty.(results, Symbol(field, :_linf)))
        l2_rates = log2.(l2_errors[1:2] ./ l2_errors[2:3])
        linf_rates = log2.(linf_errors[1:2] ./ linf_errors[2:3])

        @test all(diff(l2_errors) .< 0)
        @test all(diff(linf_errors) .< 0)
        @test minimum(l2_rates) > 1.5
        # The largest velocity-gradient error is localized at coarse--fine
        # faces, so its maximum norm is first-order although its L2 norm is
        # higher order. Pressure is smoother in both norms.
        minimum_linf_rate = field == :pressure ? 1.5 : 1.0
        @test minimum(linf_rates) > minimum_linf_rate
    end
end

using Statistics: mean

function anisotropic_interface_errors(n)
    mesh = BlockTreeMesh(
        base_mesh = (2n, n), block_size = (8, 8),
        periodic = (true, true),
    )
    centre = (0.5, 0.5)
    radius = 0.22
    interface = InterfaceField(
        mesh,
        (x, y) -> hypot(x - centre[1], y - centre[2]) - radius;
        lipschitz = 1,
    )
    normal_errors = Float64[]
    curvature_errors = Float64[]
    for ref in TwoPhaseAMR._cell_references(mesh)
        fraction = TwoPhaseAMR._get(interface.fraction, ref)
        0 < fraction < 1 || continue
        block = mesh.blocks[ref.block]
        mx = TwoPhaseAMR._get(interface.normal_x, ref)
        my = TwoPhaseAMR._get(interface.normal_y, ref)
        intercept = TwoPhaseAMR._get(interface.intercept, ref)
        xi, eta = TwoPhaseAMR._interface_centroid(mx, my, intercept)
        x, y = cellcenter(mesh, ref.block, ref.i, ref.j)
        x += xi * block.spacing[1]
        y += eta * block.spacing[2]
        exact_x, exact_y = x - centre[1], y - centre[2]
        exact_scale = hypot(exact_x, exact_y)
        exact_x /= exact_scale
        exact_y /= exact_scale
        numerical_x = mx / block.spacing[1]
        numerical_y = my / block.spacing[2]
        numerical_scale = hypot(numerical_x, numerical_y)
        numerical_x /= numerical_scale
        numerical_y /= numerical_scale
        push!(
            normal_errors,
            hypot(numerical_x - exact_x, numerical_y - exact_y),
        )
        curvature = TwoPhaseAMR._get(interface.curvature, ref)
        isfinite(curvature) &&
            push!(curvature_errors, abs(curvature - inv(radius)))
    end
    return mean(normal_errors), mean(curvature_errors)
end

function anisotropic_scalar_advection_error(n)
    mesh = BlockTreeMesh(
        base_mesh = (2n, n), block_size = (8, 8),
        periodic = (true, true),
    )
    velocity_components = (0.35, -0.2)
    final_time = 0.2
    initial = (x, y) -> 1 + 0.2sinpi(2x) * cospi(2y)
    scalar = CellField(mesh, initial)
    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> velocity_components[axis],
    )
    workspace = AdvectionWorkspace(scalar)
    time = 0.0
    while time < final_time
        dt = min(
            stable_timestep(velocity, workspace; cfl = 0.4),
            final_time - time,
        )
        advect!(scalar, velocity, dt, workspace)
        time += dt
    end
    squared_error = total_volume = 0.0
    nx, ny = mesh.block_size
    for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        x, y = cellcenter(mesh, block, i, j)
        exact = initial(
            mod(x - velocity_components[1] * final_time, 1),
            mod(y - velocity_components[2] * final_time, 1),
        )
        volume = mesh.blocks[block].volume
        squared_error += volume * (interior(scalar, block)[i, j] - exact)^2
        total_volume += volume
    end
    return sqrt(squared_error / total_volume)
end

@testset "anisotropic reconstruction refinement" begin
    resolutions = (32, 64, 128)
    errors = anisotropic_interface_errors.(resolutions)
    normal_errors = collect(first.(errors))
    curvature_errors = collect(last.(errors))
    normal_order = log2(normal_errors[1] / normal_errors[end]) / 2
    curvature_order = log2(curvature_errors[1] / curvature_errors[end]) / 2
    @test all(diff(normal_errors) .< 0)
    @test all(diff(curvature_errors) .< 0)
    @test normal_order > 0.9
    @test curvature_order > 1.5
end

@testset "anisotropic centred advection refinement" begin
    errors = collect(anisotropic_scalar_advection_error.((8, 16, 32)))
    rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
    @test all(diff(errors) .< 0)
    @test minimum(rates) > 0.9
end

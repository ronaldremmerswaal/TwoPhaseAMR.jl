using Test
using TwoPhaseAMR
include("MainCellLW.jl")
using .MainCellLW

function lw_reference_step!(q, first, second, u, v, dt)
    nx, ny = size(q)
    flux(a, b, speed, ratio) = speed * ((a + b) * 0.5) -
        ((speed * speed) * ratio * 0.5) * (b - a)
    rx, ry = dt * nx * 0.5, dt * ny
    for j in 1:ny, i in 1:nx
        im, ip = mod1(i - 1, nx), mod1(i + 1, nx)
        first[i, j] = q[i, j] - rx * (
            flux(q[i, j], q[ip, j], u[i, j], rx) -
            flux(q[im, j], q[i, j], u[im, j], rx))
    end
    for j in 1:ny, i in 1:nx
        jm, jp = mod1(j - 1, ny), mod1(j + 1, ny)
        second[i, j] = first[i, j] - ry * (
            flux(first[i, j], first[i, jp], v[i, j], ry) -
            flux(first[i, jm], first[i, j], v[i, jm], ry))
    end
    for j in 1:ny, i in 1:nx
        im, ip = mod1(i - 1, nx), mod1(i + 1, nx)
        q[i, j] = second[i, j] - rx * (
            flux(second[i, j], second[ip, j], u[i, j], rx) -
            flux(second[im, j], second[i, j], u[im, j], rx))
    end
    q
end

function lw_scatter!(field, q)
    nx, ny = size(q)
    bx, by = field.mesh.block_size
    for (b, block) in pairs(field.mesh.blocks)
        x0 = round(Int, block.origin[1] * nx)
        y0 = round(Int, block.origin[2] * ny)
        vals = interior(field, b)
        for j in 1:by, i in 1:bx
            vals[i, j] = q[x0 + i, y0 + j]
        end
    end
    field
end

function lw_gather(field, nx, ny)
    q = Matrix{Float64}(undef, nx, ny)
    bx, by = field.mesh.block_size
    for (b, block) in pairs(field.mesh.blocks)
        x0 = round(Int, block.origin[1] * nx)
        y0 = round(Int, block.origin[2] * ny)
        vals = interior(field, b)
        for j in 1:by, i in 1:bx
            q[x0 + i, y0 + j] = vals[i, j]
        end
    end
    q
end

@testset "uniform cell-centred LW" begin
    for ((nx, ny), block_size) in (((24, 16), (24, 16)),
                                    ((24, 16), (8, 8)),
                                    ((40, 24), (8, 8)),
                                    ((32, 18), (8, 6)))
        mesh = BlockTreeMesh(base_mesh=(nx, ny), block_size=block_size,
                             periodic=(true, true))
        field = CellField(mesh, Float64)
        velocity = FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? 0.7 : -0.3)
        workspace = UniformCellLWWorkspace(field, velocity; tile_columns=3,
                                            threshold=0)
        q0 = [1.0 + 0.2 * sinpi(2 * (i - 0.5) / nx) +
                    0.1 * cospi(4 * (j - 0.5) / ny) +
                    0.03 * sinpi(2 * (i - 0.5) / nx) *
                        cospi(2 * (j - 0.5) / ny)
              for i in 1:nx, j in 1:ny]
        lw_scatter!(field, q0)
        reference = copy(q0)
        first, second = similar(q0), similar(q0)
        dt = 0.2 / max(0.7nx, 0.3ny)
        initial_mass = integrate(field)
        for _ in 1:7
            lw_reference_step!(reference, first, second,
                fill(0.7, nx, ny), fill(-0.3, nx, ny), dt)
            advect!(field, velocity, dt, workspace)
        end
        @test lw_gather(field, nx, ny) ≈ reference atol=3e-15 rtol=3e-15
        @test integrate(field) ≈ initial_mass atol=3e-15 rtol=3e-15
        fill!(reference, 2.0)
        lw_scatter!(field, reference)
        advect!(field, velocity, dt, workspace)
        @test all(==(2.0), lw_gather(field, nx, ny))
        set_uniform_lw_velocity!(workspace, velocity, -0.2, 0.5)
        advect!(field, velocity, dt, workspace)
        @test all(==(2.0), lw_gather(field, nx, ny))
        @test_throws ArgumentError advect!(field, velocity, 10.0, workspace)
        other = UniformCellLWWorkspace(field, velocity)
        @test other.first !== workspace.first
    end
    mesh = BlockTreeMesh(base_mesh=(8, 8), block_size=(8, 8),
                         periodic=(false, true))
    q = CellField(mesh, Float64)
    v = FaceVelocity(mesh, Float64)
    @test_throws ArgumentError UniformCellLWWorkspace(q, v)
end

@testset "uniform LW order, lifecycle, and independent workspaces" begin
    errors = Float64[]
    for n in (16, 32, 64)
        mesh = BlockTreeMesh(base_mesh=(n, n), block_size=(8, 8))
        field = CellField(mesh, Float64)
        velocity = FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? 0.7 : -0.3)
        workspace = UniformCellLWWorkspace(field, velocity)
        average(i, j, time) = begin
            x = (i - 0.5) / n - 0.7time
            y = (j - 0.5) / n + 0.3time
            1.0 + 0.2sinc(1 / n) * sinpi(2x) +
                0.1sinc(2 / n) * cospi(4y) +
                0.15sinc(1 / n) * sinc(1 / n) * sinpi(2(x + y))
        end
        initial = [average(i, j, 0.0) for i in 1:n, j in 1:n]
        lw_scatter!(field, initial)
        steps = ceil(Int, 0.1 / (0.2 / (0.7n)))
        dt = 0.1 / steps
        for _ in 1:steps
            advect!(field, velocity, dt, workspace)
        end
        result = lw_gather(field, n, n)
        exact = [average(i, j, 0.1) for i in 1:n, j in 1:n]
        push!(errors, sqrt(sum(abs2, result .- exact) / n^2))
    end
    @test log2(errors[1] / errors[2]) > 1.8
    @test log2(errors[2] / errors[3]) > 1.8

    mesh = BlockTreeMesh(base_mesh=(16, 16), block_size=(8, 8))
    q1, q2 = CellField(mesh, Float64), CellField(mesh, Float64)
    v1 = FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? 0.7 : -0.3)
    v2 = copy(v1)
    w1, w2 = UniformCellLWWorkspace(q1, v1), UniformCellLWWorkspace(q2, v2)
    initial = [sinpi(2i / 16) + cospi(2j / 16) for i in 1:16, j in 1:16]
    lw_scatter!(q1, initial)
    lw_scatter!(q2, initial)
    @sync begin
        Threads.@spawn advect!(q1, v1, 0.01, w1)
        Threads.@spawn advect!(q2, v2, 0.01, w2)
    end
    @test lw_gather(q1, 16, 16) == lw_gather(q2, 16, 16)
    @test_throws ArgumentError advect!(q1, v2, 0.01, w1)
    TwoPhaseAMR._invalidate_mesh_topology_cache!(mesh)
    @test_throws ArgumentError advect!(q1, v1, 0.01, w1)

    mixed = BlockTreeMesh(base_mesh=(16, 8), block_size=(8, 8), max_level=1,
                          refine=(key, origin, extent) -> key.i == 0 && key.j == 0)
    mixed_q = CellField(mixed, Float64)
    mixed_v = FaceVelocity(mixed, Float64)
    @test_throws ArgumentError UniformCellLWWorkspace(mixed_q, mixed_v)

    refined = BlockTreeMesh(base_mesh=(8, 8), block_size=(8, 8),
                            max_level=1, refine=(key, origin, extent) -> true)
    refined_q = CellField(refined, Float64)
    refined_v = FaceVelocity(refined,
        (x, y, t, axis) -> axis == 1 ? 0.7 : -0.3)
    refined_w = UniformCellLWWorkspace(refined_q, refined_v)
    refined_initial = [sinpi(2i / 16) + cospi(2j / 16)
                       for i in 1:16, j in 1:16]
    lw_scatter!(refined_q, refined_initial)
    refined_reference = copy(refined_initial)
    lw_reference_step!(refined_reference, similar(refined_initial),
        similar(refined_initial), fill(0.7, 16, 16), fill(-0.3, 16, 16), 0.01)
    advect!(refined_q, refined_v, 0.01, refined_w)
    @test lw_gather(refined_q, 16, 16) ≈ refined_reference atol=2e-15 rtol=2e-15
end

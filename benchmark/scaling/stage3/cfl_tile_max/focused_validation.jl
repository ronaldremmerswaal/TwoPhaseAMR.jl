"""Focused exact CFL checks shared by copied control and tile-max candidates."""

const TILE_MAX_FOCUSED_CHECKS = Ref(0)
macro tile_check(expression)
    return esc(quote
        @test $expression
        TILE_MAX_FOCUSED_CHECKS[] += 1
    end)
end

function check_rates_and_bound(velocity, workspace, frozen; cfl, tile_max_expected)
    mesh = velocity.mesh
    got = stable_timestep(velocity, workspace; cfl)
    expected = frozen_cell_cfl(velocity, frozen; cfl)
    @tile_check isequal(got, expected)
    @tile_check typeof(got) === promote_type(eltype(velocity),
                                             eltype(workspace.rates), typeof(cfl))
    for block in eachindex(mesh.blocks)
        @tile_check isequal(interior(workspace.rates, block),
                           interior(frozen.rates, block))
    end
    if tile_max_expected
        nx, ny = mesh.block_size
        columns = workspace.rate_columns_per_tile
        tiles_per_block = workspace.rate_tiles_per_block
        @tile_check length(workspace.rate_tile_max) ==
            length(mesh.blocks) * tiles_per_block
        @tile_check length(workspace.rate_tile_exception) ==
            length(workspace.rate_tile_max)
        @tile_check tiles_per_block == cld(ny, columns)
        coverage = zeros(Int, ny)
        for local_tile in 1:tiles_per_block
            first_j = (local_tile - 1) * columns + 1
            last_j = min(ny, first_j + columns - 1)
            coverage[first_j:last_j] .+= 1
        end
        @tile_check all(==(1), coverage)
        T = promote_type(eltype(velocity), eltype(workspace.rates), typeof(cfl))
        for block in eachindex(mesh.blocks), local_tile in 1:tiles_per_block
            first_j = (local_tile - 1) * columns + 1
            last_j = min(ny, first_j + columns - 1)
            expected_max = zero(T)
            exceptional = false
            for j in first_j:last_j, i in 1:nx
                rate = workspace.rates.blocks[block][i + mesh.halo, j + mesh.halo]
                expected_max = max(expected_max, rate)
                exceptional |= !isfinite(rate) || (iszero(rate) && signbit(rate))
            end
            tile = (block - 1) * tiles_per_block + local_tile
            @tile_check isequal(workspace.rate_tile_max[tile],
                               convert(eltype(workspace.rates), expected_max))
            @tile_check workspace.rate_tile_exception[tile] ==
                (exceptional ? 0x01 : 0x00)
        end
    end
    return got
end

function run_tile_max_focused(tile_max_expected)
    @testset "CFL tile maximum: general leaves and live carriers" begin
        meshes = (
            BlockTreeMesh(base_mesh=(64, 32), block_size=(64, 32),
                          periodic=(false, false)),
            BlockTreeMesh(base_mesh=(32, 64), block_size=(32, 64),
                          periodic=(true, false)),
            BlockTreeMesh(base_mesh=(64, 64), block_size=(8, 8), max_level=1,
                refine=(key, origin, extent)->
                    key.level == 0 && key.i == 0 && key.j == 0),
            BlockTreeMesh(base_mesh=(2, 16), block_size=(2, 16),
                          periodic=(false, true)),
        )
        for mesh in meshes, (FieldT, VelocityT) in
                ((Float32, Float32), (Float32, Float64), (Float64, Float32),
                 (Float64, Float64))
            field = CellField(mesh, FieldT)
            workspace = AdvectionWorkspace(field)
            frozen = FrozenCellAdvectionWorkspace(field)
            velocity = FaceVelocity(mesh, VelocityT; init=zero(VelocityT))
            if tile_max_expected
                @tile_check workspace.rate_tile_max !== workspace.x_flux
                @tile_check workspace.rate_tile_max !== workspace.y_flux
                if Threads.nthreads(:default) in (3, 7) &&
                        mesh.block_size == (32, 64)
                    @tile_check mesh.block_size[2] %
                        workspace.rate_columns_per_tile != 0
                end
            end
            for phase in 1:4
                # High -> low -> zero -> renewed carrier catches stale tile
                # maxima and direct edits to owned regular/residual faces.
                magnitude = (0.75, 0.03, 0.0, 0.19)[phase]
                for index in eachindex(velocity.interior)
                    velocity.interior[index] = VelocityT(
                        (isodd(index + phase) ? -1 : 1) * magnitude)
                end
                for index in eachindex(velocity.boundary)
                    velocity.boundary[index] = VelocityT(
                        (isodd(index + phase) ? 1 : -1) * magnitude / 2)
                end
                if phase == 3 && !isempty(velocity.interior)
                    velocity.interior[1] = -zero(VelocityT)
                end
                for cfl in (Float32(0.6), Float64(0.6))
                    if tile_max_expected
                        fill!(workspace.rate_tile_max, FieldT(NaN))
                        fill!(workspace.rate_tile_exception, 0xff)
                    end
                    check_rates_and_bound(velocity, workspace, frozen;
                        cfl, tile_max_expected)
                    if tile_max_expected
                        @tile_check all(iszero, workspace.rate_tile_exception)
                        @tile_check all(isfinite, workspace.rate_tile_max)
                    end
                end
            end
            TwoPhaseAMR._invalidate_mesh_topology_cache!(mesh)
            @test_throws ArgumentError stable_timestep(velocity, workspace)
            TILE_MAX_FOCUSED_CHECKS[] += 1
        end
    end

    @testset "CFL tile maximum: exceptional stored rates" begin
        for (FieldT, VelocityT) in ((Float32, Float64), (Float64, Float64))
            mesh = BlockTreeMesh(base_mesh=(32, 64), block_size=(32, 64),
                                 periodic=(false, false))
            field = CellField(mesh, FieldT)
            workspace = AdvectionWorkspace(field)
            frozen = FrozenCellAdvectionWorkspace(field)
            velocity = FaceVelocity(mesh, VelocityT; init=zero(VelocityT))
            for special in (VelocityT(Inf), VelocityT(NaN), VelocityT(0.125))
                fill!(velocity.interior, zero(VelocityT))
                fill!(velocity.boundary, zero(VelocityT))
                velocity.interior[1] = special
                if tile_max_expected
                    fill!(workspace.rate_tile_max, FieldT(NaN))
                    fill!(workspace.rate_tile_exception, 0xff)
                end
                check_rates_and_bound(velocity, workspace, frozen;
                    cfl=Float64(0.7), tile_max_expected)
                if tile_max_expected
                    @tile_check any(flag -> !iszero(flag),
                                    workspace.rate_tile_exception) ==
                        !isfinite(special)
                end
            end
            # A synthetic negative metric forces a stored -0.0 rate. The
            # production mesh builder normally gives positive volumes, so
            # negative-zero fallback is also tested directly here.
            old = mesh.blocks[1]
            mesh.blocks[1] = TwoPhaseAMR.LeafBlock(
                old.key, old.origin, old.extent, old.spacing, -old.volume)
            fill!(velocity.interior, zero(VelocityT))
            fill!(velocity.boundary, zero(VelocityT))
            check_rates_and_bound(velocity, workspace, frozen;
                cfl=Float64(0.7), tile_max_expected)
            @tile_check any(rate -> iszero(rate) && signbit(rate),
                            interior(workspace.rates, 1))
            if tile_max_expected
                @tile_check any(flag -> !iszero(flag),
                                workspace.rate_tile_exception)
            end
        end
    end

    @testset "CFL tile maximum: unsupported rate type keeps original scan" begin
        mesh = BlockTreeMesh(base_mesh=(4, 4), block_size=(4, 4),
                             periodic=(true, false))
        field = CellField(mesh, BigFloat)
        workspace = AdvectionWorkspace(field)
        frozen = FrozenCellAdvectionWorkspace(field)
        velocity = FaceVelocity(mesh, BigFloat; init=BigFloat(0.125))
        if tile_max_expected
            fill!(workspace.rate_tile_max, BigFloat(17))
            fill!(workspace.rate_tile_exception, 0xa5)
        end
        got = stable_timestep(velocity, workspace; cfl=BigFloat(0.6))
        expected = frozen_cell_cfl(velocity, frozen; cfl=BigFloat(0.6))
        @tile_check isequal(got, expected)
        @tile_check isequal(interior(workspace.rates, 1), interior(frozen.rates, 1))
        if tile_max_expected
            @tile_check all(==(BigFloat(17)), workspace.rate_tile_max)
            @tile_check all(==(0xa5), workspace.rate_tile_exception)
        end
    end
    return TILE_MAX_FOCUSED_CHECKS[]
end

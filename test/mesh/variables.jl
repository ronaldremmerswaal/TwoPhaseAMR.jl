function all_interior_values(field)
    return reduce(vcat, (vec(collect(interior(field, b))) for b in 1:nblocks(field.mesh)))
end

@testset "cell fields, halos, restriction, and prolongation" begin
    meshes = (
        BlockTreeMesh(block_size = (4, 4)),
        BlockTreeMesh(
            block_size = (4, 4), max_level = 1,
            refine = (key, origin, extent) -> true
        ),
        BlockTreeMesh(
            block_size = (4, 4), max_level = 2,
            refine = (key, origin, extent) ->
            key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
        ),
    )

    for mesh in meshes
        field = CellField(mesh, Float64)
        @test placement(field) == Cell()
        initialize!(field, (x, y) -> 3.25)
        fill_halo!(field)
        @test integrate(field) ≈ 3.25 * prod(mesh.extent)
        for block in field.blocks
            @test all(==(3.25), block)
        end
    end

    # A coarse ghost cell adjacent to finer leaves receives the conservative
    # average of the four fine cells occupying its footprint.
    twolevel = meshes[3]
    linear = CellField(twolevel)
    initialize!(linear, (x, y) -> x)
    fill_halo!(linear)
    coarse_index = findfirst(block -> block.key == BlockKey(1, 1, 0), twolevel.blocks)
    coarse = twolevel.blocks[coarse_index]
    for j in 1:twolevel.block_size[2]
        expected_x = coarse.origin[1] - coarse.spacing[1] / 2
        @test blockdata(linear, coarse_index)[1, j + twolevel.halo] ≈ expected_x
    end

    physical = BlockTreeMesh(block_size = (4, 4), periodic = (false, false))
    field = CellField(physical)
    initialize!(field, (x, y) -> x + 2y)
    fill_halo!(field, Dirichlet((x, y, t) -> t); time = 7.0)
    h = physical.halo
    @test all(field.blocks[1][1, :] .== 7.0)
    @test all(field.blocks[1][end, :] .== 7.0)
    @test all(field.blocks[1][:, 1] .== 7.0)
    @test all(field.blocks[1][:, end] .== 7.0)

    h = 1
    parent = zeros(6, 6)
    for j in 1:4, i in 1:4
        parent[i + h, j + h] = i + 10j
    end
    children = ntuple(_ -> fill(NaN, 6, 6), 4)
    prolong!(children, parent, h)
    reconstructed = zeros(6, 6)
    restrict!(reconstructed, children, h)
    @test reconstructed[2:5, 2:5] == parent[2:5, 2:5]

    # Limited linear prolongation is affine-exact when a valid one-cell parent
    # halo is available, while its four subcell averages remain conservative.
    for j in 0:5, i in 0:5
        parent[i + 1, j + 1] = 1 + 0.3(i - 1.5) - 0.2(j - 1.5)
    end
    prolong!(children, parent, h)
    for (child_index, child) in pairs(children), j in 1:4, i in 1:4
        child_x = mod(child_index - 1, 2)
        child_y = (child_index - 1) ÷ 2
        fine_i = child_x * 4 + i
        fine_j = child_y * 4 + j
        x = (fine_i - 0.5) / 2 - 1
        y = (fine_j - 0.5) / 2 - 1
        @test child[i + h, j + h] ≈ 1 + 0.3x - 0.2y atol = 4eps()
    end
    restrict!(reconstructed, children, h)
    @test reconstructed[2:5, 2:5] ≈ parent[2:5, 2:5] atol = 4eps()

    for (child_index, child) in pairs(children), j in 1:4, i in 1:4
        child[i + h, j + h] = child_index + i / 10 + j / 100
    end
    restrict!(reconstructed, children, h)
    parent_mass = sum(@view reconstructed[2:5, 2:5])
    fine_mass = sum(sum(@view child[2:5, 2:5]) for child in children) / 4
    @test parent_mass ≈ fine_mass rtol = 8eps()

    mesh32 = BlockTreeMesh(
        origin = (0.0f0, 0.0f0), extent = (1.0f0, 1.0f0),
        block_size = (4, 4)
    )
    @test eltype(CellField(mesh32)) == Float32
    @test eltype(FaceVelocity(mesh32, (x, y, t, axis) -> 0.0f0).interior) == Float32
end

@testset "conditional fraction halos restrict numerator and denominator" begin
    # A level-jump mesh with a plane wall: coarse halo slots beside the finer
    # leaves have four donors, some of them covered, so their conditional
    # fraction must be the ratio of the restricted liquid and open fractions.
    mesh = BlockTreeMesh(
        base_mesh = (16, 16), block_size = (8, 8), max_level = 2,
        periodic = (false, false),
        refine = (key, origin, extent) -> key.level == 0 || (
            key.level == 1 && abs(origin[1] - 0.25) < 1.0e-12 &&
                abs(origin[2] - 0.25) < 1.0e-12
        ),
    )
    solid = InterfaceField(mesh; contact_angle = nothing)
    initialize!(solid, (-0.23, 1.0), 0.17)
    geometry = GeometryField(solid)
    field = InterfaceField(mesh; geometry, contact_angle = nothing, init = 0)
    initialize!(field, (0.2, 1.0), 0.5)
    floor = field.fraction_floor
    restricted = 0
    covered_donors = 0
    ratio_of_means_error = 0.0
    for entry in mesh.halo_entries
        entry.physical_boundary && continue
        entry.donor_count > 1 || continue
        restricted += 1
        liquid_mass = 0.0
        open_mass = 0.0
        mean_of_ratios = 0.0
        for index in entry.first_donor:(entry.first_donor + entry.donor_count - 1)
            weight = mesh.halo_weights[index]
            donor = mesh.halo_donors[index]
            liquid = TwoPhaseAMR._get(field.fraction, donor)
            open = TwoPhaseAMR._get(geometry.open_fraction, donor)
            liquid_mass += weight * liquid
            open_mass += weight * open
            mean_of_ratios += weight * TwoPhaseAMR._conditional_fraction(field, donor)
            iszero(open) && (covered_donors += 1)
        end
        expected = open_mass > 0 ? clamp(liquid_mass, 0.0, open_mass) / open_mass : 0.0
        expected <= floor && (expected = 0.0)
        expected >= 1 - floor && (expected = 1.0)
        @test TwoPhaseAMR._get(field.conditional_fraction, entry.destination) ≈
            expected atol = 1.0e-12
        ratio_of_means_error = max(ratio_of_means_error, abs(mean_of_ratios - expected))
    end
    @test restricted > 0
    @test covered_donors > 0
    # Three covered children and one liquid child restrict to one, while the
    # mean of the per-cell ratios is 0.25.
    @test ratio_of_means_error > 0.25
end

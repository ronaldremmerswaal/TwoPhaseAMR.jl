using LinearAlgebra: Symmetric, eigvals, isposdef
using Random

@testset "placements and block tree" begin
    @test Cell() isa AbstractPlacement
    @test Face{1}() isa AbstractPlacement
    @test Face{2}() isa AbstractPlacement
    @test_throws ArgumentError Face{3}()

    oneblock = BlockTreeMesh(block_size = (8, 6))
    @test nblocks(oneblock) == 1
    @test ncells(oneblock) == 48
    @test isbalanced(oneblock)
    @test length(oneblock.faces) == 2ncells(oneblock)
    @test isempty(oneblock.boundary_faces)

    uniform = BlockTreeMesh(
        block_size = (4, 4), max_level = 1,
        refine = (key, origin, extent) -> true
    )
    @test nblocks(uniform) == 4
    @test all(block -> block.key.level == 1, uniform.blocks)
    @test isbalanced(uniform)

    twolevel = BlockTreeMesh(
        block_size = (4, 4), max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
    )
    @test nblocks(twolevel) == 7
    @test count(block -> block.key.level == 1, twolevel.blocks) == 3
    @test count(block -> block.key.level == 2, twolevel.blocks) == 4
    @test isbalanced(twolevel)

    # A sharply localized level-three request is expanded to a fully 2:1
    # balanced leaf set, including diagonal neighbors.
    balanced = BlockTreeMesh(
        block_size = (2, 2), max_level = 3,
        refine = (key, origin, extent) ->
        key.level == 0 || (key.i == 0 && key.j == 0),
    )
    @test isbalanced(balanced)
    @test maximum(block.key.level for block in balanced.blocks) == 3
    right_edge_levels = [
        block.key.level for block in balanced.blocks
            if block.origin[1] + block.extent[1] ≈ 1.0
    ]
    @test maximum(right_edge_levels) >= 2

    physical = BlockTreeMesh(block_size = (4, 4), periodic = (false, false))
    @test length(physical.faces) == 24
    @test length(physical.boundary_faces) == 16

    @test_throws ArgumentError BlockTreeMesh(block_size = (3, 4))
    @test_throws ArgumentError BlockTreeMesh(halo = 0)
    @test_throws ArgumentError BlockTreeMesh(max_level = -1)
end

@testset "face schedules depend on active topology, not refinement ceiling" begin
    for T in (Float32, Float64),
            periodic in ((false, false), (true, false), (true, true)), level in 0:2
        refine = (key, origin, extent) ->
        key.level < level &&
            (key.level == 0 || (iseven(key.i) && iseven(key.j)))
        reference = BlockTreeMesh(
            origin = (T(-0.3), T(1.25)), extent = (T(1.3), T(2.7)),
            base_mesh = (8, 12), block_size = (4, 6), periodic = periodic,
            max_level = level, refine = refine,
        )
        higher_ceiling = BlockTreeMesh(
            origin = (T(-0.3), T(1.25)), extent = (T(1.3), T(2.7)),
            base_mesh = (8, 12), block_size = (4, 6), periodic = periodic,
            max_level = level + 2, refine = refine,
        )
        @test reference.faces == higher_ceiling.faces
        @test reference.boundary_faces == higher_ceiling.boundary_faces
        @test reference.face_dual_volumes == higher_ceiling.face_dual_volumes
        @test TwoPhaseAMR._same_tree_topology(reference, higher_ceiling)
    end
end

@testset "face schedule cell and axis invariant" begin
    mesh = BlockTreeMesh(block_size = (4, 6), periodic = (false, false))
    faces = copy(mesh.faces)
    boundaries = copy(mesh.boundary_faces)
    check() = TwoPhaseAMR._check_face_schedules(
        faces, boundaries, nblocks(mesh), mesh.block_size,
    )
    @test isnothing(check())

    face = first(faces)
    ref = face.minus
    nx, ny = mesh.block_size
    bad_refs = (
        CellRef(0, ref.i, ref.j), CellRef(nblocks(mesh) + 1, ref.i, ref.j),
        CellRef(ref.block, 0, ref.j), CellRef(ref.block, nx + 1, ref.j),
        CellRef(ref.block, ref.i, 0), CellRef(ref.block, ref.i, ny + 1),
    )
    function check_error(expected)
        err = try
            check()
            nothing
        catch exception
            exception
        end
        @test err isa ErrorException &&
            occursin(expected, sprint(showerror, err))
    end
    for bad_ref in bad_refs, side in (:minus, :plus)
        faces[1] = TwoPhaseAMR.InteriorFace(
            side == :minus ? bad_ref : face.minus,
            side == :plus ? bad_ref : face.plus,
            face.axis, face.center, face.area,
        )
        check_error("invalid interior face 1")
    end
    for axis in (0, 3)
        faces[1] = TwoPhaseAMR.InteriorFace(
            face.minus, face.plus, axis, face.center, face.area,
        )
        check_error("invalid interior face 1")
    end
    faces[1] = face
    # Report the first bad face in schedule order, not merely any bad face.
    second, third = faces[2], faces[3]
    faces[2] = TwoPhaseAMR.InteriorFace(
        second.minus, second.plus, 0, second.center, second.area,
    )
    faces[3] = TwoPhaseAMR.InteriorFace(
        third.minus, third.plus, 3, third.center, third.area,
    )
    check_error("invalid interior face 2")
    faces[2], faces[3] = second, third

    boundary = first(boundaries)
    for bad_ref in bad_refs
        boundaries[1] = TwoPhaseAMR.BoundaryFace(
            bad_ref, boundary.axis, boundary.side, boundary.center, boundary.area,
        )
        check_error("invalid boundary face 1")
    end
    for axis in (0, 3)
        boundaries[1] = TwoPhaseAMR.BoundaryFace(
            boundary.cell, axis, boundary.side, boundary.center, boundary.area,
        )
        check_error("invalid boundary face 1")
    end
    boundaries[1] = boundary

    # Per-axis face fields index their faces as one contiguous range.
    @test faces[1].axis == 1 && faces[end].axis == 2
    faces[1], faces[end] = faces[end], faces[1]
    check_error("every axis-1 face before every axis-2 face")
    faces[1], faces[end] = faces[end], faces[1]
    @test boundaries[1].axis == 1 && boundaries[end].axis == 2
    boundaries[1], boundaries[end] = boundaries[end], boundaries[1]
    check_error("every axis-1 face before every axis-2 face")
end


@testset "multi-root base mesh" begin
    extent = (1.0, 3.0)
    single = BlockTreeMesh(
        extent = extent, block_size = (8, 24), periodic = (false, false),
    )
    partitioned = BlockTreeMesh(
        extent = extent, base_mesh = (8, 24), block_size = (8, 8),
        periodic = (false, false),
    )

    @test partitioned.base_mesh == (8, 24)
    @test partitioned.root.layout == (1, 3)
    @test length(partitioned.root.roots) == 3
    @test [root.data.key for root in partitioned.root.roots] ==
        [BlockKey(0, 0, j) for j in 0:2]
    @test ncells(partitioned) == ncells(single) == 8 * 24

    function cell_geometry(mesh)
        result = NTuple{3, Float64}[]
        nx, ny = mesh.block_size
        for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
            x, y = cellcenter(mesh, block, i, j)
            push!(result, (x, y, mesh.blocks[block].volume))
        end
        return sort!(result)
    end
    face_geometry(mesh) = sort!(
        [
            (face.axis, face.center[1], face.center[2], face.area)
                for face in mesh.faces
        ]
    )
    boundary_geometry(mesh) = sort!(
        [
            (face.axis, face.side, face.center[1], face.center[2], face.area)
                for face in mesh.boundary_faces
        ]
    )
    @test cell_geometry(partitioned) == cell_geometry(single)
    @test face_geometry(partitioned) == face_geometry(single)
    @test boundary_geometry(partitioned) == boundary_geometry(single)

    field = CellField(partitioned)
    initialize!(field, (x, y) -> 1 + 2x - 3y)
    fill_halo!(field)
    for entry in partitioned.halo_entries
        entry.physical_boundary && continue
        ref = entry.destination
        value = field.blocks[ref.block][
            ref.i + partitioned.halo, ref.j + partitioned.halo,
        ]
        @test value ≈ 1 + 2entry.center[1] - 3entry.center[2] atol = 8eps()
    end

    periodic = BlockTreeMesh(
        extent = extent, base_mesh = (8, 24), block_size = (8, 8),
        periodic = (true, false),
    )
    @test length(periodic.faces) == 8 * 24 + 8 * 23
    @test length(periodic.boundary_faces) == 2 * 8

    adaptive = BlockTreeMesh(
        extent = (1.0, 2.0), base_mesh = (4, 8), block_size = (4, 4),
        periodic = (false, false), max_level = 2,
        refine = (key, origin, extent) ->
        (key.level == 0 && key.j == 0) ||
            (key.level == 1 && key.j == 1),
    )
    @test isbalanced(adaptive)
    @test maximum(block.key.level for block in adaptive.blocks) == 2
    @test any(
        block -> block.key.level == 1 && block.origin[2] >= 1,
        adaptive.blocks,
    )
    @test length(adaptive.root.roots) == 2

    @test_throws ArgumentError BlockTreeMesh(
        base_mesh = (12, 8), block_size = (8, 8),
    )
    @test_throws ArgumentError BlockTreeMesh(
        base_mesh = (8.0, 8), block_size = (8, 8),
    )
    @test_throws ArgumentError BlockTreeMesh(
        base_mesh = (false, 8), block_size = (8, 8),
    )
end

@testset "indexed face-line neighbourhood search" begin
    lines = Dict(
        0 => [(lo = 0, hi = 3), (lo = 3, hi = 8)],
        2 => [(lo = 0, hi = 2), (lo = 4, hi = 8)],
        4 => [(lo = 1, hi = 5), (lo = 5, hi = 8)],
        6 => [(lo = 0, hi = 4), (lo = 6, hi = 8)],
    )
    coordinates = sort!(collect(keys(lines)))
    period = 8

    function exhaustive(normal, tangent, periodic)
        lower_distance = typemin(Int)
        upper_distance = typemax(Int)
        lower_coordinate = lower_segment = 0
        upper_coordinate = upper_segment = 0
        for coordinate in coordinates
            delta = coordinate - normal
            periodic && (delta -= round(Int, delta / period) * period)
            iszero(delta) && continue
            segment = findfirst(
                entry -> entry.lo <= tangent <= entry.hi,
                lines[coordinate],
            )
            isnothing(segment) && continue
            if delta < 0 && delta > lower_distance
                lower_distance = delta
                lower_coordinate = coordinate
                lower_segment = segment
            elseif delta > 0 && delta < upper_distance
                upper_distance = delta
                upper_coordinate = coordinate
                upper_segment = segment
            end
        end
        return (
            lower_distance, lower_coordinate, lower_segment,
            upper_distance, upper_coordinate, upper_segment,
        )
    end

    for periodic in (false, true), normal in 0:period,
            tangent in (0.5, 2.0, 3.5, 5.0, 6.5, 7.5)
        indexed = TwoPhaseAMR._bracketing_line_segments(
            lines, coordinates, normal, tangent, period, periodic,
        )
        @test indexed == exhaustive(normal, tangent, periodic)
    end
    line = lines[0]
    for tangent in (0, 2, 3, 4, 8)
        expected = findfirst(entry -> entry.lo <= tangent <= entry.hi, line)
        @test TwoPhaseAMR._line_segment_at(line, tangent) == expected
    end
end

# Independent oracle in a common block-key lattice. A zero overlap in exactly
# one direction is a face; zero in both is a corner. Periodic images are
# checked explicitly, independently of the implementation's neighbour search.
function _test_ungraded_pairs(mesh)
    keys = [block.key for block in mesh.blocks]
    level = maximum(key.level for key in keys)
    period = (mesh.base_mesh .÷ mesh.block_size) .* (1 << level)
    boxes = map(keys) do key
        scale = 1 << (level - key.level)
        (key.i * scale, (key.i + 1) * scale, key.j * scale, (key.j + 1) * scale)
    end
    faces = corners = 0
    for i in eachindex(keys), j in (i + 1):length(keys)
        abs(keys[i].level - keys[j].level) > 1 || continue
        a, b = boxes[i], boxes[j]
        xshifts = mesh.periodic[1] ? (-period[1], 0, period[1]) : (0,)
        yshifts = mesh.periodic[2] ? (-period[2], 0, period[2]) : (0,)
        face = corner = false
        for dx in xshifts, dy in yshifts
            x = min(a[2], b[2] + dx) - max(a[1], b[1] + dx)
            y = min(a[4], b[4] + dy) - max(a[3], b[3] + dy)
            face |= (x == 0 && y > 0) || (y == 0 && x > 0)
            corner |= x == 0 && y == 0
        end
        faces += face
        corners += corner && !face
    end
    return faces, corners
end

@testset "integer balance on non-dyadic extents" begin
    rng = Random.MersenneTwister(46)
    extents = 0.3 .+ 12 .* rand(rng, 200)
    for extent in extents, periodic in (
                (false, false), (true, false), (false, true), (true, true),
            ), variant in 1:2
        refine_rng = Random.MersenneTwister(355739)
        refine = variant == 1 ? (key, o, e) -> key.i == key.j == 0 :
            (key, o, e) -> rand(refine_rng) <
            (0.5, 0.45, 0.4, 0.35)[key.level + 1]
        mesh = BlockTreeMesh(
            base_mesh = (8, 8), block_size = (2, 2),
            extent = (extent, extent), periodic = periodic,
            max_level = 4, refine = refine,
        )
        faces, corners = _test_ungraded_pairs(mesh)
        @test faces == 0
        @test corners == 0
        @test isbalanced(mesh) == (faces == 0 && corners == 0)
    end
end

@testset "balance detects periodic seam violations" begin
    function unbalanced_seam(periodic)
        mesh = BlockTreeMesh(
            base_mesh = (8, 8), block_size = (2, 2),
            extent = (0.7, 3.3), periodic = periodic, max_level = 3,
        )
        root = mesh.root.roots[1]
        TwoPhaseAMR._split!(root)
        TwoPhaseAMR._split!(root[1, 1])
        TwoPhaseAMR._split!(root[1, 1][1, 1])
        _, blocks = TwoPhaseAMR._leaf_blocks(mesh.root, mesh.block_size)
        empty!(mesh.blocks)
        append!(mesh.blocks, blocks)
        return mesh
    end
    for periodic in ((false, false), (true, false), (false, true), (true, true))
        mesh = unbalanced_seam(periodic)
        faces, corners = _test_ungraded_pairs(mesh)
        @test isbalanced(mesh) == (faces == 0 && corners == 0)
        @test isbalanced(mesh) == !any(periodic)
    end
end

@testset "LGPI root seam stays 2:1 balanced" begin
    # At step 2404 of #85 the level-zero block (5, 5) was separated by one
    # ulp from level-two blocks at (24, 22) and (24, 23).
    mesh = BlockTreeMesh(
        base_mesh = (40, 24), block_size = (4, 4),
        extent = (4.0, 2.4), periodic = (false, false), max_level = 2,
        refine = (key, origin, extent) ->
        key == BlockKey(0, 6, 5) || key == BlockKey(1, 12, 11),
    )
    @test any(block -> block.key == BlockKey(2, 24, 22), mesh.blocks)
    @test !any(block -> block.key == BlockKey(0, 5, 5), mesh.blocks)
    @test _test_ungraded_pairs(mesh) == (0, 0)
    @test isbalanced(mesh)
    @test GradientOperator(mesh) isa GradientOperator
end

@testset "face and halo topology is independent of geometry" begin
    refine = (key, o, e) ->
    (key.level == 0 && key.i == 1 && key.j == 0) ||
        (key.level == 1 && key.i == 3 && key.j == 0)
    function topology_signature(mesh)
        interface = InterfaceField(mesh; T = Float64)
        height_cache = TwoPhaseAMR._height_cache!(interface)
        lattice = TwoPhaseAMR._finest_lattice_size(mesh)
        shear_topology = map((1, 2)) do axis
            partition = TwoPhaseAMR.ShearPartition(Float64)
            TwoPhaseAMR._shear_nodes!(partition.nodes, mesh, lattice, axis)
            TwoPhaseAMR._shear_contacts!(partition, mesh, lattice, axis)
            (partition.nodes, partition.contacts)
        end
        coarse = TwoPhaseAMR._capped_tree_mesh(mesh, 1)
        cell_transfer = TwoPhaseAMR._tree_cell_prolongation(mesh, coarse, Float64)
        face_transfer = TwoPhaseAMR._tree_face_prolongation(mesh, coarse, Float64)
        dual_topology = map((1, 2)) do axis
            schedule = TwoPhaseAMR._amr_dual_schedule(
                mesh, Val(axis), Float64; build_three_point_stencils = false,
            )
            (schedule.faces, schedule.flux_indices, schedule.flux_weights)
        end
        return (
            [block.key for block in mesh.blocks],
            [(face.minus, face.plus, face.axis) for face in mesh.faces],
            [(face.cell, face.axis, face.side) for face in mesh.boundary_faces],
            [TwoPhaseAMR._face_lattice_geometry(mesh, face) for face in mesh.faces],
            [TwoPhaseAMR._face_lattice_geometry(mesh, face) for face in mesh.boundary_faces],
            [
                (TwoPhaseAMR._vertex_key(mesh, face, 1), TwoPhaseAMR._vertex_key(mesh, face, 2))
                    for face in mesh.faces
            ],
            shear_topology,
            cell_transfer, face_transfer,
            dual_topology,
            [
                (
                    ghost.block, ghost.i, ghost.j,
                    [(donor.cell, donor.bounds, donor.weight) for donor in ghost.donors],
                )
                    for ghost in height_cache.ghosts
            ],
            [
                (
                    entry.destination, entry.physical_boundary,
                    mesh.halo_donors[entry.first_donor:(entry.first_donor + entry.donor_count - 1)],
                    mesh.halo_weights[entry.first_donor:(entry.first_donor + entry.donor_count - 1)],
                ) for entry in mesh.halo_entries
            ],
        )
    end
    for periodic in ((false, false), (true, false), (false, true), (true, true))
        reference = BlockTreeMesh(
            base_mesh = (16, 8), block_size = (4, 4), halo = 2,
            max_level = 2, periodic = periodic, refine = refine,
        )
        for (origin, extent) in (
                ((0.0, 0.0), (0.7, 3.3)),
                ((1.0e12, -1.0e12), (0.7, 3.3)),
                ((1.0f4, -1.0f4), (0.7f0, 3.3f0)),
            )
            mesh = BlockTreeMesh(
                base_mesh = (16, 8), block_size = (4, 4), halo = 2,
                origin = origin, extent = extent, max_level = 2,
                periodic = periodic, refine = refine,
            )
            @test topology_signature(mesh) == topology_signature(reference)
        end
    end
end

@testset "periodic direction resolved by exactly two cells" begin
    function positive_definite_metric(metric)
        metric isa TwoPhaseAMR.SparseInverseFaceMetric &&
            return isposdef(Symmetric(metric.matrix))
        return all(>(0), metric.diagonal)
    end
    # A periodic direction with two cells gives a centre-to-centre difference of
    # +-L/2, and both periodic images are equally long: the wrapped difference
    # alone is a tie, which `round(-0.5) == -0.0` used to settle in favour of
    # the negative image (a reversed seam displacement). The documented rule is
    # that the face orientation decides the normal sign -- the plus cell always
    # lies on the +axis side of the shared face -- and that the magnitude is the
    # shortest periodic image. The tangential component keeps its wrapped sign,
    # which stays unambiguous because the tangential offset of two centres is at
    # most half a coarse cell.
    for (base, periodic) in (
            ((2, 4), (true, false)),
            ((4, 2), (false, true)),
            ((2, 2), (true, true)),
        )
        mesh = BlockTreeMesh(
            base_mesh = base, block_size = (2, 2),
            extent = (Float64(base[1]), Float64(base[2])), periodic = periodic,
        )
        for (index, face) in pairs(mesh.faces)
            displacement = TwoPhaseAMR._face_displacement(mesh, face)
            normal = face.axis
            @test displacement[normal] ==
                (
                mesh.blocks[face.minus.block].spacing[normal] +
                    mesh.blocks[face.plus.block].spacing[normal]
            ) / 2
            @test displacement[3 - normal] == 0
        end
        # The seam face and the interior face of a two-cell direction carry the
        # same displacement, so neither is singled out by the metric.
        for axis in 1:2
            @test unique(
                TwoPhaseAMR._face_displacement(mesh, face)[axis]
                    for face in mesh.faces if face.axis == axis
            ) == [mesh.extent[axis] / base[axis]]
        end

        operator = GradientOperator(mesh)
        @test positive_definite_metric(operator.inverse_metric)
        pressure = CellField(mesh)
        initialize!(pressure, (x, y) -> sinpi(x) + 0.3cospi(2y) - 0.2sinpi(3y))
        numerical = FaceVelocity(mesh)
        gradient!(numerical, pressure, operator)
        for (index, face) in pairs(mesh.faces)
            distance = TwoPhaseAMR._face_displacement(mesh, face)[face.axis]
            expected = (
                TwoPhaseAMR._get(pressure, face.plus) -
                    TwoPhaseAMR._get(pressure, face.minus)
            ) / distance
            @test numerical.interior[index] ≈ expected atol = 8eps() rtol = 8eps()
        end
        # The area-weighted pressure jumps of an affine field must map to its
        # face-normal gradient; a reversed seam displacement breaks this.
        jump = similar(operator.pressure_jump)
        result = similar(jump)
        for coordinate_axis in 1:2
            for (index, face) in pairs(mesh.faces)
                jump[index] = face.area *
                    TwoPhaseAMR._face_displacement(mesh, face)[coordinate_axis]
            end
            TwoPhaseAMR._mul_inverse_metric!(
                result, operator.inverse_metric, jump
            )
            for (index, face) in pairs(mesh.faces)
                expected = face.axis == coordinate_axis ? 1.0 : 0.0
                @test result[index] ≈ expected atol = 2.0e-13 rtol = 2.0e-13
            end
        end
    end

    # The operator built from the corrected seam displacement is the ordinary
    # MAC operator of that grid: symmetric, semidefinite, and with exactly the
    # constant pressure null mode.
    mesh = BlockTreeMesh(base_mesh = (2, 2), block_size = (2, 2), extent = (2.0, 2.0))
    operator = GradientOperator(mesh)
    numerical = FaceVelocity(mesh)
    divergence = CellField(mesh)
    basis = CellField(mesh)
    nx, ny = mesh.block_size
    refs = [
        TwoPhaseAMR.CellRef(block, i, j)
            for block in eachindex(mesh.blocks) for j in 1:ny for i in 1:nx
    ]
    pressure_matrix = zeros(length(refs), length(refs))
    for (column, source) in pairs(refs)
        for block in eachindex(mesh.blocks)
            fill!(interior(basis, block), 0)
        end
        TwoPhaseAMR._set!(basis, 1.0, source)
        gradient!(numerical, basis, operator)
        divergence!(divergence, numerical)
        for (row, target) in pairs(refs)
            volume = mesh.blocks[target.block].volume
            pressure_matrix[row, column] =
                -volume * TwoPhaseAMR._get(divergence, target)
        end
    end
    @test pressure_matrix ≈ transpose(pressure_matrix) atol = 2.0e-13 rtol = 2.0e-13
    eigenvalues = eigvals(Symmetric(pressure_matrix))
    spectral_tolerance = 2.0e-12 * maximum(abs, eigenvalues)
    @test first(eigenvalues) >= -spectral_tolerance
    @test count(value -> abs(value) <= spectral_tolerance, eigenvalues) == 1

    # Non-dyadic extents put the centre difference at +-L/2 up to round-off, so
    # the tie must not depend on which side of the half period the floating
    # point difference lands on.
    for extent in ((1.2, 0.8), (0.7, 3.3))
        non_dyadic = BlockTreeMesh(
            base_mesh = (2, 2), block_size = (2, 2), extent = extent,
            periodic = (true, true),
        )
        non_dyadic_operator = GradientOperator(non_dyadic)
        @test positive_definite_metric(non_dyadic_operator.inverse_metric)
        for face in non_dyadic.faces
            normal = face.axis
            @test TwoPhaseAMR._face_displacement(non_dyadic, face)[normal] ≈
                (
                non_dyadic.blocks[face.minus.block].spacing[normal] +
                    non_dyadic.blocks[face.plus.block].spacing[normal]
            ) / 2 rtol = 8eps()
        end
        jump = similar(non_dyadic_operator.pressure_jump)
        result = similar(jump)
        for coordinate_axis in 1:2
            for (index, face) in pairs(non_dyadic.faces)
                jump[index] = face.area *
                    TwoPhaseAMR._face_displacement(non_dyadic, face)[coordinate_axis]
            end
            TwoPhaseAMR._mul_inverse_metric!(
                result, non_dyadic_operator.inverse_metric, jump
            )
            for (index, face) in pairs(non_dyadic.faces)
                expected = face.axis == coordinate_axis ? 1.0 : 0.0
                @test result[index] ≈ expected atol = 2.0e-13 rtol = 2.0e-13
            end
        end
    end

    # Refining one of the two roots leaves level-0 cells in the other, so the
    # two-cell tie survives beside coarse--fine faces: the coarse seam faces
    # keep the covering-space displacement, the patch metric is no longer
    # diagonal, and it stays positive definite.
    refined = BlockTreeMesh(
        base_mesh = (2, 4), block_size = (2, 2), extent = (2.0, 4.0),
        periodic = (true, false), max_level = 1,
        refine = (key, origin, extent) -> key.level == 0 && key.j == 0,
    )
    levels = [block.key.level for block in refined.blocks]
    @test 0 in levels && 1 in levels
    tie = [
        face for face in refined.faces
            if face.axis == 1 &&
            refined.blocks[face.minus.block].key.level == 0 &&
            refined.blocks[face.plus.block].key.level == 0
    ]
    @test !isempty(tie)
    @test all(face -> TwoPhaseAMR._face_displacement(refined, face)[1] == 1.0, tie)
    refined_operator = GradientOperator(refined)
    @test refined_operator.inverse_metric isa
        TwoPhaseAMR.SparseInverseFaceMetric
    @test isposdef(Symmetric(refined_operator.inverse_metric.matrix))
    for (index, face) in pairs(refined.faces)
        normal = face.axis
        @test TwoPhaseAMR._face_displacement(refined, face)[normal] ==
            (
            refined.blocks[face.minus.block].spacing[normal] +
                refined.blocks[face.plus.block].spacing[normal]
        ) / 2
    end
end

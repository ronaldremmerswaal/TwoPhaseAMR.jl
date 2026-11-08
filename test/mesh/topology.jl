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

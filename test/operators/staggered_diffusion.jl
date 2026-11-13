using LinearAlgebra: Diagonal, I, Symmetric, dot, eigvals, opnorm
using SparseArrays: SparseMatrixCSC, nnz, nonzeros, nzrange, rowvals, spdiagm
using Random: MersenneTwister, rand
using Krylov

function _face_component_energy(velocity, tendency, component)
    result = 0.0
    for local_index in eachindex(component.local_to_global)
        global_index = component.local_to_global[local_index]
        result += component.volumes[local_index] *
            velocity.interior[global_index] *
            tendency.interior[global_index]
    end
    return result
end

@testset "staggered diffusion viscosity dispatch" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    boundary = FreeSlip()
    scalar = StaggeredDiffusionOperator(mesh; viscosity = 0.2, boundary)
    scalar_compat = TwoPhaseAMR._staggered_diffusion_operator(
        mesh; viscosity = 0.2, boundary, T = Float64,
        construction_cache = nothing, reuse = nothing, cache_layer = 1,
        geometry = nothing,
    )
    @test scalar isa StaggeredDiffusionOperator
    @test scalar_compat.viscosity == scalar.viscosity
    @test scalar_compat.x.stiffness == scalar.x.stiffness
    @test scalar_compat.y.stiffness == scalar.y.stiffness
    @test scalar_compat.x.laplacian == scalar.x.laplacian
    @test scalar_compat.y.laplacian == scalar.y.laplacian

    mesh32 = BlockTreeMesh(
        block_size = (4, 4), origin = (0.0f0, 0.0f0), extent = (1.0f0, 1.0f0),
    )
    scalar32 = StaggeredDiffusionOperator(
        mesh32; viscosity = 0.2f0, boundary, T = Float32,
    )
    viscosity32 = CellField(mesh32; init = 0.7f0)
    variable32 = TwoPhaseAMR._staggered_diffusion_operator(
        mesh32; viscosity = viscosity32, boundary, T = Float32,
        construction_cache = TwoPhaseAMR.VariableDiffusionConstructionCache(Float32),
    )
    @test eltype(scalar32.x.laplacian) === Float32
    @test eltype(variable32.stiffness) === Float32

    viscosity = CellField(mesh, (x, y) -> 1 + 0.2sinpi(2x) * cospi(2y))
    variable = StaggeredDiffusionOperator(mesh; viscosity, boundary)
    cache = TwoPhaseAMR.VariableDiffusionConstructionCache(Float64)
    cached = TwoPhaseAMR._staggered_diffusion_operator(
        mesh; viscosity, boundary, T = Float64, construction_cache = cache,
        cache_layer = 2, geometry = nothing,
    )
    reused = TwoPhaseAMR._staggered_diffusion_operator(
        mesh; viscosity, boundary, T = Float64, construction_cache = cache,
        reuse = cached, cache_layer = 2, geometry = nothing,
    )
    @test variable isa VariableStaggeredDiffusionOperator
    @test reused isa VariableStaggeredDiffusionOperator
    @test reused.strain == variable.strain
    @test reused.stress_interpolation == variable.stress_interpolation
    @test reused.stress_weights == variable.stress_weights
    @test reused.stiffness == variable.stiffness
    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? sinpi(2y) : cospi(2x),
    )
    expected = FaceVelocity(mesh)
    actual = FaceVelocity(mesh)
    diffusion!(expected, velocity, variable)
    diffusion!(actual, velocity, reused)
    @test isequal(actual.interior, expected.interior)
    @test isequal(actual.boundary, expected.boundary)

    geometry = GeometryField(mesh, (x, y) -> y - 0.18 - 0.1x)
    interface = InterfaceField(
        mesh, (x, y) -> x + 0.2y - 0.47; geometry, lipschitz = hypot(1, 0.2),
    )
    phases = (2.0, 0.4)
    interface_operator = StaggeredDiffusionOperator(
        interface; viscosity = phases, boundary,
    )
    interface_cache = TwoPhaseAMR.VariableDiffusionConstructionCache(Float64)
    cached_interface = TwoPhaseAMR._staggered_diffusion_operator(
        interface; viscosity = phases, boundary,
        construction_cache = interface_cache, cache_layer = 2,
    )
    reused_interface = TwoPhaseAMR._staggered_diffusion_operator(
        interface; viscosity = phases, boundary,
        construction_cache = interface_cache, reuse = cached_interface,
        cache_layer = 2,
    )
    @test reused_interface.volumes == interface_operator.volumes
    @test reused_interface.strain == interface_operator.strain
    @test reused_interface.stress_weights == interface_operator.stress_weights
    @test reused_interface.stiffness == interface_operator.stiffness

    for (invalid, message) in (
            (0.0, "constant viscosity must be strictly positive"),
            (NaN, "constant viscosity must be strictly positive"),
            (
                CellField(mesh; init = 0.0),
                "cell dynamic viscosity must be finite and strictly positive",
            ),
            (
                CellField(mesh; init = Inf),
                "cell dynamic viscosity must be finite and strictly positive",
            ),
        )
        exception = try
            StaggeredDiffusionOperator(mesh; viscosity = invalid)
            nothing
        catch error
            error
        end
        @test exception isa ArgumentError
        @test exception isa ArgumentError && exception.msg == message
    end
    @test StaggeredDiffusionOperator(mesh; viscosity = Inf).viscosity == Inf
    for invalid_options in (
            (; geometry = :invalid), (; construction_cache = cache),
            (; reuse = reused), (; cache_layer = 2),
        )
        @test_throws ArgumentError TwoPhaseAMR._staggered_diffusion_operator(
            mesh; viscosity = 0.2, invalid_options...,
        )
    end
    other_mesh = BlockTreeMesh(block_size = (4, 4))
    @test_throws ArgumentError StaggeredDiffusionOperator(
        mesh; viscosity = CellField(other_mesh; init = 1.0),
    )
    @test_throws ArgumentError TwoPhaseAMR._staggered_diffusion_operator(
        mesh; viscosity,
        construction_cache = TwoPhaseAMR.VariableDiffusionConstructionCache(Float32),
    )
    @test_throws ArgumentError TwoPhaseAMR._staggered_diffusion_operator(
        mesh; viscosity, reuse = :invalid,
    )
end

"""Direct `transpose(E) * Diagonal(W) * E` reference of the stiffness built by
`_symmetric_stiffness!`, with the former `sortperm` refresh map."""
function reference_symmetric_stiffness(strain, weights)
    T = eltype(strain)
    assembled = transpose(strain) * spdiagm(0 => weights) * strain
    # Unit entries expose the full structural pattern: outer products that
    # cancel for the current viscosity must stay stored.
    unit_strain = copy(strain)
    fill!(nonzeros(unit_strain), one(T))
    reference = transpose(unit_strain) * unit_strain
    reference_rows = rowvals(reference)
    reference_values = nonzeros(reference)
    for column in axes(reference, 2), pointer in nzrange(reference, column)
        row = reference_rows[pointer]
        reference_values[pointer] =
            (assembled[row, column] + assembled[column, row]) / 2
    end
    positions = Dict{Tuple{Int, Int}, Int}()
    for column in axes(reference, 2), pointer in nzrange(reference, column)
        positions[(reference_rows[pointer], column)] = pointer
    end
    stress_entries = [Tuple{Int, T}[] for _ in 1:size(strain, 1)]
    strain_rows = rowvals(strain)
    strain_values = nonzeros(strain)
    for column in axes(strain, 2), pointer in nzrange(strain, column)
        push!(stress_entries[strain_rows[pointer]], (column, strain_values[pointer]))
    end
    pointers = Int[]
    stresses = Int[]
    factors = T[]
    for stress in eachindex(stress_entries),
            (column, column_value) in stress_entries[stress],
            (row, row_value) in stress_entries[stress]

        pointer = get(positions, (row, column), 0)
        iszero(pointer) && error("stress refresh changed diffusion sparsity")
        push!(pointers, pointer)
        push!(stresses, stress)
        push!(factors, row_value * column_value)
    end
    permutation = sortperm(pointers)
    permute!(pointers, permutation)
    permute!(stresses, permutation)
    permute!(factors, permutation)
    offsets = Vector{Int}(undef, nnz(reference) + 1)
    contribution = 1
    for pointer in 1:nnz(reference)
        offsets[pointer] = contribution
        while contribution <= length(pointers) && pointers[contribution] == pointer
            contribution += 1
        end
    end
    offsets[end] = contribution
    contribution == length(pointers) + 1 ||
        error("diffusion coefficient refresh contains an invalid sparse pointer")
    return reference, pointers, stresses, factors, offsets
end

@testset "variable-viscosity stiffness matches a direct symmetric assembly" begin
    refine = (key, origin, extent) ->
    key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0)
    meshes = (
        BlockTreeMesh(block_size = (4, 4), max_level = 2, refine = refine),
        BlockTreeMesh(
            block_size = (4, 4), periodic = (false, false), max_level = 2,
            refine = refine,
        ),
    )
    for mesh in meshes
        viscosity = CellField(mesh, (x, y) -> 1 + 0.2sinpi(2x) * cospi(2y))
        operator = TwoPhaseAMR._staggered_diffusion_operator(
            mesh; viscosity, boundary = FreeSlip(), T = Float64,
        )
        reference, pointers, stresses, factors, offsets =
            reference_symmetric_stiffness(
            operator.strain, operator.stress_weights,
        )
        stiffness = operator.stiffness
        # Column pointers, row indices and stored values in storage order.
        @test (stiffness.colptr, rowvals(stiffness), nonzeros(stiffness)) ==
            (reference.colptr, rowvals(reference), nonzeros(reference))
        @test operator.stiffness_pointers == pointers
        @test operator.stiffness_stresses == stresses
        @test operator.stiffness_factors == factors
        @test operator.stiffness_refresh_offsets == offsets
    end
end

@testset "stiffness reuse survives a shrink and grow chain" begin
    meshes = (
        BlockTreeMesh(base_mesh = (8, 8), block_size = (4, 4)),
        BlockTreeMesh(block_size = (2, 2)),
        BlockTreeMesh(
            block_size = (4, 4), periodic = (true, false), max_level = 1,
            refine = (key, origin, extent) ->
            key.level == 0 && key.i == 0 && key.j == 0,
        ),
        BlockTreeMesh(
            block_size = (4, 4), max_level = 3,
            refine = (key, origin, extent) -> key.level <= 2,
        ),
        BlockTreeMesh(block_size = (2, 2)),
        BlockTreeMesh(base_mesh = (8, 8), block_size = (4, 4)),
    )
    cache = TwoPhaseAMR.VariableDiffusionConstructionCache(Float64)
    previous = nothing
    for mesh in meshes
        viscosity = CellField(mesh, (x, y) -> 1 + 0.2sinpi(2x) * cospi(2y))
        reused = TwoPhaseAMR._staggered_diffusion_operator(
            mesh; viscosity, boundary = FreeSlip(), T = Float64,
            construction_cache = cache, reuse = previous,
        )
        fresh = TwoPhaseAMR._staggered_diffusion_operator(
            mesh; viscosity, boundary = FreeSlip(), T = Float64,
        )
        @test reused.stiffness_pointers == fresh.stiffness_pointers
        @test reused.stiffness_refresh_offsets == fresh.stiffness_refresh_offsets
        @test reused.stiffness_stresses == fresh.stiffness_stresses
        @test reused.stiffness.colptr == fresh.stiffness.colptr
        @test rowvals(reused.stiffness) == rowvals(fresh.stiffness)
        @test reused.strain.colptr == fresh.strain.colptr
        @test rowvals(reused.strain) == rowvals(fresh.strain)
        # The shared shear-template cache can move an earlier build's stencils
        # by a few ulps; that effect predates the reuse path.
        @test nonzeros(reused.stiffness) ≈
            nonzeros(fresh.stiffness) rtol = 1.0e-13 atol = 0.0
        @test nonzeros(reused.strain) ≈
            nonzeros(fresh.strain) rtol = 1.0e-13 atol = 0.0
        @test reused.stiffness_factors ≈
            fresh.stiffness_factors rtol = 1.0e-13 atol = 0.0
        previous = reused
    end
end

@testset "staggered diffusion stable core on uniform and AMR meshes" begin
    meshes = (
        BlockTreeMesh(block_size = (8, 8)),
        BlockTreeMesh(block_size = (8, 8), periodic = (false, false)),
        BlockTreeMesh(
            block_size = (4, 4), periodic = (false, false), max_level = 2,
            refine = (key, origin, extent) ->
            key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
        ),
    )
    for mesh in meshes, boundary in (NoSlip(), FreeSlip())
        operator = StaggeredDiffusionOperator(mesh; viscosity = 0.2, boundary)
        velocity = FaceVelocity(
            mesh, (x, y, time, axis) ->
            axis == 1 ? sinpi(2x) + 0.3cospi(2y) :
                cospi(2x) - 0.2sinpi(4y)
        )
        tendency = FaceVelocity(
            mesh, (x, y, time, axis) -> 0.0
        )
        diffusion!(tendency, velocity, operator)
        energy_rate = _face_component_energy(velocity, tendency, operator.x) +
            _face_component_energy(velocity, tendency, operator.y)
        @test energy_rate <= 256eps() * length(velocity.interior)
        @test operator.x.stiffness ≈ transpose(operator.x.stiffness)
        @test operator.y.stiffness ≈ transpose(operator.y.stiffness)
        @test minimum(eigvals(Symmetric(Matrix(operator.x.stiffness)))) >= -2.0e-12
        @test minimum(eigvals(Symmetric(Matrix(operator.y.stiffness)))) >= -2.0e-12
        @test stable_timestep(operator) > 0
    end
end

@testset "unembedded geometric diffusion matches fully open geometry (#80)" begin
    for periodic in ((true, true), (false, false), (true, false)),
            boundary in (NoSlip(), FreeSlip(), NoSlip((x, y, t, axis) -> x + 2y + t))
        mesh = BlockTreeMesh(
            base_mesh = (8, 8), block_size = (4, 4), extent = (8.0, 8.0),
            periodic = periodic, max_level = 1,
            refine = (key, origin, extent) -> key.level == 0 && key.i == 0 && key.j == 0,
        )
        geometry = GeometryField(mesh, (x, y) -> 1.0; lipschitz = 1)
        bare = StaggeredDiffusionOperator(mesh; boundary)
        open = StaggeredDiffusionOperator(geometry; boundary)
        velocity = FaceVelocity(
            mesh, (x, y, t, axis) -> sin(x + 0.3y + axis),
        )
        bare_rate, open_rate = FaceVelocity(mesh), FaceVelocity(mesh)
        diffusion!(bare_rate, velocity, bare; time = 0.4)
        diffusion!(open_rate, velocity, open; time = 0.4)
        for (a, b) in zip((bare.x, bare.y), (open.x, open.y))
            @test a.local_to_global == b.local_to_global
            @test a.volumes == b.volumes
            @test a.stiffness ≈ b.stiffness rtol = 1.0e-13 atol = 1.0e-13
        end
        @test bare_rate.interior ≈ open_rate.interior rtol = 1.0e-13 atol = 1.0e-13
    end
    mesh32 = BlockTreeMesh(
        base_mesh = (8, 8), block_size = (4, 4),
        origin = (0.0f0, 0.0f0), extent = (8.0f0, 8.0f0),
        periodic = (false, false), max_level = 1,
        refine = (key, origin, extent) -> key.level == 0 && key.i == 0 && key.j == 0,
    )
    bare32 = StaggeredDiffusionOperator(mesh32; viscosity = 1.0f0)
    open32 = StaggeredDiffusionOperator(
        GeometryField(mesh32, (x, y) -> 1.0f0; lipschitz = 1); viscosity = 1.0f0,
    )
    for (a, b) in zip((bare32.x, bare32.y), (open32.x, open32.y))
        @test eltype(a.stiffness) === Float32
        @test a.stiffness ≈ b.stiffness rtol = 64eps(Float32)
        @test TwoPhaseAMR._staggered_conservative_laplacian(a)
    end
end

@testset "cut-cell diffusion connections at an AMR transition (#78)" begin
    mesh = BlockTreeMesh(
        base_mesh = (8, 8), block_size = (4, 4), extent = (8.0, 8.0),
        periodic = (false, false), max_level = 1,
        refine = (key, origin, extent) ->
        key.level == 0 && key.i == 0 && key.j == 0,
    )
    geometry = GeometryField(mesh, (x, y) -> y - 2.3 - 0.1x; lipschitz = 2)
    operator = StaggeredDiffusionOperator(geometry; viscosity = 1.0)
    @test operator isa StaggeredDiffusionOperator

    # The assembled stiffness also contains stationary-wall penalties.
    # Remove precisely those penalties to test the conservative flux core:
    # a nonzero constant does not satisfy the stationary-wall condition.
    function conservative_core(component, boundary)
        core = Matrix(component.stiffness)
        for connection in component.wall_connections
            TwoPhaseAMR._include_wall_connection(boundary, connection) || continue
            i = connection.degree_of_freedom
            core[i, i] -= connection.conductance
        end
        return core
    end

    function check_core(component, boundary, rng)
        core = conservative_core(component, boundary)
        n = size(core, 1)
        # Matrix assembly and wall subtraction accumulate round-off; scale by
        # matrix magnitude and the maximum possible row length.
        tolerance = 128eps(Float64) * n * max(1.0, maximum(abs, core))
        @test maximum(abs, core * ones(n)) <= tolerance
        @test maximum(abs, core - transpose(core)) <= tolerance
        values = rand(rng, n) .- 0.5
        @test dot(values, core * values) >= -tolerance * dot(values, values)
        @test dot(values, core * values) > tolerance
        return core, tolerance
    end

    function cut_connections(geometry, axis)
        mesh = geometry.mesh
        schedule = TwoPhaseAMR._cut_cell_dual_schedule(geometry, Val(axis), false)
        incidences, _, _ = TwoPhaseAMR._component_incidences(
            mesh, Val(axis), Float64,
        )
        _, local_to_global = TwoPhaseAMR._component_local_to_global(mesh, axis)
        connections = TwoPhaseAMR._geometric_diffusion_connections(
            geometry, Val(axis), schedule, incidences, local_to_global,
        )
        return connections, local_to_global
    end

    rng = MersenneTwister(78)
    for component in (operator.x, operator.y)
        core, tolerance = check_core(component, operator.boundary, rng)
        @test minimum(eigvals(Symmetric(core))) >= -tolerance
        points = [mesh.faces[i].center for i in component.local_to_global]
        # On the uniform upper-right patch, these DOFs are more than one
        # coarse-cell spacing from the solid, domain walls and level change.
        # Do not demand pointwise linear exactness at cut or transition DOFs:
        # the symmetric conservative core need not have that property there.
        interior = findall(p -> 5.0 < p[1] < 7.0 && 5.0 < p[2] < 7.0, points)
        @test !isempty(interior)
        for axis in 1:2
            linear = [p[axis] for p in points]
            @test maximum(abs, (core * linear)[interior]) <=
                tolerance * maximum(abs, linear)
        end
    end

    # Check affine-exact gradients, including split and widened stencils at
    # the level interface but away from embedded and domain walls.
    for axis in 1:2
        connections, local_to_global = cut_connections(geometry, axis)
        adjacency = TwoPhaseAMR._connection_adjacency(
            length(local_to_global), connections,
        )
        checked = 0
        for connection in connections
            connection.refinement_interface || continue
            points = [
                mesh.faces[local_to_global[i]].center
                    for i in (connection.minus, connection.plus)
            ]
            all(p -> 0.8 < p[1] < 7.2 && 3.4 < p[2] < 7.2, points) || continue
            indices, stencil = TwoPhaseAMR._cut_cell_gradient_stencil(
                mesh, connection, local_to_global, adjacency,
            )
            @test abs(sum(stencil)) <= 1.0e-10
            for direction in 1:2
                coordinates = [
                    mesh.faces[local_to_global[i]].center[direction] for i in indices
                ]
                derivative = dot(stencil, coordinates)
                expected = direction == connection.axis ? 1.0 : 0.0
                @test abs(derivative) ≈ expected atol = 1.0e-10
            end
            checked += 1
        end
        @test checked > 0
    end

    open_geometry = GeometryField(mesh, (x, y) -> 1.0; lipschitz = 1)
    open_operator = StaggeredDiffusionOperator(open_geometry; viscosity = 1.0)

    # The fine-cell transverse face is blocked in both orientations, but the
    # coarse-cell half-strip remains open over x (or y) in [4.1, 5]. Simply
    # dropping the degenerate star would disconnect two active volumes.
    for axis in 1:2
        levelset = axis == 1 ?
            ((x, y) -> y - 2.55 + 0.5 * (x - 4)) :
            ((x, y) -> x - 2.55 + 0.5 * (y - 4))
        oblique = GeometryField(mesh, levelset; lipschitz = 2)
        oblique_operator = StaggeredDiffusionOperator(oblique; viscosity = 1.0)
        connections, local_to_global = cut_connections(oblique, axis)
        centres = [mesh.faces[i].center for i in local_to_global]
        first_point = axis == 1 ? (4.0, 2.25) : (2.25, 4.0)
        second_point = axis == 1 ? (4.0, 2.75) : (2.75, 4.0)
        first = findfirst(==(first_point), centres)
        second = findfirst(==(second_point), centres)
        @test !isnothing(first) && !isnothing(second)
        @test oblique.dual_volumes[axis][first] > 0
        @test oblique.dual_volumes[axis][second] > 0
        a, b = minmax(first, second)
        shared = filter(c -> c.minus == a && c.plus == b, connections)
        @test all(c -> c.axis == 3 - axis, shared)
        @test sum(c -> c.measure, shared; init = 0.0) ≈ 0.45 atol = 1.0e-12
        # For a completely open coarse cell the interface includes 1/2 of the
        # coarse width plus 1/2 of the fine width.
        open_connections, _ = cut_connections(open_geometry, axis)
        open_shared = filter(c -> c.minus == a && c.plus == b, open_connections)
        @test sum(c -> c.measure, open_shared; init = 0.0) ≈ 0.75 atol = 1.0e-12
        open_component = axis == 1 ? open_operator.x : open_operator.y
        open_core = conservative_core(open_component, open_operator.boundary)
        coarse_point = axis == 1 ? (5.0, 2.5) : (2.5, 5.0)
        coarse_index = findfirst(==(coarse_point), centres)
        corner_point = axis == 1 ? (3.5, 3.75) : (3.75, 3.5)
        corner_index = findfirst(==(corner_point), centres)
        for direction in 1:2
            linear = [p[direction] for p in centres]
            residual = open_core * linear
            @test maximum(abs, residual[[a, b, coarse_index, corner_index]]) <= 1.0e-11
        end
        # The transverse face measures around each open sibling must close:
        # a constant transverse gradient then has zero net dual flux.
        tangent = 3 - axis
        for index in (a, b)
            net_measure = 0.0
            first_moment = 0.0
            for connection in open_connections
                connection.axis == tangent || continue
                neighbour = if connection.minus == index
                    connection.plus
                elseif connection.plus == index
                    connection.minus
                else
                    continue
                end
                displacement = centres[neighbour][tangent] - centres[index][tangent]
                net_measure += sign(displacement) * connection.measure
                first_moment += abs(displacement) * connection.measure
            end
            @test abs(net_measure) <= 1.0e-12
            @test first_moment ≈ 2open_geometry.dual_volumes[axis][index] atol = 1.0e-12
        end

        # Both fine half-strips overlap the same coarse half-strip, including
        # the cross-side connection. The aligned pair additionally shares the
        # matching half-strip on the other side of its component face.
        coarse_point = axis == 1 ? (1.0, 4.5) : (4.5, 1.0)
        fine_points = axis == 1 ?
            ((1.0, 3.75), (1.5, 3.75)) :
            ((3.75, 1.0), (3.75, 1.5))
        coarse = findfirst(==(coarse_point), centres)
        for (fine_point, expected) in zip(fine_points, (0.5, 0.25))
            fine = findfirst(==(fine_point), centres)
            lo, hi = minmax(coarse, fine)
            overlap = sum(
                c.measure for c in open_connections
                    if c.minus == lo && c.plus == hi && c.axis == tangent;
                init = 0.0,
            )
            @test overlap ≈ expected atol = 1.0e-12
        end

        # A zero-distance flux unrelated to either sibling is still invalid;
        # it must not be silently mistaken for the normal-flux star.
        schedule = TwoPhaseAMR._cut_cell_dual_schedule(oblique, Val(axis), false)
        original = only(
            filter(
                f -> f.minus == a && f.plus == b && f.axis == axis, schedule.faces,
            )
        )
        malformed_face = TwoPhaseAMR.AMRDualFace(
            a, b, 1, 1, axis, true, original.minus_stencil, original.plus_stencil,
        )
        unrelated = findfirst(
            i -> mesh.faces[i].axis == axis && i != local_to_global[a] &&
                i != local_to_global[b],
            eachindex(mesh.faces),
        )
        malformed = TwoPhaseAMR.AMRDualAdvectionSchedule(
            [malformed_face], [unrelated], [1.0], schedule.volumes,
        )
        @test_throws ArgumentError TwoPhaseAMR._dual_diffusion_connections(
            mesh, malformed, local_to_global,
        )
        check_core(axis == 1 ? oblique_operator.x : oblique_operator.y, oblique_operator.boundary, rng)
    end

    blocked_connections, blocked_map = cut_connections(geometry, 1)
    blocked_centres = [mesh.faces[i].center for i in blocked_map]
    lower = findfirst(==((4.0, 2.25)), blocked_centres)
    upper = findfirst(==((4.0, 2.75)), blocked_centres)
    @test iszero(geometry.dual_volumes[1][lower])
    @test !any(
        c -> (c.minus, c.plus) == minmax(lower, upper), blocked_connections,
    )

    mesh32 = BlockTreeMesh(
        base_mesh = (8, 8), block_size = (4, 4), extent = (8.0f0, 8.0f0),
        origin = (0.0f0, 0.0f0), periodic = (false, false), max_level = 1,
        refine = (key, origin, extent) ->
        key.level == 0 && key.i == 0 && key.j == 0,
    )
    geometry32 = GeometryField(
        mesh32, (x, y) -> y - 2.3 - 0.1x; lipschitz = 2,
    )
    @test StaggeredDiffusionOperator(geometry32; viscosity = 1.0f0) isa StaggeredDiffusionOperator

    fixture_mesh = BlockTreeMesh(
        base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0),
        periodic = (false, false), max_level = 1,
        refine = (key, origin, extent) -> key.level == 0 && key.i >= 2,
    )
    fixture = GeometryField(
        fixture_mesh, (x, y) -> y - 0.9 - 0.1x; lipschitz = hypot(1, 0.1),
    )
    @test StaggeredDiffusionOperator(fixture; viscosity = 1.0) isa StaggeredDiffusionOperator

    # Discs centered near the level interface exercise many different cut
    # configurations without skipping failed geometry/operator constructions.
    for sample in 1:30
        cx = 3.0 + 2.0rand(rng)
        cy = 2.0 + 3.0rand(rng)
        radius = 0.8 + 1.4rand(rng)
        disc = GeometryField(
            mesh, (x, y) -> hypot(x - cx, y - cy) - radius; lipschitz = 2,
        )
        boundary = isodd(sample) ? NoSlip() : FreeSlip()
        disc_operator = StaggeredDiffusionOperator(disc; viscosity = 1.0, boundary)
        for component in (disc_operator.x, disc_operator.y)
            check_core(component, disc_operator.boundary, rng)
        end
    end

    # Randomize both the refinement pattern and the obstacle. Exact polygon
    # clipping avoids making this structural test depend on level-set
    # quadrature resolving a randomly near-tangent circular boundary.
    mesh_rng = MersenneTwister(87)
    for sample in 1:30
        refined = rand(mesh_rng, Bool, 2, 2)
        while all(refined) || !any(refined)
            refined = rand(mesh_rng, Bool, 2, 2)
        end
        sample_mesh = BlockTreeMesh(
            base_mesh = (8, 8), block_size = (4, 4), extent = (8.0, 8.0),
            periodic = (false, false), max_level = 1,
            refine = (key, origin, extent) ->
            key.level == 0 && refined[key.i + 1, key.j + 1],
        )
        cx = 3.0 + 2.0rand(mesh_rng)
        cy = 2.0 + 3.0rand(mesh_rng)
        radius = 0.8 + 1.4rand(mesh_rng)
        vertices = [
            (cx + radius * cospi(k / 6), cy + radius * sinpi(k / 6))
                for k in 0:11
        ]
        solid = InterfaceField(sample_mesh, PolygonIndicator(vertices))
        geometry = GeometryField(solid)
        boundary = isodd(sample) ? NoSlip() : FreeSlip()
        random_operator = StaggeredDiffusionOperator(geometry; viscosity = 1.0, boundary)
        for (axis, component) in enumerate((random_operator.x, random_operator.y))
            @test component.stiffness == transpose(component.stiffness)
            core, _ = check_core(component, random_operator.boundary, mesh_rng)
            @test core == transpose(core)
            spectrum = eigvals(Symmetric(core))
            @test minimum(spectrum) >= -1.0e-12 * max(maximum(abs, spectrum), eps())
            connections, local_to_global = cut_connections(geometry, axis)
            adjacency = TwoPhaseAMR._connection_adjacency(length(local_to_global), connections)
            stencils = TwoPhaseAMR._cut_cell_transverse_stencils(
                sample_mesh, connections, local_to_global,
                copy(component.wall_connections), boundary, Val(axis), Float64,
            )
            for connection in connections
                connection.axis == 3 - axis || continue
                first, second = (
                    sample_mesh.faces[local_to_global[i]].center
                        for i in (connection.minus, connection.plus)
                )
                TwoPhaseAMR._component_face_distance(sample_mesh, first, second, axis) >
                    256eps() * sample_mesh.extent[axis] || continue
                edge = (connection.minus, connection.plus)
                @test !isnothing(stencils) && haskey(stencils, edge)
                isnothing(stencils) && continue
                haskey(stencils, edge) || continue
                indices, stencil = stencils[edge]
                # The closure stays inside the two-ring face graph of the
                # contact, which is the neighbourhood the minimum-norm
                # derivative of an unembedded transition is built from, and
                # its coefficients sum to zero. An offset contact whose wall
                # value is unknown is differenced on a level-matched pair
                # instead of on the offset pair itself, so its own two
                # endpoints need not both appear in the row (#99).
                @test all(
                    index -> index in TwoPhaseAMR._nearby_degrees_of_freedom(
                        connection.minus, connection.plus, adjacency,
                    ), indices,
                )
                @test abs(sum(stencil)) <= 1.0e-10
            end
        end
    end

    # Exercise wrapped geometry and graph distances independently in both axes.
    for periodic in ((false, true), (true, false), (true, true))
        periodic_mesh = BlockTreeMesh(
            base_mesh = (8, 8), block_size = (4, 4), extent = (8.0, 8.0),
            periodic = periodic, max_level = 1,
            refine = (key, origin, extent) ->
            key.level == 0 && key.i == 0 && key.j == 0,
        )
        disc = GeometryField(
            periodic_mesh, (x, y) -> hypot(x - 4.0, y - 4.0) - 1.2;
            lipschitz = 2,
        )
        periodic_operator = StaggeredDiffusionOperator(disc; viscosity = 1.0)
        for component in (periodic_operator.x, periodic_operator.y)
            check_core(component, periodic_operator.boundary, rng)
        end
    end
end

@testset "cut transverse measures use overlapping traces (#87)" begin
    for refine_corner in (false, true)
        mesh = BlockTreeMesh(
            base_mesh = (8, 8), block_size = (4, 4), extent = (8.0, 8.0),
            periodic = (false, false), max_level = 1,
            refine = (key, origin, extent) ->
            refine_corner && key.level == 0 && key.i == 0 && key.j == 0,
        )
        geometry = GeometryField(
            mesh, (x, y) -> hypot(x - 5.8, y - 5.7) - 1.35; lipschitz = 1,
        )
        nx, ny = mesh.block_size
        for axis in 1:2
            schedule = TwoPhaseAMR._cut_cell_dual_schedule(geometry, Val(axis), false)
            incidences, _, _ = TwoPhaseAMR._component_incidences(mesh, Val(axis), Float64)
            _, local_to_global = TwoPhaseAMR._component_local_to_global(mesh, axis)
            connections = TwoPhaseAMR._geometric_diffusion_connections(
                geometry, Val(axis), schedule, incidences, local_to_global,
            )
            expected = Dict{Tuple{Int, Int}, Float64}()
            cut_edges = Set{Tuple{Int, Int}}()
            for (index, flux) in pairs(mesh.faces)
                flux.axis == 3 - axis || continue
                geometry.aperture[index] > 0 || continue
                minus = incidences[TwoPhaseAMR._cell_index(flux.minus, nx, ny)]
                plus = incidences[TwoPhaseAMR._cell_index(flux.plus, nx, ny)]
                mlo, mmid, mhi = TwoPhaseAMR._open_dual_trace(
                    geometry, Val(axis), flux.minus, 1, flux, Float64,
                )
                plo, pmid, phi = TwoPhaseAMR._open_dual_trace(
                    geometry, Val(axis), flux.plus, -1, flux, Float64,
                )
                for a in minus, b in plus
                    a.local_index == b.local_index && continue
                    geometry.dual_volumes[axis][a.local_index] > 0 || continue
                    geometry.dual_volumes[axis][b.local_index] > 0 || continue
                    aface, bface = mesh.faces[a.global_index], mesh.faces[b.global_index]
                    _, ahi = TwoPhaseAMR._face_local_tangent_interval(mesh, flux.minus, aface)
                    blo, _ = TwoPhaseAMR._face_local_tangent_interval(mesh, flux.plus, bface)
                    abs(ahi - 0.5) <= 256eps() || continue
                    abs(blo + 0.5) <= 256eps() || continue
                    alo, aend = a.side < 0 ? (mlo, mmid) : (mmid, mhi)
                    bstart, bend = b.side < 0 ? (plo, pmid) : (pmid, phi)
                    overlap = max(
                        0.0, min(aend, bend, flux.area / 2) -
                            max(alo, bstart, -flux.area / 2)
                    )
                    edge = minmax(a.local_index, b.local_index)
                    expected[edge] = get(expected, edge, 0.0) + overlap
                    0 < geometry.aperture[index] < 1 && push!(cut_edges, edge)
                end
            end
            actual = Dict{Tuple{Int, Int}, Float64}()
            for connection in connections
                connection.axis == 3 - axis || continue
                edge = (connection.minus, connection.plus)
                actual[edge] = get(actual, edge, 0.0) + connection.measure
            end
            checked = 0
            for edge in cut_edges
                points = (mesh.faces[local_to_global[i]].center for i in edge)
                # This region is on a regular coarse patch, away from the hanging
                # edge and domain boundary; split-side contributions cannot enter.
                all(p -> 4.25 < p[1] < 7.75 && 4.25 < p[2] < 7.75, points) || continue
                @test get(actual, edge, 0.0) ≈ expected[edge] atol = 1.0e-12
                checked += 1
            end
            @test checked > 0
        end
    end
end

@testset "wall conductance follows split dual strips (#87)" begin
    n = 8
    h = 1 / n
    for axis in 1:2
        mesh = BlockTreeMesh(
            base_mesh = (n, n), block_size = (4, 4),
            periodic = (false, false), max_level = 1,
            refine = (key, origin, extent) -> key.level == 0 && origin[axis] < 0.5,
        )
        geometry = GeometryField(mesh, (x, y) -> 1.0; lipschitz = 1)
        operator = StaggeredDiffusionOperator(geometry; boundary = NoSlip())
        component = axis == 1 ? operator.x : operator.y
        wall_axis = 3 - axis
        for side in (-1, 1)
            wall = side < 0 ? 0.0 : 1.0
            near = wall - side * h / 4
            far = wall - 3side * h / 4
            find_face(coordinate) = findfirst(
                index -> begin
                    point = mesh.faces[index].center
                    point[axis] == 0.5 && point[wall_axis] == coordinate
                end,
                component.local_to_global,
            )
            near_index = find_face(near)
            far_index = find_face(far)
            @test !isnothing(near_index)
            @test !isnothing(far_index)
            near_connections = filter(
                c -> c.degree_of_freedom == near_index &&
                    c.boundary_axis == wall_axis && c.boundary_side == side,
                component.wall_connections,
            )
            far_connections = filter(
                c -> c.degree_of_freedom == far_index &&
                    c.boundary_axis == wall_axis && c.boundary_side == side,
                component.wall_connections,
            )
            # One half-wall comes from the fine cell (conductance 1), the
            # other from the coarse cell (conductance 2). The adjacent split
            # segment does not touch the wall and gets no wall row.
            @test sort(getproperty.(near_connections, :conductance)) == [1.0, 2.0]
            @test isempty(far_connections)
        end
    end
end

@testset "wall-contact flux respects a linear normal profile (#87)" begin
    n = 16
    h = 1 / n
    mesh = BlockTreeMesh(
        base_mesh = (n, n), block_size = (4, 4),
        periodic = (false, false), max_level = 1,
        refine = (key, origin, extent) -> key.level == 0 && origin[1] < 0.5,
    )
    geometry = GeometryField(mesh, (x, y) -> 1.0; lipschitz = 1)
    for boundary in (NoSlip(), FreeSlip())
        component = StaggeredDiffusionOperator(geometry; boundary).y
        for side in (-1, 1)
            wall = side < 0 ? 0.0 : 1.0
            near = (0.5 - h / 4, wall - side * h / 2)
            far = (0.5 + h / 2, wall - side * h)
            points = [mesh.faces[index].center for index in component.local_to_global]
            indices = [findfirst(==(point), points) for point in (near, far)]
            profile = [side < 0 ? p[2] : 1 - p[2] for p in points]
            residual = component.stiffness * profile
            @test all(!isnothing, indices) &&
                maximum(abs, residual[Int.(indices)]) <= 1.0e-12
        end
    end
end

@testset "cut-cell Helmholtz convergence at walls and level transitions (#87)" begin
    function helmholtz_errors(n, boundary, quadrant)
        mesh = BlockTreeMesh(
            base_mesh = (n, n), block_size = (4, 4),
            periodic = (false, false), max_level = 1,
            refine = (key, origin, extent) -> key.level == 0 &&
                origin[1] < 0.5 && (!quadrant || origin[2] < 0.5),
        )
        # A constant positive level set exercises the cut-cell operator while
        # leaving the manufactured domain unobstructed.
        geometry = GeometryField(mesh, (x, y) -> 1.0; lipschitz = 1)
        diffusion = StaggeredDiffusionOperator(geometry; viscosity = 1.0, boundary)
        shift = 0.1
        errors = Float64[]
        for (axis, component) in enumerate((diffusion.x, diffusion.y))
            exact = [
                begin
                    x, y = mesh.faces[index].center
                    if boundary isa FreeSlip
                        axis == 1 ? sinpi(x) * cospi(2y) : cospi(2x) * sinpi(y)
                    else
                        sinpi(x) * sinpi(y)
                    end
                end for index in component.local_to_global
            ]
            # Both fields have homogeneous normal velocity at the walls.
            # Free slip has zero tangential normal derivative; no slip has
            # zero tangential velocity. Both have nonzero wall-normal
            # derivatives at the intersection with the level transition.
            laplacian_factor = boundary isa FreeSlip ? 5 : 2
            rhs = component.volumes .* ((1 + laplacian_factor * shift * π^2) .* exact)
            matrix = spdiagm(0 => component.volumes) + shift * component.stiffness
            solution = matrix \ rhs
            push!(errors, maximum(abs, solution - exact))
        end
        return errors
    end

    for boundary in (NoSlip(), FreeSlip()), quadrant in (false, true)
        errors = [helmholtz_errors(n, boundary, quadrant) for n in (16, 32, 64)]
        for axis in 1:2, level in 1:2
            @test log2(errors[level][axis] / errors[level + 1][axis]) >= 1.8
        end
    end
end

@testset "orphan transverse rows are linear-exact at open walls (#99)" begin
    # A transverse contact across a level transition that ends at a domain
    # wall has no mirrored partner, and an Outflow() or ZeroGradientVelocity()
    # side prescribes no wall value to close it against. The former two-point
    # difference carried the tangential derivative of the contact offset into
    # the normal gradient, which is order one at those edges; before the
    # level-matched closure these fixtures failed the three exactness checks
    # below at the contacts that end the transition next to the wall.
    function cut_transverse(mesh, geometry, operator, axis, boundary)
        schedule = TwoPhaseAMR._cut_cell_dual_schedule(geometry, Val(axis), false)
        incidences, _, _ = TwoPhaseAMR._component_incidences(
            mesh, Val(axis), Float64,
        )
        _, local_to_global = TwoPhaseAMR._component_local_to_global(mesh, axis)
        connections = TwoPhaseAMR._geometric_diffusion_connections(
            geometry, Val(axis), schedule, incidences, local_to_global,
        )
        component = axis == 1 ? operator.x : operator.y
        stencils = TwoPhaseAMR._cut_cell_transverse_stencils(
            mesh, connections, local_to_global, copy(component.wall_connections),
            boundary, Val(axis), Float64,
        )
        return connections, stencils, local_to_global
    end

    # The transverse contacts of a component are offset across a level
    # transition along the other direction, so an `:strip` variant refines
    # across the component axis while `:corner` refines both.
    function open_wall_operator(n, refined, axis, side)
        mesh = BlockTreeMesh(
            base_mesh = (n, n), block_size = (4, 4), extent = (1.0, 1.0),
            periodic = (false, false), max_level = 1,
            refine = (key, origin, extent) -> key.level == 0 && (
                refined === :corner ?
                    origin[1] < 0.5 && origin[2] < 0.5 :
                    origin[3 - axis] < 0.5
            ),
        )
        # The solid touches the left wall, so cut contacts, level transitions
        # and a non-Dirichlet wall meet.
        geometry = GeometryField(
            mesh, (x, y) -> hypot(x - 0.1, y - 0.55) - 0.24; lipschitz = 1,
        )
        boundary = VelocityBoundaryConditions(
            left = side, right = side, bottom = side, top = side,
        )
        return mesh, geometry,
            StaggeredDiffusionOperator(geometry; viscosity = 1.0, boundary), boundary
    end

    for n in (8, 16), side in (Outflow(), ZeroGradientVelocity()),
            refined in (:corner, :strip)
        for axis in 1:2
            mesh, geometry, operator, boundary = open_wall_operator(
                n, refined, axis, side,
            )
            connections, stencils, local_to_global = cut_transverse(
                mesh, geometry, operator, axis, boundary,
            )
            normal = 3 - axis
            skew = 0
            for connection in connections
                connection.axis == normal || continue
                first, second = (
                    mesh.faces[local_to_global[index]].center
                        for index in (connection.minus, connection.plus)
                )
                TwoPhaseAMR._component_face_distance(mesh, first, second, axis) >
                    256eps() * mesh.extent[axis] || continue
                skew += 1
                indices, coefficients = stencils[
                    (connection.minus, connection.plus),
                ]
                total, x_moment, y_moment = 0.0, 0.0, 0.0
                for (index, dof) in pairs(indices)
                    point = mesh.faces[local_to_global[dof]].center
                    total += coefficients[index]
                    x_moment += coefficients[index] * point[1]
                    y_moment += coefficients[index] * point[2]
                end
                @test abs(total) <= 1.0e-10
                @test abs(x_moment - (normal == 1 ? 1.0 : 0.0)) <= 1.0e-10
                @test abs(y_moment - (normal == 2 ? 1.0 : 0.0)) <= 1.0e-10
            end
            @test skew > 0
        end
    end

    # Every row keeps a non-negative energy and a zero coefficient sum: the
    # flux core alone, with all wall penalties removed, is symmetric positive
    # semidefinite and conserves a constant.
    rng = MersenneTwister(99)
    for side in (Outflow(), ZeroGradientVelocity()), refined in (:corner, :strip)
        mesh, geometry, operator, boundary = open_wall_operator(8, refined, 1, side)
        for component in (operator.x, operator.y)
            stiffness = component.stiffness
            scale = max(1.0, maximum(abs, nonzeros(stiffness)))
            @test stiffness == transpose(stiffness)
            core = Matrix(stiffness)
            for connection in component.wall_connections
                TwoPhaseAMR._include_wall_connection(boundary, connection) ||
                    continue
                index = connection.degree_of_freedom
                core[index, index] -= connection.conductance
            end
            n_rows = size(core, 1)
            tolerance = 128eps(Float64) * n_rows * scale
            @test maximum(abs, core * ones(n_rows)) <= tolerance
            values = rand(rng, n_rows) .- 0.5
            @test dot(values, core * values) >= -tolerance * dot(values, values)
            if side isa Outflow
                @test eigvals(Symmetric(core), 1:1)[1] >= -tolerance
            end
        end
    end
end

@testset "inflow sides close orphan contacts like no-slip sides (#99)" begin
    # The wall-normal velocity is prescribed data at an inflow side, so its
    # stiffness must equal the no-slip one: only the wall value, which never
    # enters the stiffness, differs.
    function stiffness_pattern(boundary)
        mesh = BlockTreeMesh(
            base_mesh = (16, 16), block_size = (4, 4), extent = (1.0, 1.0),
            periodic = (false, false), max_level = 1,
            refine = (key, origin, extent) ->
            key.level == 0 && origin[1] < 0.5 && origin[2] < 0.5,
        )
        geometry = GeometryField(
            mesh, (x, y) -> hypot(x - 0.15, y - 0.6) - 0.28; lipschitz = 1,
        )
        operator = StaggeredDiffusionOperator(geometry; viscosity = 1.0, boundary)
        return operator.x.stiffness, operator.y.stiffness
    end

    inflow_velocity = (x, y, time, axis) -> 1 + 0.3x - 0.2y
    inflow = VelocityBoundaryConditions(
        left = Inflow(inflow_velocity), right = Inflow(inflow_velocity),
        bottom = Inflow(inflow_velocity), top = Inflow(inflow_velocity),
    )
    inflow_x, inflow_y = stiffness_pattern(inflow)
    noslip_x, noslip_y = stiffness_pattern(NoSlip())
    @test inflow_x == noslip_x
    @test inflow_y == noslip_y
    # An open side closes no contact against the wall, so it must differ.
    open = VelocityBoundaryConditions(
        left = Outflow(), right = Outflow(), bottom = Outflow(), top = Outflow(),
    )
    open_x, open_y = stiffness_pattern(open)
    @test open_x != inflow_x
    @test open_y != inflow_y
end

@testset "orphan transverse rows converge on a manufactured field (#99)" begin
    # A transverse contact whose two dual faces sit on different tangential
    # levels must not be differenced directly: the row then reproduces
    # (offset/separation) times the tangential derivative instead of the
    # normal one, and that factor does not vanish under refinement. The
    # manufactured field (1 + 3x)(1 + 3y) is affine in the wall-normal
    # direction at every wall, so the wall terms and the wall-normal
    # differences of an all-open fixture are exact and the transverse rows are
    # the only error source left.
    exact(x, y) = (1 + 3x) * (1 + 3y)
    derivative(x, y, axis) = axis == 1 ? 3 * (1 + 3y) : 3 * (1 + 3x)

    function row_errors(n)
        mesh = BlockTreeMesh(
            base_mesh = (n, n), block_size = (4, 4), extent = (1.0, 1.0),
            periodic = (false, false), max_level = 1,
            refine = (key, origin, extent) -> key.level == 0 && origin[1] < 0.5,
        )
        geometry = GeometryField(mesh, (x, y) -> 1.0; lipschitz = 1)
        side = Outflow()
        boundary = VelocityBoundaryConditions(
            left = side, right = side, bottom = side, top = side,
        )
        operator = StaggeredDiffusionOperator(geometry; viscosity = 1.0, boundary)
        errors = Float64[]
        for axis in 1:2
            schedule = TwoPhaseAMR._cut_cell_dual_schedule(
                geometry, Val(axis), false,
            )
            incidences, _, _ = TwoPhaseAMR._component_incidences(
                mesh, Val(axis), Float64,
            )
            _, local_to_global = TwoPhaseAMR._component_local_to_global(mesh, axis)
            connections = TwoPhaseAMR._geometric_diffusion_connections(
                geometry, Val(axis), schedule, incidences, local_to_global,
            )
            component = axis == 1 ? operator.x : operator.y
            stencils = TwoPhaseAMR._cut_cell_transverse_stencils(
                mesh, connections, local_to_global,
                copy(component.wall_connections), boundary, Val(axis), Float64,
            )
            normal = 3 - axis
            for connection in connections
                connection.axis == normal || continue
                first, second = (
                    mesh.faces[local_to_global[index]].center
                        for index in (connection.minus, connection.plus)
                )
                TwoPhaseAMR._component_face_distance(
                    mesh, first, second, axis,
                ) > 256eps() * mesh.extent[axis] || continue
                indices, coefficients = stencils[
                    (connection.minus, connection.plus),
                ]
                value = 0.0
                for (index, dof) in pairs(indices)
                    point = mesh.faces[local_to_global[dof]].center
                    value += coefficients[index] * exact(point...)
                end
                centroid = (
                    (first[1] + second[1]) / 2, (first[2] + second[2]) / 2,
                )
                push!(errors, value - derivative(centroid..., normal))
            end
        end
        return errors
    end

    maximum_errors = Float64[]
    rms_errors = Float64[]
    for n in (16, 32, 64)
        errors = row_errors(n)
        @test !isempty(errors)
        push!(maximum_errors, maximum(abs, errors))
        push!(rms_errors, sqrt(sum(abs2, errors) / length(errors)))
    end
    for (name, values) in (("maximum", maximum_errors), ("rms", rms_errors))
        orders = log2.(values[1:2] ./ values[2:3])
        println("issue99 transverse row $name errors=$values eoc=$orders")
        @test all(diff(values) .< 0)
        # Directly differencing the offset pair left the largest row error at
        # 18.1, 18.6, 18.8 on these meshes (order zero) and the rms order at
        # 0.5; a linear-exact row is first order in both.
        @test orders[1] >= 0.9
        @test orders[2] >= 0.9
    end
end

@testset "composite diffusion solution convergence (#80)" begin
    function solution_errors(n, layout)
        mesh = BlockTreeMesh(
            base_mesh = (n, n), block_size = (4, 4), extent = (8.0, 8.0),
            periodic = (false, false), max_level = 1,
            refine = (key, origin, extent) ->
            key.level == 0 && key.i >= div(n, 8) &&
                (layout === :strip || key.j >= div(n, 8)),
        )
        operator = StaggeredDiffusionOperator(mesh; viscosity = 1.0)
        maximum_error, squared_error, total_volume = 0.0, 0.0, 0.0
        for component in (operator.x, operator.y)
            exact = [
                sinpi(mesh.faces[index].center[1] / 8) *
                    sinpi(mesh.faces[index].center[2] / 8)
                    for index in component.local_to_global
            ]
            volumes = component.volumes
            rhs = volumes .* ((1 + 2 * (pi / 8)^2) .* exact)
            numerical = (spdiagm(0 => volumes) + component.stiffness) \ rhs
            error = numerical - exact
            maximum_error = max(maximum_error, maximum(abs, error))
            squared_error += dot(volumes, abs2.(error))
            total_volume += sum(volumes)
        end
        return (; maximum = maximum_error, l2 = sqrt(squared_error / total_volume))
    end

    for layout in (:strip, :corner)
        errors = [solution_errors(n, layout) for n in (16, 32, 64)]
        for name in (:maximum, :l2)
            values = collect(getproperty.(errors, name))
            orders = log2.(values[1:2] ./ values[2:3])
            println("issue80 composite $layout $name errors=$values eoc=$orders")
            @test all(diff(values) .< 0)
            # Transition rows are not pointwise second order, so the observed
            # slopes depend on where the transition crosses the solution: they
            # approach two from below and are not uniformly above 1.8.
            @test orders[1] >= 1.7
            @test orders[2] >= 1.85
        end
    end
    # Transition rows can have O(1) pointwise truncation error. The
    # manufactured Helmholtz solution above checks the resulting global
    # accuracy instead of requiring the removed quadratic-exact row closure.
end

@testset "composite implicit diffusion is energy contractive (#80)" begin
    function energy_ratios(diffusion, periodic)
        worst_backward_euler, worst_midpoint = 0.0, 0.0
        for component in (diffusion.x, diffusion.y)
            volumes = component.volumes
            stiffness = component.stiffness
            @test stiffness == transpose(stiffness)
            spectrum = eigvals(Symmetric(Matrix(stiffness)))
            @test minimum(spectrum) >= -1.0e-12 * max(1.0, maximum(abs, spectrum))
            @test TwoPhaseAMR._staggered_conservative_laplacian(component)
            if periodic == (true, true)
                constant = ones(length(volumes))
                @test maximum(abs, stiffness * constant) <=
                    1.0e-12 * max(1.0, maximum(abs, nonzeros(stiffness)))
            end
            laplacian = Matrix(component.laplacian)
            identity = Matrix{Float64}(I, length(volumes), length(volumes))
            volume_half = Diagonal(sqrt.(volumes))
            volume_inverse_half = Diagonal(inv.(sqrt.(volumes)))
            for dt in (1.0e-3, 1.0e-1, 10.0, 1.0e3)
                backward = volume_half * (
                    (identity - dt * laplacian) \ volume_inverse_half
                )
                midpoint = volume_half * (
                    (identity - dt * laplacian / 2) \
                        ((identity + dt * laplacian / 2) * volume_inverse_half)
                )
                worst_backward_euler = max(worst_backward_euler, opnorm(backward)^2)
                worst_midpoint = max(worst_midpoint, opnorm(midpoint)^2)
            end
        end
        return (; worst_backward_euler, worst_midpoint)
    end

    function expect_cg(diffusion)
        solver = StaggeredEllipticSolver(
            StaggeredEllipticOperator(diffusion; diffusion = 0.1);
            preconditioner = nothing,
        )
        @test solver.x.krylov isa Krylov.CgWorkspace
        @test solver.y.krylov isa Krylov.CgWorkspace
        return nothing
    end

    function composite_mesh(; periodic, pattern, extent, max_level)
        return BlockTreeMesh(
            base_mesh = (8, 8), block_size = (4, 4), extent = extent,
            periodic = periodic, max_level = max_level,
            refine = (key, origin, extent) ->
            (
                key.level == 0 &&
                    (pattern == :strip ? key.i == 0 : key.i == 0 && key.j == 0)
            ) ||
                (pattern == :nested && key.level == 1 && key.i == 0 && key.j == 0),
        )
    end

    function split_mesh(periodic)
        return BlockTreeMesh(
            base_mesh = (8, 8), block_size = (4, 4), extent = (8.0, 8.0),
            periodic = periodic, max_level = 1,
            refine = (key, origin, extent) -> key.level == 0 && key.i == 0,
        )
    end

    maximum_backward_euler = 0.0
    maximum_midpoint = 0.0
    for periodic in ((true, true), (false, false), (true, false)),
            pattern in (:strip, :corner, :nested), boundary in (NoSlip(), FreeSlip())
        mesh = composite_mesh(
            periodic = periodic, pattern = pattern,
            extent = pattern == :nested ? (16.0, 16.0) : (8.0, 8.0),
            max_level = pattern == :nested ? 2 : 1,
        )
        diffusion = StaggeredDiffusionOperator(mesh; viscosity = 1.0, boundary)
        expect_cg(diffusion)
        ratios = energy_ratios(diffusion, periodic)
        maximum_backward_euler = max(maximum_backward_euler, ratios.worst_backward_euler)
        maximum_midpoint = max(maximum_midpoint, ratios.worst_midpoint)
    end

    # An embedded disc straddling the refinement interface exercises the
    # signed wall-contact corrections of _wall_contact_stencil!. A moving
    # NoSlip wall is a regression guard only: the wall value enters
    # _add_wall_source! and never the stiffness or the Laplacian, so its rows
    # must equal the stationary ones here.
    for boundary in (
            NoSlip(), FreeSlip(),
            NoSlip((x, y, t, axis) -> axis == 1 ? 1 + t : 0.0),
        )
        geometry = GeometryField(
            split_mesh((false, false)),
            (x, y) -> hypot(x - 4.0, y - 4.0) - 2.0; lipschitz = 1,
        )
        diffusion = StaggeredDiffusionOperator(
            geometry; viscosity = 1.0, boundary,
        )
        expect_cg(diffusion)
        ratios = energy_ratios(diffusion, (false, false))
        maximum_backward_euler = max(maximum_backward_euler, ratios.worst_backward_euler)
        maximum_midpoint = max(maximum_midpoint, ratios.worst_midpoint)
    end

    # Mixed side conditions: the split boundary strips and the sides without
    # a wall row must keep the stiffness positive semidefinite.
    for sides in (
            VelocityBoundaryConditions(
                left = Inflow((x, y, t, axis) -> axis == 1 ? y : 0.0),
                right = Outflow(), bottom = ZeroGradientVelocity(),
                top = NoSlip((x, y, t, axis) -> axis == 1 ? 1 + t : 0.0),
            ),
            VelocityBoundaryConditions(
                left = ZeroGradientVelocity(),
                right = Inflow((x, y, t, axis) -> axis == 1 ? 1 + t : 0.0),
                bottom = Outflow(), top = NoSlip(),
            ),
        )
        diffusion = StaggeredDiffusionOperator(
            split_mesh((false, false)); viscosity = 1.0, boundary = sides,
        )
        expect_cg(diffusion)
        ratios = energy_ratios(diffusion, (false, false))
        maximum_backward_euler = max(maximum_backward_euler, ratios.worst_backward_euler)
        maximum_midpoint = max(maximum_midpoint, ratios.worst_midpoint)
    end

    println(
        "issue80 worst energy ratio backward=$maximum_backward_euler ",
        "midpoint=$maximum_midpoint",
    )
    @test maximum_backward_euler <= 1 + 1.0e-12
    @test maximum_midpoint <= 1 + 1.0e-12
end

@testset "wall slip and no-slip closures" begin
    mesh = BlockTreeMesh(
        block_size = (12, 12), periodic = (true, false), extent = (1.0, 1.0)
    )
    couette = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? y : 0.0
    )
    tendency = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    moving_lid = NoSlip(
        (x, y, time, axis) ->
        axis == 1 && y > 0.5 ? 1.0 : 0.0
    )
    no_slip = StaggeredDiffusionOperator(mesh; boundary = moving_lid)
    diffusion!(tendency, couette, no_slip)
    @test maximum(abs, tendency.interior) <= 2.0e-12

    constant_tangent = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? 2.0 : 0.0
    )
    free_slip = StaggeredDiffusionOperator(mesh; boundary = FreeSlip())
    diffusion!(tendency, constant_tangent, free_slip)
    @test maximum(abs, tendency.interior) <= 2.0e-12

    stationary = StaggeredDiffusionOperator(mesh)
    diffusion!(tendency, constant_tangent, stationary)
    @test minimum(tendency.interior) < 0

    # Couette flow between a stationary floor and a moving lid is reproduced
    # on a composite mesh as well, including the split boundary strips of the
    # level transition.
    composite = BlockTreeMesh(
        base_mesh = (8, 8), block_size = (4, 4), extent = (1.0, 1.0),
        periodic = (true, false), max_level = 1,
        refine = (key, origin, extent) -> key.level == 0 && key.i == 0,
    )
    composite_couette = FaceVelocity(
        composite, (x, y, time, axis) -> axis == 1 ? y : 0.0
    )
    composite_tendency = FaceVelocity(composite, (x, y, time, axis) -> 0.0)
    composite_operator = StaggeredDiffusionOperator(composite; boundary = moving_lid)
    diffusion!(composite_tendency, composite_couette, composite_operator)
    @test maximum(abs, composite_tendency.interior) <= 2.0e-12
end

@testset "moving domain walls do not move an embedded disc" begin
    for T in (Float32, Float64)
        mesh = BlockTreeMesh(
            block_size = (16, 16), periodic = (false, false),
            origin = (zero(T), zero(T)), extent = (one(T), one(T)),
        )
        geometry = GeometryField(
            mesh, (x, y) -> hypot(x - T(0.5), y - T(0.5)) - T(0.18);
            lipschitz = one(T),
        )
        time = T(0.375)
        moving = NoSlip(
            (x, y, t, axis) -> (axis == 1 ? one(T) : -T(2)) * (one(T) + t),
        )
        sides = VelocityBoundaryConditions(
            left = moving, right = moving, bottom = moving, top = moving,
        )
        operator = StaggeredDiffusionOperator(geometry; boundary = moving)
        side_operator = StaggeredDiffusionOperator(geometry; boundary = sides)
        velocity = FaceVelocity(mesh)
        tendency = FaceVelocity(mesh)
        side_tendency = FaceVelocity(mesh)
        diffusion!(tendency, velocity, operator; time)
        diffusion!(side_tendency, velocity, side_operator; time)
        @test tendency.interior == side_tendency.interior
        @test all(iszero, tendency.boundary)

        wall_source = zeros(T, length(mesh.faces))
        for axis in 1:2
            component = axis == 1 ? operator.x : operator.y
            side_component = axis == 1 ? side_operator.x : side_operator.y
            @test component.stiffness == side_component.stiffness
            solid = filter(c -> iszero(c.boundary_axis), component.wall_connections)
            domain = filter(
                c -> !iszero(c.boundary_axis) && c.tangential,
                component.wall_connections,
            )
            @test any(c -> c.tangential, solid)
            @test !isempty(domain)
            @test all(
                c -> iszero(TwoPhaseAMR._wall_value(moving, c, time, axis, T)),
                solid,
            )
            solid_indices = [
                component.local_to_global[c.degree_of_freedom] for c in solid
            ]
            domain_indices = [
                component.local_to_global[c.degree_of_freedom] for c in domain
            ]
            @test isempty(intersect(solid_indices, domain_indices))
            @test all(i -> iszero(tendency.interior[i]), solid_indices)
            @test any(i -> !iszero(tendency.interior[i]), domain_indices)
            TwoPhaseAMR._add_wall_source!(
                wall_source, component, operator, one(T), time, axis,
            )
        end
        # The implicit midpoint source uses the same wall values as diffusion!.
        @test wall_source == tendency.interior

        # This callback is nonzero at the disc but zero on the domain boundary,
        # so both the explicit tendency and the implicit step must stay zero.
        disc_only = NoSlip(
            (x, y, t, axis) ->
            hypot(x - T(0.5), y - T(0.5)) < T(0.3) ? one(T) : zero(T),
        )
        @test disc_only.velocity(T(0.5), T(0.5), time, 1) == one(T)
        disc_operator = StaggeredDiffusionOperator(geometry; boundary = disc_only)
        diffusion!(tendency, velocity, disc_operator; time)
        @test all(iszero, tendency.interior)
        dt = T(0.1)
        solver = StaggeredEllipticSolver(
            StaggeredEllipticOperator(disc_operator; diffusion = dt / 2);
            preconditioner = nothing,
        )
        result = FaceVelocity(mesh)
        work = FaceVelocity(mesh)
        implicit_midpoint_diffusion!(
            result, velocity, velocity, disc_operator, dt, solver, work; time,
        )
        @test all(iszero, work.interior)
        @test all(iszero, result.interior)
        @test all(iszero, result.boundary)
    end
end

@testset "domain-wall diffusion uses only uncovered wall segments" begin
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (false, false))
    solid = InterfaceField(mesh)
    initialize!(solid, (1.0, 0.0), 0.45)
    geometry = GeometryField(solid)
    moving = NoSlip((x, y, time, axis) -> axis == 1 ? 1.0 : -2.0)
    sides = VelocityBoundaryConditions(
        left = moving, right = moving, bottom = moving, top = moving,
    )
    operator = StaggeredDiffusionOperator(geometry; boundary = moving)
    side_operator = StaggeredDiffusionOperator(geometry; boundary = sides)
    velocity = FaceVelocity(mesh)
    tendency = FaceVelocity(mesh)
    side_tendency = FaceVelocity(mesh)
    diffusion!(tendency, velocity, operator)
    diffusion!(side_tendency, velocity, side_operator)
    @test tendency.interior == side_tendency.interior
    for axis in 1:2
        covered = [
            i for (i, face) in pairs(mesh.faces)
                if face.axis == axis && iszero(geometry.face_dual_volumes[i])
        ]
        @test !isempty(covered)
        @test all(i -> iszero(tendency.interior[i]), covered)
        @test any(
            i -> mesh.faces[i].axis == axis && !iszero(tendency.interior[i]),
            eachindex(mesh.faces),
        )
    end

    # At x=0.5 the upper wall comprises a full half-cell on the right and
    # 0.05 of the left half-cell. Both component faces are 1/16 from the wall.
    top_connections = filter(
        c -> c.boundary_axis == 2 && c.boundary_side == 1 &&
            c.tangential && c.point[1] == 0.5,
        operator.x.wall_connections,
    )
    @test length(top_connections) == 2
    @test sort([c.conductance for c in top_connections]) ≈ [0.8, 1.0] atol = 1.0e-12

    # The normal component uses the entire 0.05-long open part of the cut
    # boundary face, with a full-cell distance of 1/8 to its interior face.
    bottom_normal = filter(
        c -> c.boundary_axis == 2 && c.boundary_side == -1 &&
            !c.tangential && c.point[1] == 0.4375,
        operator.y.wall_connections,
    )
    @test [c.conductance for c in bottom_normal] ≈ [0.4] atol = 1.0e-12
end

@testset "uniform periodic staggered Laplacian convergence" begin
    function error(nx)
        mesh = BlockTreeMesh(block_size = (nx, nx))
        velocity = FaceVelocity(
            mesh, (x, y, time, axis) -> axis == 1 ?
                sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y)
        )
        tendency = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
        operator = StaggeredDiffusionOperator(mesh; boundary = FreeSlip())
        diffusion!(tendency, velocity, operator)
        squared = 0.0
        volume = 0.0
        for component in (operator.x, operator.y)
            for local_index in eachindex(component.local_to_global)
                global_index = component.local_to_global[local_index]
                face = mesh.faces[global_index]
                x, y = face.center
                exact = -8pi^2 * (
                    face.axis == 1 ?
                        sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y)
                )
                squared += component.volumes[local_index] *
                    (tendency.interior[global_index] - exact)^2
                volume += component.volumes[local_index]
            end
        end
        return sqrt(squared / volume)
    end
    errors = error.((8, 16, 32))
    @test log2(errors[1] / errors[2]) > 1.9
    @test log2(errors[2] / errors[3]) > 1.9
end

@testset "implicit midpoint staggered elliptic solve" begin
    mesh = BlockTreeMesh(block_size = (16, 16))
    viscosity = 0.1
    diffusion_operator = StaggeredDiffusionOperator(
        mesh; viscosity, boundary = FreeSlip()
    )
    elliptic_operator = StaggeredEllipticOperator(
        diffusion_operator; diffusion = 0.05 * viscosity
    )
    elliptic_solver = StaggeredEllipticSolver(elliptic_operator)
    old_velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? sinpi(2x) : cospi(2y)
    )
    convected = copy(old_velocity)
    numerical = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    work = copy(numerical)
    dt = 0.1
    implicit_midpoint_diffusion!(
        numerical, convected, old_velocity, diffusion_operator,
        dt, elliptic_solver, work
    )

    dx = mesh.blocks[1].spacing[1]
    discrete_eigenvalue = -4sinpi(1 / 16)^2 / dx^2
    amplification = (1 + dt * viscosity * discrete_eigenvalue / 2) /
        (1 - dt * viscosity * discrete_eigenvalue / 2)
    for (index, face) in pairs(mesh.faces)
        initial = face.axis == 1 ? sinpi(2face.center[1]) :
            cospi(2face.center[2])
        @test numerical.interior[index] ≈ amplification * initial atol = 2.0e-13 rtol = 2.0e-13
    end
    @test elliptic_solver.operator.diffusion ≈ dt * viscosity / 2
    @test elliptic_solver isa StaggeredEllipticSolver
    @test elliptic_solver.x.krylov isa Krylov.CgWorkspace
    @test elliptic_solver.x.preconditioner isa
        TwoPhaseAMR.TreeStaggeredMultigridComponent
    @test elliptic_solver.y.preconditioner isa
        TwoPhaseAMR.TreeStaggeredMultigridComponent
    # A uniform root block has no block-tree level to prune.
    @test length(elliptic_solver.x.preconditioner.levels) == 1
    @test length(elliptic_solver.y.preconditioner.levels) == 1
    @test elliptic_solver.x.stats.converged
    @test elliptic_solver.y.stats.converged
    @test elliptic_solver.x.stats.initial_linf >=
        elliptic_solver.x.stats.final_linf
    @test elliptic_solver.x.stats.target_linf >=
        sqrt(eps()) * elliptic_solver.x.stats.residual_scale

    x_hierarchy = elliptic_solver.x.preconditioner
    y_hierarchy = elliptic_solver.y.preconditioner
    implicit_midpoint_diffusion!(
        numerical, convected, old_velocity, diffusion_operator,
        dt, elliptic_solver, work
    )
    @test elliptic_solver.x.preconditioner === x_hierarchy
    @test elliptic_solver.y.preconditioner === y_hierarchy
end

@testset "staggered base-mesh multigrid" begin
    for periodic in ((true, true), (true, false), (false, false))
        mesh = BlockTreeMesh(
            base_mesh = (16, 16), block_size = (4, 4),
            periodic = periodic, max_level = 1,
        )
        diffusion = StaggeredDiffusionOperator(
            mesh; viscosity = 0.2, boundary = FreeSlip(),
        )
        operator = StaggeredEllipticOperator(diffusion; diffusion = 0.03)
        solver = StaggeredEllipticSolver(operator; coarse_size = 16)
        hierarchy = solver.x.preconditioner
        @test [op.mesh.base_mesh for op in hierarchy.diffusion_operators] ==
            [(16, 16), (8, 8), (4, 4)]
        @test all(op -> op.mesh.periodic == periodic, hierarchy.diffusion_operators)
        @test length(solver.y.preconditioner.levels) == 3
        for index in 1:2
            fine = hierarchy.diffusion_operators[index].mesh
            coarse = hierarchy.diffusion_operators[index + 1].mesh
            transfer = hierarchy.face_prolongations[index]
            @test size(transfer) == (length(fine.faces), length(coarse.faces))
            if periodic == (true, true)
                @test transfer * ones(length(coarse.faces)) ≈
                    ones(length(fine.faces)) atol = 64eps()
                @test transpose(transfer) * fine.face_dual_volumes ≈
                    coarse.face_dual_volumes atol = 64eps()
            end
        end
        source = FaceVelocity(
            mesh, (x, y, time, axis) -> axis == 1 ?
                sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y),
        )
        destination = FaceVelocity(mesh)
        stats = solve!(destination, solver, source; tolerance = 1.0e-9, max_cycles = 100)
        @test stats.converged
        if periodic == (true, true)
            old_mesh = last(hierarchy.diffusion_operators).mesh
            old = TwoPhaseAMR.TreeStaggeredGeometricMultigridPreconditioner(
                operator, solver.x.preconditioner, solver.y.preconditioner,
            )
            marker = CellField(mesh)
            adaptation = AMRWorkspace(mesh, marker)
            labels = [
                block.key == BlockKey(0, 0, 0) ? :refine : :keep
                    for block in mesh.blocks
            ]
            @test adapt!(adaptation, labels; step = 1, force = true).changed
            new_diffusion = StaggeredDiffusionOperator(
                mesh; viscosity = 0.2, boundary = FreeSlip(),
            )
            new_operator = StaggeredEllipticOperator(new_diffusion; diffusion = 0.03)
            rebuilt = TwoPhaseAMR.TreeStaggeredGeometricMultigridPreconditioner(
                new_operator; coarse_size = 16, reuse = old,
            )
            @test rebuilt.x.reused_levels > 0
            @test rebuilt.y.reused_levels > 0
            @test last(rebuilt.x.diffusion_operators).mesh === old_mesh
        end
    end
    # Physical walls remove staggered unknowns. The base stopping test is
    # still in cells, not in component-face counts.
    wall_mesh = BlockTreeMesh(
        base_mesh = (8, 8), block_size = (4, 4), periodic = (false, false),
    )
    wall_diffusion = StaggeredDiffusionOperator(
        wall_mesh; viscosity = 0.2, boundary = FreeSlip(),
    )
    wall_operator = StaggeredEllipticOperator(wall_diffusion; diffusion = 0.03)
    @test length(wall_operator.x.volumes) < 60 < ncells(wall_mesh)
    wall_solver = StaggeredEllipticSolver(wall_operator; coarse_size = 60)
    @test [op.mesh.base_mesh for op in wall_solver.x.preconditioner.diffusion_operators] ==
        [(8, 8), (4, 4)]
    @test length(wall_solver.y.preconditioner.levels) == 2

    periodic_mesh = BlockTreeMesh(
        base_mesh = (4, 4), block_size = (2, 2), periodic = (true, true),
    )
    periodic_diffusion = StaggeredDiffusionOperator(
        periodic_mesh; viscosity = 0.2, boundary = FreeSlip(),
    )
    periodic_solver = StaggeredEllipticSolver(
        StaggeredEllipticOperator(periodic_diffusion; diffusion = 0.03);
        coarse_size = 1,
    )
    @test length(periodic_solver.x.preconditioner.levels) == 1
    @test length(periodic_solver.y.preconditioner.levels) == 1
end

@testset "dimensionless diffusion tolerance and failure report" begin
    mesh = BlockTreeMesh(block_size = (8, 8))
    diffusion = StaggeredDiffusionOperator(
        mesh; viscosity = 0.1, boundary = FreeSlip(),
    )
    operator = StaggeredEllipticOperator(diffusion; diffusion = 0.01)
    solver = StaggeredEllipticSolver(
        operator; multigrid = MultigridOptions(coarse_size = 8),
    )
    right_hand_side = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ? sinpi(2y) : cospi(2x),
    )
    solution = FaceVelocity(mesh)
    tolerance = 1.0e-3
    stats = solve!(
        solution, solver, right_hand_side; tolerance, max_cycles = 0,
    )
    @test !stats.converged
    @test stats.x.target_linf ≈ tolerance * stats.x.residual_scale
    @test stats.y.target_linf ≈ tolerance * stats.y.residual_scale

    destination = FaceVelocity(mesh)
    work = FaceVelocity(mesh)
    exception = try
        implicit_midpoint_diffusion!(
            destination, right_hand_side, right_hand_side,
            diffusion, 0.1, solver, work;
            tolerance = 1.0e-12, max_cycles = 0,
        )
        nothing
    catch caught
        caught
    end
    @test exception isa ErrorException
    message = sprint(showerror, exception)
    for field in (
            "tolerance=", "dimensionless_residual=", "max_cycles=0",
            "cycles=", "initial_l2=", "initial_linf=", "final_l2=",
            "final_linf=", "target_linf=", "residual_scale=",
            "removed_rhs_mean=", "method=", "preconditioner=",
        )
        @test occursin(field, message)
    end
end

@testset "external BiCGSTAB and staggered preconditioners" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 0),
    )
    diffusion = StaggeredDiffusionOperator(
        mesh; viscosity = 0.1, boundary = FreeSlip()
    )
    operator = StaggeredEllipticOperator(diffusion; diffusion = 0.1)
    solver = StaggeredEllipticSolver(
        operator; coarse_size = 16, krylov = Krylov.BicgstabWorkspace,
    )
    right_hand_side = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y)
    )
    solution = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    stats = solve!(
        solution, solver, right_hand_side;
        tolerance = 1.0e-10, max_cycles = 80
    )
    @test stats.converged
    @test stats.x.method === :bicgstab
    @test stats.y.method === :bicgstab
    @test stats.x.preconditioner === :tree_staggered_geometric_multigrid
    @test stats.y.preconditioner === :tree_staggered_geometric_multigrid
    @test stats.x.cycles < 80
    @test stats.y.cycles < 80
    @test length(solver.x.preconditioner.levels) > 2
    @test length(solver.y.preconditioner.levels) > 2
    for (hierarchy, component, axis) in (
            (solver.x.preconditioner, solver.operator.x, 1),
            (solver.y.preconditioner, solver.operator.y, 2),
        )
        levels = hierarchy.levels
        @test first(levels).matrix === component.matrix
        @test all(
            level -> level.prolongation isa SparseMatrixCSC,
            levels
        )
        @test all(
            level -> nnz(level.prolongation) <=
                2size(level.prolongation, 1), levels
        )
        @test all(
            levels[index].matrix === hierarchy.operators[index].matrix
                for index in eachindex(levels)
        )
        for index in 2:length(levels)
            rediscretized = StaggeredEllipticOperator(
                hierarchy.diffusion_operators[index];
                reaction = operator.reaction,
                diffusion = operator.diffusion,
            )
            reference = axis == 1 ? rediscretized.x : rediscretized.y
            @test levels[index].matrix == reference.matrix
        end
        @test all(diff(length.(getproperty.(hierarchy.operators, :volumes))) .< 0)
        @test nblocks(last(hierarchy.diffusion_operators).mesh) == 1

        transfer = first(levels).prolongation
        fine = collect(range(0.1, 1.0; length = size(transfer, 1)))
        coarse = collect(range(-0.2, 0.7; length = size(transfer, 2)))
        prolonged = similar(fine)
        restricted = similar(coarse)
        TwoPhaseAMR._prolongate!(prolonged, first(levels), coarse)
        TwoPhaseAMR._restrict!(restricted, transfer, fine)
        @test dot(fine, prolonged) ≈ dot(restricted, coarse)
    end
    residuals = residual_norms(solver)
    @test residuals.x[2] < 1.0e-9
    @test residuals.y[2] < 1.0e-9

    plain_solution = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    plain_solver = StaggeredEllipticSolver(
        operator; preconditioner = nothing
    )
    plain_stats = solve!(
        plain_solution, plain_solver, right_hand_side;
        tolerance = 1.0e-10, max_cycles = 500
    )
    @test plain_stats.converged
    @test plain_stats.x.preconditioner === :none
    @test maximum(abs, solution.interior - plain_solution.interior) < 2.0e-9

    for (method, workspace) in (
            (:gmres, Krylov.GmresWorkspace),
            (:fgmres, Krylov.FgmresWorkspace),
        )
        gmres_solution = FaceVelocity(
            mesh, (x, y, time, axis) -> 0.0
        )
        gmres_solver = StaggeredEllipticSolver(
            operator;
            krylov = (matrix, rhs) -> workspace(
                matrix, rhs; memory = 20
            ),
            krylov_options = (; restart = true, reorthogonalization = true),
            coarse_size = 16
        )
        gmres_stats = solve!(
            gmres_solution, gmres_solver, right_hand_side;
            tolerance = 1.0e-10, max_cycles = 80
        )
        @test gmres_stats.converged
        @test gmres_stats.x.method === method
        @test maximum(
            abs,
            gmres_solution.interior - plain_solution.interior
        ) < 2.0e-9
    end

    @test_throws ArgumentError StaggeredEllipticSolver(
        operator; preconditioner = :unsupported
    )
    @test_throws ArgumentError TreeStaggeredGeometricMultigridPreconditioner(
        operator; pruning = :unsupported
    )
end

@testset "boundary extrapolation cache follows mesh identity and epoch" begin
    boundary = VelocityBoundaryConditions(
        left = Inflow((x, y, t, axis) -> axis == 1 ? 1 : 0),
        right = Outflow(), bottom = FreeSlip(), top = FreeSlip(),
    )
    mesh = BlockTreeMesh(block_size = (4, 4), periodic = (false, false))
    sources = TwoPhaseAMR._boundary_extrapolation_schedule!(boundary, mesh)
    @test sources === TwoPhaseAMR._boundary_extrapolation_schedule!(boundary, mesh)
    @test all(index isa Int for face_sources in sources for index in face_sources)
    generation = mesh.topology_cache.generation
    TwoPhaseAMR._invalidate_mesh_topology_cache!(mesh)
    refreshed = TwoPhaseAMR._boundary_extrapolation_schedule!(boundary, mesh)
    @test refreshed !== sources
    @test refreshed == sources
    @test boundary.cache[].generation == generation + 1

    first_mesh = BlockTreeMesh(
        block_size = (4, 4), periodic = (false, false),
        extent = (2.0, 1.0)
    )
    second_mesh = BlockTreeMesh(
        block_size = (4, 4), periodic = (false, false),
        extent = (3.0, 1.0)
    )
    @test first_mesh.topology_cache.generation == second_mesh.topology_cache.generation
    first_sources = TwoPhaseAMR._boundary_extrapolation_schedule!(boundary, first_mesh)
    second_sources = TwoPhaseAMR._boundary_extrapolation_schedule!(boundary, second_mesh)
    @test first_sources !== second_sources
    @test boundary.cache[].topology_identity === second_mesh.topology_cache.identity

    mesh32 = BlockTreeMesh(
        block_size = (4, 4), periodic = (false, false),
        origin = (0.0f0, 0.0f0), extent = (2.0f0, 1.0f0)
    )
    other = TwoPhaseAMR._boundary_extrapolation_schedule!(boundary, mesh32)
    @test other !== second_sources
    @test boundary.cache[].topology_identity === mesh32.topology_cache.identity
    velocity = FaceVelocity(mesh32, (x, y, t, axis) -> axis == 1 ? x + 1 : 0)
    enforce_velocity_boundary!(velocity, boundary)
    net_flux = sum(
        face.side * face.area * velocity.boundary[index]
            for (index, face) in pairs(mesh32.boundary_faces)
    )
    @test abs(net_flux) <= 256eps(Float32)
end

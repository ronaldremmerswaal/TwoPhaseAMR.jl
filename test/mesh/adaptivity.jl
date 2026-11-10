import LinearAlgebra: ldiv!

function _maximum_divergence(mesh, velocity)
    divergence = CellField(mesh)
    divergence!(divergence, velocity)
    return maximum(
        maximum(abs, interior(divergence, block))
            for block in eachindex(mesh.blocks)
    )
end

function _component_momenta(velocity)
    x = FaceField{1}(velocity)
    y = FaceField{2}(velocity)
    return (
        integrate(x, StaggeredAdvectionWorkspace(x)),
        integrate(y, StaggeredAdvectionWorkspace(y)),
    )
end

function _face_momentum(field::FaceField)
    return integrate(field, StaggeredAdvectionWorkspace(field))
end

@testset "mesh face dual volumes follow AMR topology" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 1,
        adaptation = MeshAdaptationOptions(coarsen_delay = 0),
    )
    scalar = CellField(mesh, (x, y) -> 1 + sinpi(4x) * cospi(4y))
    workspace = AMRWorkspace(mesh, scalar)
    indicator = InterpolationErrorIndicator(
        scalar; E_refine = 0.01, E_coarsen = 0.005,
    )
    function check_volumes()
        @test length(mesh.face_dual_volumes) == length(mesh.faces)
        for (index, face) in pairs(mesh.faces)
            minus = mesh.blocks[face.minus.block].spacing[face.axis]
            plus = mesh.blocks[face.plus.block].spacing[face.axis]
            @test mesh.face_dual_volumes[index] ==
                face.area * (minus + plus) / 2
        end
    end
    check_volumes()
    @test adapt!(workspace, indicator; step = 1, force = true).changed
    check_volumes()
    initialize!(scalar, (x, y) -> 1.0)
    @test adapt!(workspace, indicator; step = 2, force = true).changed
    check_volumes()
end

@testset "interpolation-error refinement indicators" begin
    mesh = BlockTreeMesh(block_size = (8, 8), max_level = 2)
    constant = CellField(mesh; init = 2.0)
    oscillatory = CellField(mesh)
    initialize!(oscillatory, (x, y) -> 1 + 0.5sinpi(4x) * cospi(4y))
    workspace = AMRWorkspace(mesh, constant, oscillatory)

    coarsen = InterpolationErrorIndicator(
        constant; E_refine = 0.2, E_coarsen = 0.1
    )
    keep = InterpolationErrorIndicator(
        constant; E_refine = 0.2, E_coarsen = 0.0
    )
    refine = InterpolationErrorIndicator(
        oscillatory; E_refine = 0.01, E_coarsen = 0.005
    )
    @test refinement_labels!(workspace, coarsen) == [:coarsen]
    @test refinement_labels!(workspace, (coarsen, keep)) == [:keep]
    @test refinement_labels!(workspace, (coarsen, keep, refine)) == [:refine]

    @test_throws ArgumentError InterpolationErrorIndicator(
        constant; E_refine = 0.1, E_coarsen = 0.1
    )
    @test_throws ArgumentError InterpolationErrorIndicator(
        constant; E_refine = 0.1, E_coarsen = -0.1
    )
end

@testset "indicator spatial regions" begin
    mesh = BlockTreeMesh(base_mesh = (8, 32), block_size = (8, 8), max_level = 1)
    interface = InterfaceField(mesh, (x, y) -> x - 0.5; lipschitz = 1)
    geometry = GeometryField(mesh, (x, y) -> x - 0.5; lipschitz = 1)
    scalar = CellField(mesh; init = 1.0)
    velocity = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    constructors = (
        (; kw...) -> InterfaceRefinementIndicator(interface; level = 1, kw...),
        (; kw...) -> GeometryRefinementIndicator(geometry; level = 1, kw...),
        (; kw...) -> CurvatureRefinementIndicator(interface; minimum_level = 1, kw...),
        (; kw...) -> VorticityRefinementIndicator(velocity; refine_tolerance = 0.5, minimum_level = 1, kw...),
        (; kw...) -> InterpolationErrorIndicator(scalar; E_refine = 0.2, E_coarsen = 0.1, minimum_level = 1, kw...),
    )
    labels = Vector{Symbol}(undef, nblocks(mesh))
    for make in constructors
        default = copy(refinement_labels!(labels, make()))
        explicit = copy(refinement_labels!(labels, make(; xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf)))
        @test isequal(default, explicit)
        restricted = copy(refinement_labels!(labels, make(; ymin = 0.5)))
        for (b, block) in pairs(mesh.blocks)
            @test restricted[b] == (block.origin[2] + block.extent[2] < 0.5 ? :coarsen : default[b])
        end
        for bounds in (
                (; xmin = NaN), (; xmax = NaN), (; ymin = NaN),
                (; ymax = NaN), (; xmin = 1, xmax = 1),
                (; ymin = 2, ymax = 1), (; xmin = Inf),
            )
            @test_throws ArgumentError make(; bounds...)
        end
    end
    # The root straddles the region boundary and must be considered inside.
    straddling = BlockTreeMesh(block_size = (8, 8), max_level = 1)
    crossing = InterfaceField(straddling, (x, y) -> x - 0.5; lipschitz = 1)
    @test refinement_labels!([:keep], InterfaceRefinementIndicator(crossing; level = 1, ymin = 0.5)) == [:refine]

    indicator = InterfaceRefinementIndicator(interface; level = 1, ymin = 0.5)
    workspace = AMRWorkspace(mesh, interface)
    @test adapt!(workspace, indicator; step = 1, force = true).changed
    # The first root is strictly outside. Its neighbour touches y=0.5 and
    # may create children below the cutoff when that root is refined.
    @test any(block -> block.origin[2] == 0 && block.key.level == 0, mesh.blocks)
    @test any(block -> block.key.level == 1 && block.origin[2] >= 0.5, mesh.blocks)
end

@testset "vorticity refinement indicator" begin
    mesh = BlockTreeMesh(block_size = (8, 8), max_level = 2)
    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ?
            sinpi(2y) : -sinpi(2x)
    )
    indicator = VorticityRefinementIndicator(
        velocity; refine_tolerance = 0.5,
        coarsen_tolerance = 0.1, maximum_level = 1
    )
    workspace = AMRWorkspace(mesh, velocity)
    @test refinement_labels!(workspace, indicator) == [:refine]
    @test adapt!(workspace, indicator; step = 1, force = true).changed
    @test all(block -> block.key.level == 1, mesh.blocks)
    fill!(velocity.interior, 0)
    fill!(velocity.boundary, 0)
    @test refinement_labels!(workspace, indicator) ==
        fill(:coarsen, nblocks(mesh))
    @test_throws ArgumentError VorticityRefinementIndicator(
        velocity; refine_tolerance = 0.2, coarsen_tolerance = 0.1
    )
end

@testset "AMR criterion time windows and level bounds" begin
    mesh = BlockTreeMesh(block_size = (8, 8), max_level = 2)
    scalar = CellField(
        mesh, (x, y) -> 1 + 0.5sinpi(4x) * cospi(4y)
    )
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.2;
        lipschitz = 1
    )
    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ?
            sinpi(2y) : -sinpi(2x)
    )
    bounded_indicators = (
        InterpolationErrorIndicator(
            scalar; E_refine = 0.01, E_coarsen = 0.005,
            minimum_level = 1, maximum_level = 1,
            t_init = 1, t_end = 2
        ),
        CurvatureRefinementIndicator(
            interface; minimum_level = 1, maximum_level = 1,
            t_init = 1, t_end = 2
        ),
        VorticityRefinementIndicator(
            velocity; refine_tolerance = 0.5,
            minimum_level = 1, maximum_level = 1,
            t_init = 1, t_end = 2
        ),
    )
    interface_indicator = InterfaceRefinementIndicator(
        interface; level = 1, t_init = 1, t_end = 2
    )
    labels = Vector{Symbol}(undef, nblocks(mesh))
    for indicator in bounded_indicators
        @test indicator.minimum_level == 1
        @test indicator.maximum_level == 1
        @test indicator.t_init == 1
        @test indicator.t_end == 2
        @test refinement_labels!(labels, indicator; time = 0) == [:coarsen]
        @test refinement_labels!(labels, indicator; time = 1) == [:refine]
        @test refinement_labels!(labels, indicator; time = 3) == [:coarsen]
    end
    @test interface_indicator.level == 1
    @test interface_indicator.t_init == 1
    @test interface_indicator.t_end == 2
    @test refinement_labels!(labels, interface_indicator; time = 0) == [:coarsen]
    @test refinement_labels!(labels, interface_indicator; time = 1) == [:refine]
    @test refinement_labels!(labels, interface_indicator; time = 3) == [:coarsen]

    bounded_defaults = (
        InterpolationErrorIndicator(
            scalar; E_refine = 0.1, E_coarsen = 0.05
        ),
        CurvatureRefinementIndicator(interface),
        VorticityRefinementIndicator(
            velocity; refine_tolerance = 0.5
        ),
    )
    for indicator in bounded_defaults
        @test indicator.minimum_level == 0
        @test indicator.maximum_level == mesh.max_level
        @test indicator.t_init == -Inf
        @test indicator.t_end == Inf
    end
    interface_default = InterfaceRefinementIndicator(
        interface; level = mesh.max_level
    )
    @test interface_default.level == mesh.max_level
    @test interface_default.t_init == -Inf
    @test interface_default.t_end == Inf

    workspace = AMRWorkspace(mesh, scalar)
    @test !adapt!(
        workspace, bounded_indicators[1]; step = 1, time = 0, force = true
    ).changed
    @test adapt!(
        workspace, bounded_indicators[1]; step = 2, time = 1, force = true
    ).changed
    @test all(block -> block.key.level == 1, mesh.blocks)

    @test_throws ArgumentError InterpolationErrorIndicator(
        scalar; E_refine = 0.1, E_coarsen = 0.05,
        minimum_level = 2, maximum_level = 1
    )
    @test_throws ArgumentError InterfaceRefinementIndicator(
        interface; level = mesh.max_level + 1
    )
    @test_throws ArgumentError CurvatureRefinementIndicator(
        interface; t_init = 2, t_end = 1
    )
    @test_throws ArgumentError VorticityRefinementIndicator(
        velocity; refine_tolerance = 0.5, t_init = NaN
    )
end

@testset "Fixed bands refine the interface and release distant blocks (#117)" begin
    for kind in (:interface, :geometry), buffer in (1, 2)
        @testset "$kind buffer=$buffer" begin
            mesh = BlockTreeMesh(
                base_mesh = (16, 16), block_size = (4, 4), halo = 2,
                max_level = 2, periodic = (false, false),
                adaptation = MeshAdaptationOptions(coarsen_delay = 0),
                refine = (key, origin, extent) -> key.level == 0,
            )
            interface = InterfaceField(mesh; contact_angle = nothing)
            initialize!(interface, (1.0, 0.0), 0.5)
            geometry = kind === :geometry ? GeometryField(interface) : nothing
            indicator = kind === :geometry ?
                GeometryRefinementIndicator(geometry; level = 2, buffer_cells = buffer) :
                InterfaceRefinementIndicator(interface; level = 2, buffer_cells = buffer)
            labels = Vector{Symbol}(undef, nblocks(mesh))
            refinement_labels!(labels, indicator)
            distant = [
                b for (b, block) in pairs(mesh.blocks)
                    if block.origin[1] + block.extent[1] <= 0.125
            ]
            @test !isempty(distant)
            @test all(b -> labels[b] === :coarsen, distant)
            workspace = AMRWorkspace(mesh, kind === :geometry ? geometry : interface)
            stats = adapt!(workspace, indicator; step = 1, force = true)
            @test stats.refined_blocks > 0
            @test stats.coarsened_families > 0
        end
    end
end

# Geometric oracle for a target-level band, independent of the indicator's
# labels and its treatment of coarse/fine halos. A block intersecting this
# subset of the buffer must be at the requested level.
function _intersects_static_band(block, shape, position, radius, width)
    xlo, ylo = block.origin
    xhi, yhi = block.origin .+ block.extent
    if shape === :plane
        return xlo < position[1] + width && xhi > position[1] - width
    elseif shape === :horizontal_plane
        return ylo < position[2] + width && yhi > position[2] - width
    end
    cx, cy = position
    nearest = hypot(max(xlo - cx, 0, cx - xhi), max(ylo - cy, 0, cy - yhi))
    farthest = hypot(max(abs(xlo - cx), abs(xhi - cx)), max(abs(ylo - cy), abs(yhi - cy)))
    return nearest < radius + width && farthest > radius - width
end

@testset "Static interface and geometry bands reach a fixed point (#117)" begin
    target = 2
    fine_h = 1 / (12 * 2^target)
    radius = 0.19
    # The last two plane offsets place the interface just beyond one and two
    # target-cell halos of a block boundary, respectively: at the coarser
    # parent level the same halos would reach twice as far.
    cases = (
        (:plane, (0.5, 0.0), true),
        (:plane, (0.5 - 1.5fine_h, 0.0), false),
        (:plane, (0.5 - 3.5fine_h, 0.0), false),
        (:horizontal_plane, (0.0, 0.5 - 1.5fine_h), false),
        (:horizontal_plane, (0.0, 0.5 - 3.5fine_h), false),
        (:drop, (0.5, 0.5), true),
        (:drop, (0.475, 0.525), false),
        (:drop, (0.53, 0.48), false),
    )
    for kind in (:interface, :geometry), buffer in (1, 2),
            (shape, position, initially_refined) in cases
        @testset "$kind $shape $position buffer=$buffer" begin
            mesh = BlockTreeMesh(
                base_mesh = (12, 12), block_size = (4, 4), halo = 2,
                max_level = 3, periodic = (false, false),
                adaptation = MeshAdaptationOptions(coarsen_delay = 2),
                # Include one above-target family as well as overrefined
                # distant blocks in the initially refined cases.
                refine = (key, origin, extent) -> initially_refined && (
                    key.level == 0 ||
                        (key.level == 1 && key.i == 2 && key.j == 2) ||
                        (key.level == 2 && key.i == 4 && key.j == 4)
                ),
            )
            interface = InterfaceField(mesh; contact_angle = nothing)
            levelset = (x, y) -> hypot(x - position[1], y - position[2]) - radius
            function initialize_static!()
                if shape === :plane
                    initialize!(interface, (1.0, 0.0), position[1])
                elseif shape === :horizontal_plane
                    initialize!(interface, (0.0, 1.0), position[2])
                else
                    # Per-cell quadrature error is below the fraction floor;
                    # it does not determine a numerical test tolerance.
                    initialize!(interface, levelset; lipschitz = 1, atol = 1.0e-8)
                end
                return nothing
            end
            initialize_static!()
            geometry = kind === :geometry ? GeometryField(interface) : nothing
            indicator = kind === :geometry ?
                GeometryRefinementIndicator(geometry; level = target, buffer_cells = buffer) :
                InterfaceRefinementIndicator(interface; level = target, buffer_cells = buffer)
            workspace = AMRWorkspace(mesh, kind === :geometry ? geometry : interface)
            changes = Bool[]
            for step in 1:12
                stats = adapt!(workspace, indicator; step, force = true)
                @test stats.attempted
                push!(changes, stats.changed)
                if stats.changed
                    # Re-sample the same analytic shape after transfer so the
                    # oracle checks a genuinely stationary physical interface.
                    initialize_static!()
                    isnothing(geometry) || refresh_geometry!(geometry)
                end
            end
            @test any(changes[1:6])
            # Six further attempts span two complete cooldown cycles.
            @test !any(changes[7:12])
            @test maximum(block.key.level for block in mesh.blocks) == target
            @test any(block -> block.key.level < target, mesh.blocks)
            @test isbalanced(mesh)
            in_band = filter(
                block -> _intersects_static_band(
                    block, shape, position, radius, buffer * fine_h,
                ),
                mesh.blocks,
            )
            @test !isempty(in_band)
            @test all(block -> block.key.level == target, in_band)
        end
    end
end

@testset "Stationary droplet transfer does not regrid indefinitely (#117)" begin
    for kind in (:interface, :geometry), buffer in (1, 2)
        @testset "$kind buffer=$buffer" begin
            mesh = BlockTreeMesh(
                base_mesh = (12, 12), block_size = (4, 4), halo = 2,
                max_level = 2, periodic = (false, false),
                adaptation = MeshAdaptationOptions(coarsen_delay = 2),
            )
            levelset = (x, y) -> hypot(x - 0.475, y - 0.525) - 0.19
            # Per-cell fraction quadrature is more accurate than the
            # interface fraction floor.
            interface = InterfaceField(mesh, levelset; lipschitz = 1, atol = 1.0e-8)
            geometry = kind === :geometry ? GeometryField(interface) : nothing
            indicator = kind === :geometry ?
                GeometryRefinementIndicator(geometry; level = 2, buffer_cells = buffer) :
                InterfaceRefinementIndicator(interface; level = 2, buffer_cells = buffer)
            workspace = AMRWorkspace(mesh, kind === :geometry ? geometry : interface)
            changes = Bool[]
            for step in 1:12
                # Exercise the normal PLIC transfer path: the interface is not
                # reinitialized or advected between adaptation attempts.
                stats = adapt!(workspace, indicator; step, force = true)
                @test stats.attempted
                push!(changes, stats.changed)
            end
            @test any(changes[1:6])
            @test !any(changes[7:12])
            mixed_blocks = [
                b for b in eachindex(mesh.blocks)
                    if any(c -> interface.fraction_floor < c < 1 - interface.fraction_floor, interior(interface, b))
            ]
            @test !isempty(mixed_blocks)
            @test all(b -> mesh.blocks[b].key.level == 2, mixed_blocks)
            @test any(block -> block.key.level < 2, mesh.blocks)
            @test isbalanced(mesh)
        end
    end
end

@testset "Curvature ignores liquid-filled sloped solid cuts (#118)" begin
    mesh = BlockTreeMesh(
        base_mesh = (16, 16), block_size = (8, 8), max_level = 2,
        periodic = (false, false),
        refine = (key, origin, extent) -> key.level == 0,
    )
    solid = InterfaceField(mesh; contact_angle = nothing)
    initialize!(solid, (-0.23, 1.0), 0.17)
    geometry = GeometryField(solid)
    cuts = [
        ref for ref in TwoPhaseAMR._cell_references(mesh)
            if 0 < TwoPhaseAMR._get(geometry.open_fraction, ref) < 1
    ]
    @test !isempty(cuts)
    liquid = InterfaceField(mesh; geometry, init = 1)
    @test all(cuts) do ref
        TwoPhaseAMR._get(liquid.fraction, ref) ==
            TwoPhaseAMR._get(geometry.open_fraction, ref) &&
            TwoPhaseAMR._conditional_fraction(liquid, ref) == 1 &&
            isnan(TwoPhaseAMR._get(liquid.curvature, ref))
    end
    labels = Vector{Symbol}(undef, nblocks(mesh))
    refinement_labels!(labels, CurvatureRefinementIndicator(liquid))
    @test all(==(:coarsen), labels)

    # Pure gas and explicit bounds are controls for the same cut geometry.
    gas = InterfaceField(mesh; geometry, init = 0)
    refinement_labels!(labels, CurvatureRefinementIndicator(gas))
    @test all(==(:coarsen), labels)
    refinement_labels!(labels, CurvatureRefinementIndicator(liquid; minimum_level = 2))
    @test all(==(:refine), labels)
    refinement_labels!(labels, CurvatureRefinementIndicator(liquid; maximum_level = 0))
    @test all(==(:coarsen), labels)

    # A small gas bubble is far above the wall. Its true mixed cells must
    # still refine, while the wall's liquid-filled cut blocks may coarsen.
    bubble = InterfaceField(
        mesh, (x, y) -> 0.08 - hypot(x - 0.7, y - 0.73);
        geometry, lipschitz = 1, atol = 1.0e-8,
    )
    refinement_labels!(
        labels, CurvatureRefinementIndicator(bubble; cells_per_radius = 16),
    )
    @test all(ref -> labels[ref.block] === :coarsen, cuts)
    bubble_cells = [
        ref for ref in TwoPhaseAMR._cell_references(mesh)
            if TwoPhaseAMR._get(geometry.open_fraction, ref) == 1 &&
            0 < TwoPhaseAMR._conditional_fraction(bubble, ref) < 1
    ]
    @test !isempty(bubble_cells)
    @test any(ref -> labels[ref.block] === :refine, bubble_cells)

    # A liquid drop intersects the sloping wall. Liquid-filled cut cells in
    # blocks without a liquid-gas segment must not trigger curvature refinement.
    drop = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.431, y - 0.327) - 0.317;
        geometry, lipschitz = 1, atol = 1.0e-8,
    )
    refinement_labels!(labels, CurvatureRefinementIndicator(drop))
    wetted_interior = [
        ref for ref in cuts
            if TwoPhaseAMR._conditional_fraction(drop, ref) == 1 &&
            all(c -> c == 0 || c == 1, interior(drop.conditional_fraction, ref.block))
    ]
    @test !isempty(wetted_interior)
    @test all(ref -> labels[ref.block] === :coarsen, wetted_interior)

    # An unresolved mixed cell inside the solid cut is a real liquid-gas
    # interface; simply ignoring all cut cells would lose this refinement.
    candidates = filter(cuts) do ref
        x, _ = cellcenter(mesh, ref.block, ref.i, ref.j)
        0.25 < x < 0.75 && 0.25 < TwoPhaseAMR._get(geometry.open_fraction, ref) < 0.75
    end
    @test !isempty(candidates)
    triple = first(candidates)
    unresolved = InterfaceField(mesh; geometry, init = 0)
    interior(unresolved, triple.block)[triple.i, triple.j] =
        TwoPhaseAMR._get(geometry.open_fraction, triple) / 2
    refinement_labels!(labels, CurvatureRefinementIndicator(unresolved))
    @test TwoPhaseAMR._conditional_fraction(unresolved, triple) == 0.5
    @test labels[triple.block] === :refine
    @test count(==(:refine), labels) == 1
end

@testset "in-place conservative adaptive field transfer" begin
    options = MeshAdaptationOptions(
        update_interval = 2, coarsen_delay = 3, minimum_level = 0
    )
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 2, adaptation = options
    )
    density = CellField(mesh)
    tracer = CellField(mesh)
    initialize!(density, (x, y) -> 2 + 0.2sinpi(2x) * cospi(2y))
    initialize!(tracer, (x, y) -> 1 + x + y)
    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y)
    )
    face_x = FaceField{1}(
        mesh, (x, y) -> 0.5 + 0.1sinpi(2x) * cospi(2y)
    )
    face_y = FaceField{2}(
        mesh, (x, y) -> -0.25 + 0.2cospi(2x) * sinpi(2y)
    )
    workspace = AMRWorkspace(
        mesh, density, tracer, velocity, face_x, face_y
    )

    cell_integrals = (integrate(density), integrate(tracer))
    velocity_momentum = _component_momenta(velocity)
    face_momentum = (_face_momentum(face_x), _face_momentum(face_y))
    initial_divergence = _maximum_divergence(mesh, velocity)

    refined = adapt!(workspace, [:refine]; step = 0)
    @test refined == AMRStats(true, true, 1, 4, 1, 0)
    @test all(block -> block.key.level == 1, mesh.blocks)
    @test isbalanced(mesh)
    @test integrate(density) ≈ cell_integrals[1] atol = 8.0e-14
    @test integrate(tracer) ≈ cell_integrals[2] atol = 8.0e-14
    @test all(
        isapprox.(
            _component_momenta(velocity), velocity_momentum; atol = 2.0e-13
        )
    )
    @test all(
        isapprox.(
            (
                _face_momentum(face_x), _face_momentum(face_y),
            ),
            face_momentum; atol = 2.0e-13
        )
    )
    @test _maximum_divergence(mesh, velocity) <=
        initial_divergence + 2.0e-13
    first_child_storage = Set(objectid.(density.blocks))

    skipped = adapt!(workspace, fill(:coarsen, nblocks(mesh)); step = 1)
    @test !skipped.attempted
    cooling_down = adapt!(
        workspace, fill(:coarsen, nblocks(mesh)); step = 2
    )
    @test cooling_down.attempted
    @test !cooling_down.changed
    skipped_again = adapt!(
        workspace, fill(:coarsen, nblocks(mesh)); step = 3
    )
    @test !skipped_again.attempted
    coarsened = adapt!(workspace, fill(:coarsen, nblocks(mesh)); step = 4)
    @test coarsened == AMRStats(true, true, 4, 1, 0, 1)
    @test integrate(density) ≈ cell_integrals[1] atol = 8.0e-14
    @test integrate(tracer) ≈ cell_integrals[2] atol = 8.0e-14
    @test all(
        isapprox.(
            _component_momenta(velocity), velocity_momentum; atol = 2.0e-13
        )
    )
    @test all(
        isapprox.(
            (
                _face_momentum(face_x), _face_momentum(face_y),
            ),
            face_momentum; atol = 2.0e-13
        )
    )
    @test _maximum_divergence(mesh, velocity) <=
        initial_divergence + 3.0e-13

    # The next refinement reuses the four child matrices retained by the
    # field buffer instead of allocating another set of block payloads.
    rerefined = adapt!(workspace, [:refine]; step = 6)
    @test rerefined.changed
    @test Set(objectid.(density.blocks)) == first_child_storage
end


@testset "face prolongation is affine exact" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), periodic = (false, false), max_level = 1,
        adaptation = MeshAdaptationOptions(coarsen_delay = 0)
    )
    xface = FaceField{1}(mesh, (x, y) -> 1 + 2x - 3y)
    yface = FaceField{2}(mesh, (x, y) -> -2 + 4x + y)
    workspace = AMRWorkspace(mesh, xface, yface)
    stats = adapt!(workspace, [:refine]; step = 0)
    @test stats.changed

    xerror = maximum(
        abs(
            xface.interior[index] - (1 + 2face.center[1] - 3face.center[2])
        )
            for (index, face) in enumerate(filter(f -> f.axis == 1, mesh.faces))
    )
    yerror = maximum(
        abs(
            yface.interior[index] - (-2 + 4face.center[1] + face.center[2])
        )
            for (index, face) in enumerate(filter(f -> f.axis == 2, mesh.faces))
    )
    @test xerror <= 5.0e-14
    @test yerror <= 5.0e-14
end


@testset "tree hierarchy reuse compares actual schedules" begin
    coarse = BlockTreeMesh(
        block_size = (2, 2), max_level = 0, periodic = (false, false),
    )
    same_schedule = BlockTreeMesh(
        block_size = (2, 2), max_level = 1, periodic = (false, false),
    )
    @test coarse.max_level != same_schedule.max_level
    @test coarse.faces == same_schedule.faces
    @test coarse.boundary_faces == same_schedule.boundary_faces
    @test TwoPhaseAMR._same_tree_topology(coarse, same_schedule)

    # Even with the same ceiling and leaf keys, schedule order and metrics are
    # part of the reused operators and may not be inferred from those keys.
    altered = BlockTreeMesh(
        block_size = (2, 2), max_level = 0, periodic = (false, false),
    )
    @test TwoPhaseAMR._same_tree_topology(coarse, altered)
    altered.faces[1], altered.faces[2] = altered.faces[2], altered.faces[1]
    @test !TwoPhaseAMR._same_tree_topology(coarse, altered)
    altered.faces[1], altered.faces[2] = altered.faces[2], altered.faces[1]
    face = altered.faces[1]
    altered.faces[1] = TwoPhaseAMR.InteriorFace(
        face.minus, face.plus, face.axis, face.center, 2face.area,
    )
    @test !TwoPhaseAMR._same_tree_topology(coarse, altered)
    altered.faces[1] = face
    altered.boundary_faces[1], altered.boundary_faces[2] =
        altered.boundary_faces[2], altered.boundary_faces[1]
    @test !TwoPhaseAMR._same_tree_topology(coarse, altered)
    altered.boundary_faces[1], altered.boundary_faces[2] =
        altered.boundary_faces[2], altered.boundary_faces[1]
    altered.face_dual_volumes[1] *= 2
    @test !TwoPhaseAMR._same_tree_topology(coarse, altered)
    altered.face_dual_volumes[1] /= 2
    block = altered.blocks[1]
    altered.blocks[1] = TwoPhaseAMR.LeafBlock(
        block.key, block.origin, block.extent, block.spacing, 2block.volume,
    )
    @test !TwoPhaseAMR._same_tree_topology(coarse, altered)
end


@testset "Navier--Stokes optionally adapts and rebuilds its operators" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 1,
        adaptation = MeshAdaptationOptions(coarsen_delay = 0)
    )
    marker = CellField(mesh)
    initialize!(marker, (x, y) -> 1 + 0.2sinpi(4x) * sinpi(4y))
    passive_face = FaceField{1}(mesh, (x, y) -> x - y)
    pressure = CellField(mesh)
    velocity = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    indicator = InterpolationErrorIndicator(
        marker; E_refine = 1.0e-3, E_coarsen = 1.0e-4,
        t_init = 0.015
    )
    solver = IncompressibleNavierStokesSolver(
        mesh; density = 1.0, dynamic_viscosity = 0.01,
        boundary = FreeSlip(), coarse_size = 8,
        refinement_indicators = indicator,
        adaptive_fields = (passive_face,)
    )
    workspace = NavierStokesWorkspace(solver)

    inactive = simulate!(
        velocity, pressure, solver, 0.01, workspace;
        dt = 0.01, tolerance = 1.0e-10, max_cycles = 30
    )
    @test inactive.mesh_updates == 0
    @test nblocks(mesh) == 1

    refined = simulate!(
        velocity, pressure, solver, 0.02, workspace;
        time = 0.01, dt = 0.01, tolerance = 1.0e-10, max_cycles = 30
    )
    @test refined.mesh_updates == 1
    @test nblocks(mesh) == 4
    @test length(pressure.blocks) == nblocks(mesh)
    @test length(passive_face.interior) ==
        count(face -> face.axis == 1, mesh.faces)
    @test length(workspace.projection.elliptic.input) == ncells(mesh)
    @test length(solver.pressure_solver.operator.volumes) == ncells(mesh)
    old_coarse_meshes = (
        last(solver.pressure_solver.preconditioner.operators).mesh,
        last(workspace.elliptic.x.preconditioner.diffusion_operators).mesh,
        last(workspace.elliptic.y.preconditioner.diffusion_operators).mesh,
    )

    initialize!(marker, (x, y) -> 1.0)
    coarsened = simulate!(
        velocity, pressure, solver, 0.03, workspace;
        time = 0.02, dt = 0.01, tolerance = 1.0e-10, max_cycles = 30
    )
    @test coarsened.mesh_updates == 1
    @test nblocks(mesh) == 1
    @test mesh.adaptation.completed_steps == 3
    for coarse_mesh in old_coarse_meshes
        # Pruning records the active maximum (0), while the live AMR mesh
        # retains its refinement ceiling (1); these schedules really match.
        @test coarse_mesh.max_level == 0
        @test mesh.max_level == 1
        @test coarse_mesh.faces == mesh.faces
        @test coarse_mesh.boundary_faces == mesh.boundary_faces
        @test TwoPhaseAMR._same_tree_topology(coarse_mesh, mesh)
    end
    @test solver.pressure_solver.preconditioner.reused_levels == 1
    @test workspace.elliptic.x.preconditioner.reused_levels == 1
    @test workspace.elliptic.y.preconditioner.reused_levels == 1
    fresh_pressure = TreeGeometricMultigridPreconditioner(
        solver.pressure_solver.operator; coarse_size = 8,
    )
    fresh_velocity = TreeStaggeredGeometricMultigridPreconditioner(
        workspace.elliptic.operator; coarse_size = 8,
    )
    for (reused, fresh) in (
            (solver.pressure_solver.preconditioner, fresh_pressure),
            (workspace.elliptic.x.preconditioner, fresh_velocity.x),
            (workspace.elliptic.y.preconditioner, fresh_velocity.y),
        )
        rhs = [sin(i / 3) for i in 1:size(fresh)[1]]
        reused_result, fresh_result = similar(rhs), similar(rhs)
        ldiv!(reused_result, reused, rhs)
        ldiv!(fresh_result, fresh, rhs)
        @test reused_result ≈ fresh_result rtol = 1.0e-12 atol = 1.0e-12
    end

    registry = VisualizationFieldRegistry()
    register_field!(registry, :pressure, pressure)
    exact = Recorder(
        registry; keyframes = AtTimes([0.035]), exact_keyframes = true,
    )
    @test_throws ArgumentError simulate!(
        velocity, pressure, solver, 0.04, workspace;
        time = 0.03, dt = 0.01, observer = exact,
        tolerance = 1.0e-10, max_cycles = 30
    )
end

@testset "adaptive refinement remains 2:1 graded" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 3,
        adaptation = MeshAdaptationOptions(coarsen_delay = 0),
        refine = (key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0)
    )
    scalar = CellField(mesh)
    initialize!(scalar, (x, y) -> 1 + 0.1cospi(2x) * cospi(2y))
    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y)
    )
    workspace = AMRWorkspace(mesh, scalar, velocity)
    mass = integrate(scalar)
    momentum = _component_momenta(velocity)
    divergence = _maximum_divergence(mesh, velocity)

    labels = fill(:keep, nblocks(mesh))
    selected = findfirst(block -> block.key == BlockKey(2, 0, 0), mesh.blocks)
    labels[selected] = :refine
    stats = adapt!(workspace, labels; step = 0)
    @test stats.changed
    @test stats.refined_blocks >= 1
    @test maximum(block.key.level for block in mesh.blocks) == 3
    @test isbalanced(mesh)
    @test integrate(scalar) ≈ mass atol = 2.0e-13
    @test all(
        isapprox.(
            _component_momenta(velocity), momentum; atol = 3.0e-13
        )
    )
    @test _maximum_divergence(mesh, velocity) <= divergence + 5.0e-12

    # A keep request in a sibling family prevents the other three coarsening
    # requests from removing that family.
    family = BlockKey(2, 0, 0)
    children = TwoPhaseAMR._child_keys(family)
    labels = fill(:keep, nblocks(mesh))
    for (index, block) in pairs(mesh.blocks)
        block.key in children && (labels[index] = :coarsen)
    end
    kept_child = findfirst(block -> block.key == children[1], mesh.blocks)
    labels[kept_child] = :keep
    kept = adapt!(workspace, labels; step = 1)
    @test !kept.changed
    @test all(child -> any(block -> block.key == child, mesh.blocks), children)
end

@testset "multi-root adaptation and conservative transfer" begin
    mesh = BlockTreeMesh(
        base_mesh = (8, 8), block_size = (4, 4), max_level = 2,
        adaptation = MeshAdaptationOptions(coarsen_delay = 0),
    )
    scalar = CellField(mesh; init = 3.25)
    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? 1.5 : -0.75,
    )
    workspace = AMRWorkspace(mesh, scalar, velocity)

    labels = [
        block.key == BlockKey(0, 0, 0) ? :refine : :keep
            for block in mesh.blocks
    ]
    first = adapt!(workspace, labels; step = 1, force = true)
    @test first.changed
    @test length(mesh.root.roots) == 4
    @test isbalanced(mesh)

    labels = [
        block.key.level == 1 && block.key.i == 1 && block.key.j == 0 ?
            :refine : :keep for block in mesh.blocks
    ]
    second = adapt!(workspace, labels; step = 2, force = true)
    @test second.changed
    @test maximum(block.key.level for block in mesh.blocks) == 2
    @test isbalanced(mesh)
    @test any(
        block -> block.key.level == 1 && block.origin[1] >= 0.5,
        mesh.blocks,
    )

    @test integrate(scalar) ≈ 3.25 * prod(mesh.extent) atol = 32eps()
    @test all(
        all(value -> value == 3.25, interior(scalar, block))
            for block in eachindex(mesh.blocks)
    )
    @test all(
        velocity.interior[index] == (face.axis == 1 ? 1.5 : -0.75)
            for (index, face) in pairs(mesh.faces)
    )

    for step in 3:5
        adapt!(workspace, fill(:coarsen, nblocks(mesh)); step, force = true)
    end
    @test nblocks(mesh) == 4
    @test all(block -> iszero(block.key.level), mesh.blocks)
    @test [block.key for block in mesh.blocks] ==
        [BlockKey(0, i, j) for j in 0:1 for i in 0:1]
    @test length(mesh.root.roots) == 4
    @test integrate(scalar) ≈ 3.25 * prod(mesh.extent) atol = 32eps()
end

@testset "mesh adaptation option validation" begin
    @test_throws ArgumentError MeshAdaptationOptions(update_interval = 0)
    @test_throws ArgumentError MeshAdaptationOptions(coarsen_delay = -1)
    @test_throws ArgumentError MeshAdaptationOptions(minimum_level = -1)
    @test_throws ArgumentError BlockTreeMesh(
        max_level = 1,
        adaptation = MeshAdaptationOptions(minimum_level = 2)
    )
end

@testset "non-dyadic dynamic balancing and transfer" begin
    for periodic in ((false, false), (true, false), (false, true), (true, true))
        mesh = BlockTreeMesh(
            base_mesh = (8, 8), block_size = (2, 2),
            extent = (0.7, 3.3), periodic = periodic, max_level = 3,
        )
        scalar = CellField(mesh; init = 3.25)
        velocity = FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? 1.5 : -0.75)
        workspace = AMRWorkspace(mesh, scalar, velocity)
        for level in 0:2
            target = BlockKey(level, 0, 0)
            labels = [block.key == target ? :refine : :keep for block in mesh.blocks]
            @test adapt!(workspace, labels; step = level + 1, force = true).changed
            @test _test_ungraded_pairs(mesh) == (0, 0)
            @test isbalanced(mesh)
            @test all(
                all(value -> value == 3.25, interior(scalar, b)) for b in eachindex(mesh.blocks)
            )
            @test all(
                isapprox(
                    velocity.interior[index], face.axis == 1 ? 1.5 : -0.75;
                    atol = 256eps(), rtol = 256eps(),
                ) for (index, face) in pairs(mesh.faces)
            )
            @test _maximum_divergence(mesh, velocity) <=
                512eps() / minimum(minimum(block.spacing) for block in mesh.blocks)
        end
    end
end

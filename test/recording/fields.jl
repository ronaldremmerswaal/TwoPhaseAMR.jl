using Test, TwoPhaseAMR

_frame(registry; kwargs...) = TwoPhaseAMR._registry_frame(registry; kwargs...)

_domain(topology) = (
    topology.origin[1], topology.origin[1] + topology.extent[1],
    topology.origin[2], topology.origin[2] + topology.extent[2],
)

# Rasterized value at the centre of every cell, in frame cell order.
function _cell_centre_samples(frame, name; phase_rendering = :cell)
    topology = frame.topology
    nx, ny = topology.block_size
    samples = Float32[]
    for block in 1:nblocks(topology), j in 1:ny, i in 1:nx
        dx, dy = topology.spacings[block]
        x = topology.origins[block][1] + (i - 0.5) * dx
        y = topology.origins[block][2] + (j - 0.5) * dy
        rect = (x - dx / 8, x + dx / 8, y - dy / 8, y + dy / 8)
        push!(samples, only(rasterize(frame, name, rect, (1, 1); phase_rendering)))
    end
    return samples
end

# Areas covered by any value and by `value` on a fine raster of the domain.
function _raster_areas(frame, name, value; pixels = 512, phase_rendering = :plic)
    topology = frame.topology
    image = rasterize(
        frame, name, _domain(topology), (pixels, pixels); phase_rendering,
    )
    pixel_area = prod(topology.extent) / pixels^2
    return count(!isnan, image) * pixel_area,
        count(==(Float32(value)), image) * pixel_area
end

@testset "Cell values follow the block-major frame order" begin
    mesh = BlockTreeMesh(
        block_size = (4, 2), max_level = 2,
        refine = (key, origin, extent) ->
            key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0)
    )
    field = CellField(mesh)
    initialize!(field, (x, y) -> 2x - 3y)
    values = visualization_values(field)
    @test length(values) == ncells(mesh)
    nx, ny = mesh.block_size
    index = 0
    for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        index += 1
        x, y = cellcenter(mesh, block, i, j)
        @test values[index] == 2x - 3y
    end
    registry = VisualizationFieldRegistry()
    register_field!(registry, :field, field)
    frame = _frame(registry)
    @test frame[:field] == values
    @test _cell_centre_samples(frame, :field) == Float32.(values)
end

@testset "Phase fields store both values; majority cells select one" begin
    for T in (Float32, Float64)
        mesh = BlockTreeMesh(block_size = (4, 4), max_level = 1)
        interface = InterfaceField(mesh, (x, y) -> x - 0.625; T, lipschitz = 1)
        # The interface bisects column 3: the exact tie must select liquid.
        @test any(==(T(0.5)), visualization_values(interface.conditional_fraction))
        scalar = PhaseScalar(interface; liquid = 7, gas = -11)
        xscalar = PhaseFaceScalar{1}(interface; liquid = 13, gas = -17)
        yscalar = PhaseFaceScalar{2}(interface; liquid = 19, gas = -23)
        velocity = (
            FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? 2 : 3; T),
            FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? -4 : -5; T),
        )
        TwoPhaseAMR._mark_inactive_phase_velocities!(velocity, interface)
        for ref in TwoPhaseAMR._cell_references(mesh)
            TwoPhaseAMR._phase_cell_is_active(interface, ref, :liquid) ||
                TwoPhaseAMR._set!(scalar.liquid, T(NaN), ref)
            TwoPhaseAMR._phase_cell_is_active(interface, ref, :gas) ||
                TwoPhaseAMR._set!(scalar.gas, T(NaN), ref)
        end
        registry = VisualizationFieldRegistry()
        register_field!(registry, :scalar, scalar)
        register_field!(registry, :xscalar, xscalar)
        register_field!(registry, :yscalar, yscalar)
        register_default_fields!(registry, :velocity, velocity, interface)
        register_field!(registry, :ordinary_face, xscalar.liquid)
        frame = _frame(registry)
        liquid = visualization_values(interface.conditional_fraction) .>= T(0.5)
        @test only(frame.phases).liquid == liquid
        @test frame[:scalar] isa TwoPhaseAMR.TwoPhaseVisualizationValues{T}
        @test isequal(frame[:scalar].liquid, visualization_values(scalar.liquid))
        @test frame[:xscalar].gas == visualization_values(xscalar.gas)
        for (name, lv, gv) in (
                (:scalar, 7, -11), (:xscalar, 13, -17), (:yscalar, 19, -23),
                (:velocity_x, 2, -4), (:velocity_y, 3, -5),
                (:velocity_magnitude, sqrt(13), sqrt(41)),
            )
            @test _cell_centre_samples(frame, name) ≈
                Float32[liquid[i] ? lv : gv for i in eachindex(liquid)]
        end
        # The curl is derived within each phase, not from a velocity switched
        # across the interface (which would invent a large sheet curl).
        curl = frame[:velocity_vorticity]
        for values in (curl.liquid, curl.gas)
            @test any(isfinite, values)
            @test all(value -> isnan(value) || abs(value) < 100eps(T), values)
        end
        @test frame[:ordinary_face] isa Vector{T}
        @test length(frame[:ordinary_face]) ==
            length(xscalar.liquid.interior) + length(xscalar.liquid.boundary)
        @test length(registry.sampling_workspaces) == 2
        old_values = copy(frame[:velocity_x])
        fill!(velocity[1].interior, T(99))
        fill!(velocity[1].boundary, T(99))
        next_frame = _frame(registry)
        @test isequal(frame[:velocity_x], old_values)
        @test !isequal(next_frame[:velocity_x], old_values)
        @test length(registry.sampling_workspaces) == 2

        # Exact keyframes substitute an independent interface and fields; the
        # captured majority must follow the substitutes.
        independent = InterfaceField(mesh, (x, y) -> x - 0.375; T, lipschitz = 1)
        replacements = IdDict{Any, Any}(
            interface.conditional_fraction => independent.conditional_fraction,
            interface.fraction => independent.fraction,
            interface.normal_x => independent.normal_x,
            interface.normal_y => independent.normal_y,
            interface.intercept => independent.intercept,
            scalar.liquid => CellField(mesh, T), scalar.gas => CellField(mesh, T),
        )
        initialize!(replacements[scalar.liquid], (x, y) -> T(31))
        initialize!(replacements[scalar.gas], (x, y) -> T(-37))
        exact = _frame(registry; replacements)
        expected = ifelse.(
            visualization_values(independent.conditional_fraction) .>= T(0.5),
            31.0f0, -37.0f0,
        )
        @test _cell_centre_samples(exact, :scalar) == expected
    end
end

@testset "Frames stay valid after the mesh adapts" begin
    mesh = BlockTreeMesh(block_size = (4, 4), max_level = 1, periodic = (true, false))
    geometry = GeometryField(mesh, (x, y) -> y - 0.2; lipschitz = 1)
    interface = InterfaceField(mesh, (x, y) -> x - 0.55; geometry, lipschitz = 1)
    scalar = PhaseScalar(interface; liquid = 2, gas = -3)
    face_scalar = PhaseFaceScalar{1}(interface; liquid = 4, gas = -5)
    registry = VisualizationFieldRegistry()
    register_field!(registry, :scalar, scalar)
    register_field!(registry, :face_scalar, face_scalar)
    frame = _frame(registry)
    before = [
        rasterize(frame, name, (0.0, 1.0, 0.0, 1.0), (64, 64); phase_rendering)
            for name in (:scalar, :face_scalar), phase_rendering in (:cell, :plic)
    ]
    amr = AMRWorkspace(mesh, interface, scalar, face_scalar, geometry)
    @test adapt!(amr, [:refine]; step = 1, force = true).changed
    after = [
        rasterize(frame, name, (0.0, 1.0, 0.0, 1.0), (64, 64); phase_rendering)
            for name in (:scalar, :face_scalar), phase_rendering in (:cell, :plic)
    ]
    @test isequal(before, after)
    @test nblocks(frame.topology) == 1

    # Majority cells use open-fluid fractions and leave solid cells empty.
    fresh = _frame(registry)
    samples = _cell_centre_samples(fresh, :scalar)
    for (index, ref) in pairs(TwoPhaseAMR._cell_references(mesh))
        open = TwoPhaseAMR._get(geometry.open_fraction, ref)
        fraction = TwoPhaseAMR._get(interface.conditional_fraction, ref)
        @test iszero(open) ? isnan(samples[index]) :
            samples[index] == (fraction >= 0.5 ? 2 : -3)
    end
end

@testset "Sharp phase rendering reproduces PLIC areas" begin
    mesh = BlockTreeMesh(block_size = (8, 8), max_level = 0)
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.24; lipschitz = 1,
    )
    registry = VisualizationFieldRegistry()
    register_field!(registry, :scalar, PhaseScalar(interface; liquid = 1, gas = 2))
    register_field!(
        registry, :staggered_scalar, PhaseFaceScalar{2}(interface; liquid = 1, gas = 2),
    )
    @test registry.interfaces == [interface]
    frame = _frame(registry)
    for name in (:scalar, :staggered_scalar)
        covered, liquid = _raster_areas(frame, name, 1; pixels = 1024)
        @test covered ≈ 1
        @test liquid ≈ integrate(interface) rtol = 2.0e-3
        majority_covered, majority_liquid =
            _raster_areas(frame, name, 1; phase_rendering = :cell)
        @test majority_covered ≈ 1
        @test majority_liquid ≈
            count(only(frame.phases).liquid) / ncells(mesh)
    end

    geometry = GeometryField(mesh, (x, y) -> y - 0.3; lipschitz = 1)
    cut_interface = InterfaceField(mesh, (x, y) -> x - 0.55; geometry, lipschitz = 1)
    cut_registry = VisualizationFieldRegistry()
    register_field!(cut_registry, :scalar, PhaseScalar(cut_interface; liquid = 2, gas = 3))
    register_field!(
        cut_registry, :staggered_scalar,
        PhaseFaceScalar{1}(cut_interface; liquid = 2, gas = 3),
    )
    cut_frame = _frame(cut_registry)
    open_area = integrate(geometry.open_fraction)
    for name in (:scalar, :staggered_scalar)
        covered, liquid = _raster_areas(cut_frame, name, 2; pixels = 1024)
        @test covered ≈ open_area rtol = 2.0e-3
        @test liquid ≈ integrate(cut_interface) rtol = 5.0e-3
    end
    # Frames retain the interface curve and the solid overlay.
    @test !isempty(only(cut_frame.phases).segments)
    @test !isempty(only(cut_frame.topology.solids).polygons.triangles)
    @test cut_registry.geometries == [geometry]

    # Mixed precision: Float32 phase fields on a shifted Float64 AMR mesh.
    mixed_mesh = BlockTreeMesh(
        block_size = (4, 4), origin = (0.1, -0.17), extent = (0.93, 0.81),
        periodic = (true, true), max_level = 2,
        refine = (key, origin, extent) ->
            key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
    )
    mixed_interface = InterfaceField(
        mixed_mesh, (x, y) -> hypot(x - 0.51, y - 0.23) - 0.24;
        T = Float32, lipschitz = 1,
    )
    mixed_registry = VisualizationFieldRegistry()
    register_field!(mixed_registry, :cell, PhaseScalar(mixed_interface; liquid = 1, gas = 2))
    register_field!(mixed_registry, :x, PhaseFaceScalar{1}(mixed_interface; liquid = 1, gas = 2))
    register_field!(mixed_registry, :y, PhaseFaceScalar{2}(mixed_interface; liquid = 1, gas = 2))
    mixed_frame = _frame(mixed_registry)
    @test mixed_frame[:cell].liquid isa Vector{Float32}
    @test eltype(only(mixed_frame.phases).planes) == NTuple{3, Float32}
    @test eltype(mixed_frame.topology.origins) == NTuple{2, Float64}
    for name in (:cell, :x, :y)
        covered, liquid = _raster_areas(mixed_frame, name, 1; pixels = 1024)
        @test covered ≈ prod(mixed_mesh.extent)
        @test liquid ≈ integrate(mixed_interface) rtol = 5.0e-3
    end
end

@testset "Two-velocity fields combine intrinsic phase values" begin
    mesh = BlockTreeMesh(block_size = (8, 8), max_level = 0)
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.24; lipschitz = 1,
    )
    velocity = (
        FaceVelocity(mesh, (x, y, time, axis) -> axis == 1 ? 1 + x : 2 + y),
        FaceVelocity(mesh, (x, y, time, axis) -> axis == 1 ? -1 - y : 3 - x),
    )
    registry = VisualizationFieldRegistry()
    @test register_default_fields!(registry, :velocity, velocity, interface) === registry
    @test field_names(registry) == [
        :velocity_x, :velocity_y, :velocity_magnitude, :velocity_vorticity,
    ]
    @test registry.interfaces == [interface]
    frame = _frame(registry)
    @test frame[:velocity_x].liquid == visualization_values(velocity[1], Face{1}())
    @test frame[:velocity_x].gas == visualization_values(velocity[2], Face{1}())
    @test frame[:velocity_y].liquid == visualization_values(velocity[1], Face{2}())
    @test frame[:velocity_y].gas == visualization_values(velocity[2], Face{2}())
    for (name, derived) in (
            (:velocity_magnitude, CentredMagnitude()), (:velocity_vorticity, Vorticity()),
        )
        @test isequal(
            frame[name].liquid,
            visualization_values(velocity[1], derived; interface, phase = :liquid),
        )
        @test isequal(
            frame[name].gas,
            visualization_values(velocity[2], derived; interface, phase = :gas),
        )
    end
    captured = copy(frame[:velocity_magnitude])
    fill!(velocity[1].interior, 99)
    fill!(velocity[1].boundary, 99)
    @test isequal(frame[:velocity_magnitude], captured)
    @test !isequal(_frame(registry)[:velocity_magnitude], captured)

    other_interface = InterfaceField(
        BlockTreeMesh(block_size = (8, 8)), (x, y) -> x - 0.5; lipschitz = 1,
    )
    @test_throws ArgumentError register_default_fields!(
        VisualizationFieldRegistry(), :velocity, velocity, other_interface,
    )
end

@testset "Derived two-component face fields" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), periodic = (false, false), max_level = 2,
        refine = (key, origin, extent) ->
            key.level == 0 || (key.level == 1 && key.i == 1 && key.j == 0)
    )
    velocity = FaceVelocity(mesh, (x, y, time, axis) -> axis == 1 ? x + y : 2x - y)
    magnitude = visualization_values(velocity, CentredMagnitude())
    vorticity = visualization_values(velocity, Vorticity())
    components = (FaceField{1}(velocity), FaceField{2}(velocity))
    @test length(magnitude) == length(vorticity) == ncells(mesh)
    @test visualization_values(components, CentredMagnitude()) == magnitude
    @test visualization_values(components, Vorticity()) == vorticity
    for (index, cell) in pairs(TwoPhaseAMR._cell_references(mesh))
        x, y = cellcenter(mesh, cell.block, cell.i, cell.j)
        @test magnitude[index] ≈ hypot(x + y, 2x - y)
        @test vorticity[index] ≈ 1 atol = 2.0e-13
    end
end

@testset "Visualization field registry" begin
    mesh = BlockTreeMesh(block_size = (4, 2), max_level = 1)
    pressure = CellField(mesh)
    xflux = FaceField{1}(mesh)
    yflux = FaceField{2}(mesh)
    velocity = FaceVelocity(mesh, (x, y, time, axis) -> axis == 1 ? y : -x)
    registry = VisualizationFieldRegistry()

    @test register_field!(
        registry, :pressure, pressure; label = "Pressure", units = "Pa"
    ) === registry
    @test register_field!(registry, :xflux, xflux) === registry
    @test register_field!(
        registry, :velocity_x, velocity, Face{1}();
        label = "Velocity x", colormap = :balance
    ) === registry
    @test register_field!(
        registry, :velocity_y, velocity, Face{2}();
        label = "Velocity y", colormap = :balance
    ) === registry
    @test register_field!(
        registry, :velocity_magnitude, velocity, CentredMagnitude();
        label = "Velocity magnitude"
    ) === registry
    @test register_field!(
        registry, :vorticity, velocity, Vorticity();
        label = "Vorticity", colormap = :balance
    ) === registry
    @test register_field!(
        registry, :flux_magnitude, (xflux, yflux), CentredMagnitude()
    ) === registry
    @test register_field!(
        registry, :flux_vorticity, (xflux, yflux), Vorticity()
    ) === registry
    @test field_names(registry) == [
        :pressure, :xflux, :velocity_x, :velocity_y,
        :velocity_magnitude, :vorticity, :flux_magnitude, :flux_vorticity,
    ]
    @test_throws ArgumentError register_field!(registry, :pressure, pressure)
    frame = _frame(registry)
    @test frame.manifest.entries[1].units == "Pa"
    @test frame.manifest.entries[1].placement isa Cell
    @test frame.manifest.entries[2].placement isa Face{1}
    @test [entry.symmetric for entry in frame.manifest.entries] ==
        [false, false, false, false, false, true, false, true]

    other_mesh = BlockTreeMesh(block_size = (4, 2), max_level = 1)
    @test_throws ArgumentError register_field!(registry, :other, CellField(other_mesh))
    @test_throws ArgumentError register_field!(
        VisualizationFieldRegistry(), :invalid_pair,
        (xflux, FaceField{2}(other_mesh)), CentredMagnitude()
    )

    defaults = VisualizationFieldRegistry()
    @test register_default_fields!(
        defaults, :velocity, velocity, :pressure, pressure
    ) === defaults
    @test field_names(defaults) == [
        :velocity_x, :velocity_y, :velocity_magnitude,
        :velocity_vorticity, :pressure,
    ]
    @test [entry.label for entry in defaults.entries] == [
        "Velocity x", "Velocity y", "Velocity magnitude",
        "Velocity vorticity", "Pressure",
    ]
    @test [entry.colormap for entry in defaults.entries] == [
        :balance, :balance, :viridis, :balance, :viridis,
    ]

    tuple_defaults = VisualizationFieldRegistry()
    @test register_default_fields!(tuple_defaults, :mass_flux, (xflux, yflux)) ===
        tuple_defaults
    @test field_names(tuple_defaults) == [
        :mass_flux_x, :mass_flux_y, :mass_flux_magnitude, :mass_flux_vorticity,
    ]

    directional_default = VisualizationFieldRegistry()
    @test register_default_fields!(directional_default, :xflux, xflux) ===
        directional_default
    @test field_names(directional_default) == [:xflux]
    @test directional_default.entries[1].colormap == :balance

    @test_throws ArgumentError register_default_fields!(
        VisualizationFieldRegistry(), :pressure
    )
    @test_throws ArgumentError register_default_fields!(
        VisualizationFieldRegistry(), "pressure", pressure
    )
    @test_throws ArgumentError register_default_fields!(
        VisualizationFieldRegistry(), :unsupported, zeros(2)
    )
    @test_throws ArgumentError register_default_fields!(
        VisualizationFieldRegistry(), :wrong_pair, (yflux, xflux)
    )

    atomic = VisualizationFieldRegistry()
    @test_throws ArgumentError register_default_fields!(
        atomic, :pressure, pressure, :pressure, pressure
    )
    @test isempty(field_names(atomic))
    @test_throws ArgumentError register_default_fields!(
        atomic, :pressure, pressure, :other, CellField(other_mesh)
    )
    @test isempty(field_names(atomic))

    existing = VisualizationFieldRegistry()
    register_field!(existing, :velocity_x, pressure)
    @test_throws ArgumentError register_default_fields!(existing, :velocity, velocity)
    @test field_names(existing) == [:velocity_x]
    @test_throws ArgumentError TwoPhaseAMR._registry_frame(VisualizationFieldRegistry())
end

using LinearAlgebra: diag, eigvals, I

struct _UnsupportedStaggeredScheme <: AbstractStaggeredAdvectionScheme end
struct _UnsupportedThreePointScheme <: TwoPhaseAMR.AbstractThreePointFluxScheme end

function face_values_by_coordinate(field::FaceField{Axis}) where {Axis}
    result = Dict{Tuple{Float64, Float64}, eltype(field)}()
    local_index = 0
    for face in field.mesh.faces
        face.axis == Axis || continue
        local_index += 1
        result[(Float64(face.center[1]), Float64(face.center[2]))] =
            field.interior[local_index]
    end
    return result
end

function lax_wendroff_sweep(q, point, speed, spacing, dt, axis)
    courant = speed * dt / spacing
    minus = axis == 1 ?
        (mod(point[1] - spacing, 1.0), point[2]) :
        (point[1], mod(point[2] - spacing, 1.0))
    plus = axis == 1 ?
        (mod(point[1] + spacing, 1.0), point[2]) :
        (point[1], mod(point[2] + spacing, 1.0))
    return q(point...) - courant * (q(plus...) - q(minus...)) / 2 +
        courant^2 * (q(plus...) - 2q(point...) + q(minus...)) / 2
end


function dual_volumes(mesh, axis)
    result = zeros(count(face -> face.axis == axis, mesh.faces))
    local_index = 0
    for face in mesh.faces
        face.axis == axis || continue
        local_index += 1
        for cell in (face.minus, face.plus)
            block = mesh.blocks[cell.block]
            fraction = face.area / block.spacing[3 - axis]
            result[local_index] += fraction * block.volume / 2
        end
    end
    return result
end


function interpolate_cell_to_component_faces(cell_field, axis)
    mesh = cell_field.mesh
    volumes = dual_volumes(mesh, axis)
    result = zeros(eltype(cell_field), length(volumes))
    local_index = 0
    for face in mesh.faces
        face.axis == axis || continue
        local_index += 1
        for cell in (face.minus, face.plus)
            block = mesh.blocks[cell.block]
            fraction = face.area / block.spacing[3 - axis]
            weight = fraction * block.volume / 2
            result[local_index] += weight * TwoPhaseAMR._get(cell_field, cell)
        end
        result[local_index] /= volumes[local_index]
    end
    return result
end
@testset "wall and cut-wall staggered constant identity" begin
    schemes = (
        Upwind(), Centred(), LaxWendroff(), Fromm(), Minmod(),
        MonotonizedCentral(), VanLeer(), Superbee(),
    )
    for adaptive in (false, true), cut in (false, true)
        mesh = adaptive ? BlockTreeMesh(
                block_size = cut ? (8, 8) : (4, 4), max_level = 2,
                periodic = (false, false),
                refine = (key, origin, extent) -> key.level == 0 ||
                (key.level == 1 && key.i == 0 && key.j == 0),
            ) : BlockTreeMesh(
                block_size = (16, 16), periodic = (false, false),
            )
        wall(x, y) = y - 0.31 - 0.2x
        geometry = cut ? GeometryField(mesh, wall; lipschitz = hypot(1, 0.2)) : nothing
        psi(x, y) = (!cut || wall(x, y) > 0) ?
            sinpi(x)^2 * sinpi(y)^2 / pi : 0.0
        velocity = FaceVelocity(mesh)
        for (index, face) in pairs(mesh.faces)
            x, y = face.center
            h = face.area / 2
            flux = face.axis == 1 ?
                psi(x, y + h) - psi(x, y - h) :
                -(psi(x + h, y) - psi(x - h, y))
            aperture = cut ? geometry.aperture[index] : 1.0
            @test aperture > 0 || abs(flux) <= 64eps()
            velocity.interior[index] = aperture > 0 ?
                flux / (face.area * aperture) : 0.0
        end
        divergence = CellField(mesh)
        if cut
            divergence!(divergence, velocity, geometry)
        else
            divergence!(divergence, velocity)
        end
        @test maximum(
            maximum(abs, interior(divergence, block))
                for block in 1:nblocks(mesh)
        ) <= 1.0e-10
        for Axis in (1, 2)
            one = FaceField{Axis}(mesh; init = 1)
            tendency = FaceField{Axis}(mesh)
            workspace = cut ? StaggeredAdvectionWorkspace(one, geometry) :
                StaggeredAdvectionWorkspace(one)
            rates = zeros(length(one.interior))
            TwoPhaseAMR._accumulate_staggered_rates!(rates, velocity, workspace)
            tolerance = 1024eps() * max(1, maximum(rates))
            for scheme in schemes
                convection!(
                    tendency, one, velocity, 0.0, workspace;
                    scheme, amr_interface_scheme = scheme, boundary = NoSlip(),
                )
                # Open cut volumes can be tiny; the residual is at the
                # round-off scale of the local flux divided by that volume.
                @test maximum(abs, tendency.interior) <= tolerance
            end
        end
        # Also exercise nonzero physical-boundary fluxes. The expected
        # integrated dual divergence is assembled independently from the
        # aperture-weighted cell divergences and side incidence fractions.
        arbitrary = FaceVelocity(
            mesh, (x, y, time, axis) -> axis == 1 ?
                0.2 + 0.1x - 0.05y : -0.1 + 0.08x + 0.12y,
        )
        if cut
            divergence!(divergence, arbitrary, geometry)
        else
            divergence!(divergence, arbitrary)
        end
        for Axis in (1, 2)
            one = FaceField{Axis}(mesh; init = 1)
            tendency = FaceField{Axis}(mesh)
            workspace = cut ? StaggeredAdvectionWorkspace(one, geometry) :
                StaggeredAdvectionWorkspace(one)
            expected = zeros(length(one.interior))
            local_index = 0
            for face in mesh.faces
                face.axis == Axis || continue
                local_index += 1
                for cell in (face.minus, face.plus)
                    block = mesh.blocks[cell.block]
                    open = cut ? TwoPhaseAMR._get(geometry.open_fraction, cell) : 1.0
                    side_fraction = face.area / block.spacing[3 - Axis]
                    expected[local_index] += side_fraction * block.volume *
                        open * TwoPhaseAMR._get(divergence, cell) /
                        (2workspace.schedule.volumes[local_index])
                end
            end
            convection!(
                tendency, one, arbitrary, 0.0, workspace;
                scheme = Centred(), amr_interface_scheme = Centred(),
                boundary = FreeSlip(),
            )
            @test tendency.interior ≈ expected atol = 2.0e-10 rtol = 2.0e-12

            # The LW predictor needs the same identity separately in each
            # sweep direction, including wall and wall-transition fluxes.
            for sweep in (1, 2)
                directional_velocity = copy(arbitrary)
                for (index, face) in pairs(mesh.faces)
                    face.axis == sweep || (directional_velocity.interior[index] = 0)
                end
                for (index, face) in pairs(mesh.boundary_faces)
                    face.axis == sweep || (directional_velocity.boundary[index] = 0)
                end
                directional_divergence = CellField(mesh)
                if cut
                    divergence!(directional_divergence, directional_velocity, geometry)
                else
                    divergence!(directional_divergence, directional_velocity)
                end
                fill!(expected, 0)
                local_index = 0
                for face in mesh.faces
                    face.axis == Axis || continue
                    local_index += 1
                    for cell in (face.minus, face.plus)
                        block = mesh.blocks[cell.block]
                        open = cut ? TwoPhaseAMR._get(geometry.open_fraction, cell) : 1.0
                        side_fraction = face.area / block.spacing[3 - Axis]
                        expected[local_index] += side_fraction * block.volume *
                            open * TwoPhaseAMR._get(directional_divergence, cell) /
                            (2workspace.schedule.volumes[local_index])
                    end
                end
                TwoPhaseAMR._directional_flux_divergence!(
                    workspace.rates.interior, arbitrary, workspace, sweep,
                )
                TwoPhaseAMR._add_boundary_flux_divergence!(
                    workspace.rates.interior, arbitrary, workspace, sweep,
                )
                @test workspace.rates.interior ≈ expected atol = 2.0e-10 rtol = 2.0e-12
            end
        end
    end
end

@testset "staggered CFL includes wall-transition flux" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 2, periodic = (false, false),
        refine = (key, origin, extent) -> key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 0),
    )
    field = FaceField{1}(mesh)
    workspace = StaggeredAdvectionWorkspace(field)
    index = findfirst(
        connection -> connection.normal_component && connection.wall_axis == 2,
        workspace.boundary_faces,
    )
    @test !isnothing(index)
    if !isnothing(index)
        connection = workspace.boundary_faces[index]
        velocity = FaceVelocity(mesh)
        velocity.interior[connection.opposite_flux] = 1
        rates = zeros(length(field.interior))
        TwoPhaseAMR._accumulate_staggered_rates!(rates, velocity, workspace)
        dof = connection.degree_of_freedom
        schedule = workspace.schedule
        internal_rate = sum(schedule.faces) do face
            face.minus == dof || face.plus == dof || return 0.0
            return abs(TwoPhaseAMR._amr_dual_flux(face, schedule, velocity)) /
                schedule.volumes[dof]
        end
        missing_rate = abs(connection.measure) / schedule.volumes[dof]
        @test missing_rate > 0
        @test rates[dof] >= internal_rate + missing_rate - 64eps()
        @test stable_timestep(velocity, workspace; cfl = 0.9) ≈
            0.9 / maximum(rates)

        # The extra edge uses the prescribed wall state on inflow and the
        # interior state on outflow, with the transverse flux orientation.
        fill!(field.interior, 2)
        fill!(field.boundary, 5)
        without_wall = FaceField{1}(mesh)
        with_wall = FaceField{1}(mesh)
        for speed in (-1.0, 1.0), scheme in (Upwind(), Centred())
            velocity.interior[connection.opposite_flux] = speed
            convection!(
                without_wall, field, velocity, 0.0, workspace;
                scheme, amr_interface_scheme = scheme,
            )
            convection!(
                with_wall, field, velocity, 0.0, workspace;
                scheme, amr_interface_scheme = scheme, boundary = FreeSlip(),
            )
            expected = sum(workspace.boundary_faces) do edge
                edge.normal_component && edge.wall_axis == 2 &&
                    edge.opposite_flux == connection.opposite_flux &&
                    edge.degree_of_freedom == dof || return 0.0
                state = scheme isa Upwind ?
                    (edge.side * speed < 0 ? 5.0 : 2.0) : 3.5
                return edge.side * edge.measure * speed * state /
                    schedule.volumes[dof]
            end
            @test expected != 0
            @test with_wall.interior[dof] - without_wall.interior[dof] ≈
                expected atol = 64eps() * max(1, abs(expected))
        end
    end
end


@testset "directional face fields" begin
    mesh = BlockTreeMesh(block_size = (8, 6))
    xfield = FaceField{1}(mesh, Float64; init = 2)
    yfield = FaceField(mesh, Face{2}(), (x, y) -> x - y)
    @test placement(xfield) == Face{1}()
    @test placement(yfield) == Face{2}()
    @test length(xfield.interior) == ncells(mesh)
    @test length(yfield.interior) == ncells(mesh)
    @test all(==(2.0), xfield.interior)
    @test copy(xfield).interior == xfield.interior

    velocity = FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? x + y : x - y)
    copied_x = FaceField{1}(velocity)
    copied_y = FaceField{2}(velocity)
    reconstructed = FaceVelocity(copied_x, copied_y)
    @test reconstructed.interior == velocity.interior

    mesh32 = BlockTreeMesh(
        block_size = (4, 4), origin = (0.0f0, 0.0f0), extent = (1.0f0, 1.0f0)
    )
    field32 = FaceField{1}(mesh32)
    velocity32 = FaceVelocity(mesh32, (x, y, t, axis) -> 0.25f0)
    workspace32 = StaggeredAdvectionWorkspace(field32)
    @test eltype(field32) == Float32
    @test eltype(velocity32) == Float32
    @test stable_timestep(velocity32, workspace32; cfl = 0.5f0) isa Float32

    physical = BlockTreeMesh(block_size = (8, 6), periodic = (false, false))
    @test length(FaceField{1}(physical).boundary) == 12
    @test length(FaceField{2}(physical).boundary) == 16
end

@testset "staggered scratch alias rejection" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    field = FaceField{1}(mesh)
    destination = FaceField{1}(mesh)
    velocity = FaceVelocity(mesh)
    workspace = StaggeredAdvectionWorkspace(field)
    for scratch in (workspace.tendency, workspace.rates)
        @test_throws ArgumentError advect!(scratch, velocity, 0.01, workspace)
        @test_throws ArgumentError convection!(destination, scratch, velocity, 0.01, workspace)
        @test_throws ArgumentError convection!(scratch, field, velocity, 0.01, workspace)
        shared = typeof(scratch)(mesh, scratch.interior, copy(scratch.boundary))
        @test_throws ArgumentError advect!(shared, velocity, 0.01, workspace)
        @test_throws ArgumentError convection!(destination, shared, velocity, 0.01, workspace)
    end
end

@testset "staggered scheme canonicalization" begin
    aliases = (
        (Upwind(), (:upwind,)),
        (Centred(), (:centred, :centered, :central)),
        (LaxWendroff(), (:lw, :lax_wendroff)),
        (Fromm(), (:fromm,)),
        (Minmod(), (:minmod,)),
        (MonotonizedCentral(), (:mc, :monotonized_central, :monotonised_central)),
        (VanLeer(), (:van_leer, :vanleer)),
        (Superbee(), (:superbee,)),
    )
    meshes = (
        BlockTreeMesh(block_size = (4, 4)),
        BlockTreeMesh(
            block_size = (4, 4), max_level = 2,
            refine = (key, origin, extent) ->
            key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
        ),
    )
    for mesh in meshes, Axis in (1, 2)
        field = FaceField{Axis}(mesh, (x, y) -> sinpi(2x) + cospi(2y))
        velocity = FaceVelocity(
            mesh, (x, y, time, axis) ->
            axis == 1 ? 0.2 + 0.1sinpi(2y) : -0.1 + 0.1cospi(2x),
        )
        workspace = StaggeredAdvectionWorkspace(field)
        expected = FaceField{Axis}(mesh)
        actual = FaceField{Axis}(mesh)
        for (typed, symbols) in aliases, dt in (0.0, 0.01)
            convection!(
                expected, field, velocity, dt, workspace;
                scheme = typed, amr_interface_scheme = typed,
            )
            for symbol in symbols, (scheme, interface_scheme) in (
                        (symbol, typed), (typed, symbol), (symbol, symbol),
                    )
                convection!(
                    actual, field, velocity, dt, workspace;
                    scheme, amr_interface_scheme = interface_scheme,
                )
                @test isequal(actual.interior, expected.interior)
                @test isequal(actual.boundary, expected.boundary)
            end
            convection!(
                expected, field, velocity, dt, workspace;
                scheme = Upwind(), amr_interface_scheme = typed,
            )
            for symbol in symbols
                convection!(
                    actual, field, velocity, dt, workspace;
                    scheme = Upwind(), amr_interface_scheme = symbol,
                )
                @test isequal(actual.interior, expected.interior)
                @test isequal(actual.boundary, expected.boundary)
            end
            convection!(
                expected, field, velocity, dt, workspace;
                scheme = typed, amr_interface_scheme = MonotonizedCentral(),
            )
            for symbol in symbols
                convection!(
                    actual, field, velocity, dt, workspace;
                    scheme = symbol, amr_interface_scheme = MonotonizedCentral(),
                )
                @test isequal(actual.interior, expected.interior)
                @test isequal(actual.boundary, expected.boundary)
            end
        end
        for invalid in (
                    :unknown, 3, _UnsupportedStaggeredScheme(), _UnsupportedThreePointScheme(),
                ), dt in (0.0, 0.01)
            for (scheme, interface_scheme, message) in (
                    (invalid, Centred(), "unsupported staggered advection scheme: $invalid"),
                    (Centred(), invalid, "unsupported AMR-interface advection scheme: $invalid"),
                    (invalid, :unknown, "unsupported staggered advection scheme: $invalid"),
                )
                exception = try
                    convection!(
                        actual, field, velocity, dt, workspace;
                        scheme, amr_interface_scheme = interface_scheme,
                    )
                    nothing
                catch error
                    error
                end
                @test exception isa ArgumentError
                @test exception isa ArgumentError && exception.msg == message
            end
        end
    end
end


@testset "simultaneous staggered velocity transport" begin
    mesh = BlockTreeMesh(block_size = (12, 12))
    velocity = FaceVelocity(
        mesh, (x, y, t, axis) -> axis == 1 ? 0.3 : -0.15
    )
    transported = FaceVelocity(
        mesh,
        (x, y, t, axis) -> axis == 1 ?
            1 + 0.1sinpi(2x) * cospi(2y) :
            -0.5 + 0.2cospi(2x) * sinpi(2y),
    )
    workspace = StaggeredAdvectionWorkspace(transported)
    component_x = FaceField{1}(transported)
    component_y = FaceField{2}(transported)
    workspace_x = StaggeredAdvectionWorkspace(component_x)
    workspace_y = StaggeredAdvectionWorkspace(component_y)
    dt = stable_timestep(velocity, workspace; cfl = 0.4)

    advect!(component_x, velocity, dt, workspace_x; scheme = LaxWendroff())
    advect!(component_y, velocity, dt, workspace_y; scheme = LaxWendroff())
    advect!(transported, velocity, dt, workspace; scheme = LaxWendroff())
    @test FaceField{1}(transported).interior ≈ component_x.interior
    @test FaceField{2}(transported).interior ≈ component_y.interior

    self_advected = copy(velocity)
    old_velocity = copy(velocity)
    self_workspace = StaggeredAdvectionWorkspace(self_advected)
    advect!(self_advected, dt, self_workspace; scheme = Upwind())
    @test self_advected.interior == old_velocity.interior
end


@testset "centred-to-staggered divergence compatibility" begin
    mesh = BlockTreeMesh(block_size = (8, 8))
    velocity = FaceVelocity(
        mesh,
        (x, y, t, axis) -> axis == 1 ?
            0.2 + sinpi(2x) * cospi(4y) :
            -0.1 + cospi(4x) * sinpi(2y),
    )
    cell_divergence = CellField(mesh)
    divergence!(cell_divergence, velocity)
    nx, ny = mesh.block_size
    h = mesh.halo
    cell_value(ref) = cell_divergence.blocks[ref.block][ref.i + h, ref.j + h]

    for Axis in (1, 2)
        constant_field = FaceField{Axis}(mesh; init = 1)
        dual_divergence = FaceField{Axis}(mesh)
        workspace = StaggeredAdvectionWorkspace(constant_field)
        convection!(
            dual_divergence, constant_field, velocity, 0.0, workspace;
            scheme = Centred()
        )
        local_index = 0
        for face in mesh.faces
            face.axis == Axis || continue
            local_index += 1
            expected = (cell_value(face.minus) + cell_value(face.plus)) / 2
            @test dual_divergence.interior[local_index] ≈ expected atol = 2.0e-14 rtol = 2.0e-14
        end
    end
end

@testset "staggered advection schemes" begin
    mesh = BlockTreeMesh(block_size = (16, 16))
    velocity = FaceVelocity(
        mesh,
        (x, y, t, axis) -> axis == 1 ? 0.35 : -0.2,
    )

    for Axis in (1, 2), scheme in (Upwind(), Centred(), LaxWendroff())
        field = FaceField{Axis}(mesh, (x, y) -> 1.75)
        workspace = StaggeredAdvectionWorkspace(field)
        dt = stable_timestep(velocity, workspace; cfl = 0.45)
        initial_momentum = integrate(field, workspace)
        advect!(field, velocity, dt, workspace; scheme)
        @test maximum(abs.(field.interior .- 1.75)) <= 8eps()
        @test integrate(field, workspace) ≈ initial_momentum rtol = 4eps()

        initialize!(field, (x, y) -> 1 + 0.15sinpi(2x) * cospi(2y))
        initial_momentum = integrate(field, workspace)
        advect!(field, velocity, dt, workspace; scheme)
        @test integrate(field, workspace) ≈ initial_momentum atol = 8eps()
    end

    # Upwind is monotone under the dual-control-volume CFL restriction.
    field = FaceField{1}(mesh, (x, y) -> 1 + 0.2sinpi(2x) * cospi(2y))
    workspace = StaggeredAdvectionWorkspace(field)
    dt = stable_timestep(velocity, workspace; cfl = 0.8)
    advect!(field, velocity, dt, workspace; scheme = :upwind)
    @test minimum(field.interior) >= 0.8 - 8eps()
    @test maximum(field.interior) <= 1.2 + 8eps()

    # The four standard TVD limiter functions stay in the Sweby region. An
    # isolated jump makes the distinction from LW and unrestricted Fromm
    # visible in one step without involving AMR geometry.
    @test TwoPhaseAMR._flux_limiter(0.5, Minmod()) == 0.5
    @test TwoPhaseAMR._flux_limiter(0.5, MonotonizedCentral()) == 0.75
    @test TwoPhaseAMR._flux_limiter(0.5, VanLeer()) ≈ 2 / 3
    @test TwoPhaseAMR._flux_limiter(0.5, Superbee()) == 1.0
    @test TwoPhaseAMR._flux_limiter(0.5, Fromm()) == 0.75
    step_velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? 0.5 : 0.0
    )
    for scheme in (Minmod(), MonotonizedCentral(), VanLeer(), Superbee())
        step_field = FaceField{1}(mesh, (x, y) -> x < 0.5 ? 0.0 : 1.0)
        step_workspace = StaggeredAdvectionWorkspace(step_field)
        step_dt = stable_timestep(step_velocity, step_workspace; cfl = 0.8)
        initial_integral = integrate(step_field, step_workspace)
        advect!(step_field, step_velocity, step_dt, step_workspace; scheme)
        @test minimum(step_field.interior) >= -16eps()
        @test maximum(step_field.interior) <= 1 + 16eps()
        @test integrate(step_field, step_workspace) ≈ initial_integral atol = 8eps()
    end
    uniform_centre = FaceField{1}(
        mesh, (x, y) -> 1 + 0.1sinpi(2x) * cospi(2y)
    )
    ignored_interface = copy(uniform_centre)
    centre_workspace = StaggeredAdvectionWorkspace(uniform_centre)
    ignored_workspace = StaggeredAdvectionWorkspace(ignored_interface)
    advect!(
        uniform_centre, velocity, dt, centre_workspace;
        scheme = Centred(), amr_interface_scheme = Centred()
    )
    advect!(
        ignored_interface, velocity, dt, ignored_workspace;
        scheme = Centred(), amr_interface_scheme = MonotonizedCentral()
    )
    @test ignored_interface.interior == uniform_centre.interior

    # In one dimension each directional substep reduces exactly to the
    # classical Lax--Wendroff stencil.
    line_mesh = BlockTreeMesh(block_size = (16, 4), extent = (2.0, 1.0))
    speed = 0.7
    line_velocity = FaceVelocity(
        line_mesh, (x, y, t, axis) -> axis == 1 ? speed : 0.0
    )
    line = FaceField{1}(line_mesh, (x, y) -> sinpi(x))
    old = face_values_by_coordinate(line)
    line_workspace = StaggeredAdvectionWorkspace(line)
    dx = line_mesh.blocks[1].spacing[1]
    dt = 0.4dx / speed
    advect!(line, line_velocity, dt, line_workspace; scheme = LaxWendroff())
    courant = speed * dt / dx
    for (coordinate, value) in face_values_by_coordinate(line)
        x, y = coordinate
        left = (mod(x - dx, 2.0), y)
        right = (mod(x + dx, 2.0), y)
        expected = old[coordinate] - courant * (old[right] - old[left]) / 2 +
            courant^2 *
            (old[right] - 2old[coordinate] + old[left]) / 2
        @test value ≈ expected atol = 2.0e-15 rtol = 2.0e-15
    end


    # LW is dimensionally split: complete and publish the x sweep before the
    # y sweep. Successive calls alternate x/y and y/x ordering.
    cross_mesh = BlockTreeMesh(block_size = (8, 8))
    a, b = 0.3, -0.2
    cross_velocity = FaceVelocity(
        cross_mesh, (x, y, t, axis) -> axis == 1 ? a : b
    )
    q(x, y) = 1 + 0.2sinpi(2x) * cospi(2y)
    dx, dy = cross_mesh.blocks[1].spacing
    dt = 0.2 / (abs(a) / dx + abs(b) / dy)
    x_sweep(x, y) = lax_wendroff_sweep(q, (x, y), a, dx, dt, 1)
    for Axis in (1, 2)
        cross = FaceField{Axis}(cross_mesh, q)
        cross_workspace = StaggeredAdvectionWorkspace(cross)
        advect!(cross, cross_velocity, dt, cross_workspace; scheme = LaxWendroff())
        for ((x, y), value) in face_values_by_coordinate(cross)
            expected = lax_wendroff_sweep(
                x_sweep, (x, y), b, dy, dt, 2
            )
            @test value ≈ expected atol = 3.0e-15 rtol = 3.0e-15
        end
        @test !cross_workspace.sweep_forward[]
    end

    # A divergence-free velocity has nonzero directional divergences. They
    # cancel only after both sweeps and must therefore enter the individual
    # variable-coefficient LW predictors.
    variable_velocity = FaceVelocity(
        cross_mesh,
        (x, y, time, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y),
    )
    variable_field = FaceField{1}(cross_mesh; init = 1)
    variable_workspace = StaggeredAdvectionWorkspace(variable_field)
    xdivergence = similar(variable_field.interior)
    ydivergence = similar(variable_field.interior)
    TwoPhaseAMR._directional_flux_divergence!(
        xdivergence, variable_velocity, variable_workspace, 1
    )
    TwoPhaseAMR._directional_flux_divergence!(
        ydivergence, variable_velocity, variable_workspace, 2
    )
    @test maximum(abs, xdivergence) > 0.1
    @test maximum(abs, xdivergence .+ ydivergence) <= 2.0e-14

    @test_throws ArgumentError advect!(
        line, line_velocity, dt, line_workspace;
        scheme = :unknown
    )
    physical_workspace = StaggeredAdvectionWorkspace(
        FaceField{1}(BlockTreeMesh(block_size = (4, 4), periodic = (false, false)))
    )
    @test physical_workspace.schedule isa TwoPhaseAMR.AMRDualAdvectionSchedule
    adaptive = BlockTreeMesh(
        block_size = (4, 4), max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
    )
    @test StaggeredAdvectionWorkspace(FaceField{1}(adaptive)).schedule isa
        TwoPhaseAMR.AMRDualAdvectionSchedule
end


"""Dict accumulation reference of the AMR dual flux weights: the builder before
the count, prefix sum and fill rewrite."""
function reference_amr_flux_coefficients(
        mesh, Axis, ::Type{T}, incidences, global_to_local,
    ) where {T}
    nx, ny = mesh.block_size
    coefficients = Dict{NTuple{3, Int}, T}()
    for (flux_index, flux_face) in pairs(mesh.faces)
        if flux_face.axis == Axis
            centre = global_to_local[flux_index]
            for (cell, sign) in
                ((flux_face.minus, one(T)), (flux_face.plus, -one(T)))

                for component_face in
                    incidences[TwoPhaseAMR._cell_index(cell, nx, ny)]

                    component_face.global_index == flux_index && continue
                    coefficient = sign * component_face.fraction / 2
                    add_amr_flux_weight!(
                        coefficients, component_face.local_index, centre,
                        flux_index, coefficient,
                    )
                end
            end
        else
            minus_faces =
                incidences[TwoPhaseAMR._cell_index(flux_face.minus, nx, ny)]
            plus_faces =
                incidences[TwoPhaseAMR._cell_index(flux_face.plus, nx, ny)]
            for side in (-1, 1), minus_face in minus_faces,
                    plus_face in plus_faces

                minus_face.side == side == plus_face.side || continue
                coefficient = minus_face.fraction * plus_face.fraction / 2
                add_amr_flux_weight!(
                    coefficients, minus_face.local_index, plus_face.local_index,
                    flux_index, coefficient,
                )
            end
        end
    end
    return coefficients
end

function reference_amr_flux_weights(
        mesh, Axis, ::Type{T}, incidences, global_to_local,
    ) where {T}
    coefficients = reference_amr_flux_coefficients(
        mesh, Axis, T, incidences, global_to_local,
    )
    keys_in_order = sort!(
        [key for key in keys(coefficients) if !iszero(coefficients[key])];
        by = key -> (key[1], key[2], mesh.faces[key[3]].axis, key[3]),
    )
    return [
        (key[1], key[2], key[3], coefficients[key]) for key in keys_in_order
    ]
end

function add_amr_flux_weight!(coefficients, first, second, flux, coefficient)
    first == second && throw(ArgumentError("degenerate AMR dual connection"))
    key = first < second ? (first, second, flux) : (second, first, flux)
    coefficient = first < second ? coefficient : -coefficient
    coefficients[key] = get(coefficients, key, zero(coefficient)) + coefficient
    return coefficients
end

"""Nonzero flux weights of `weights` as `(minus, plus, flux, value)` in the
builder's storage order."""
function amr_flux_weight_entries(weights, number_of_unknowns)
    entries = Tuple{Int, Int, Int, eltype(weights.weights)}[]
    for minus in 1:number_of_unknowns
        for position in weights.offsets[minus]:(
                weights.offsets[minus] + weights.counts[minus] - 1
            )

            weight = weights.weights[position]
            iszero(weight) && continue
            push!(
                entries,
                (minus, weights.plus[position], weights.flux[position], weight),
            )
        end
    end
    return entries
end

@testset "AMR dual flux weights match a Dict accumulation" begin
    refine = (key, origin, extent) ->
    key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0)
    meshes = (
        BlockTreeMesh(block_size = (4, 4), max_level = 2, refine = refine),
        BlockTreeMesh(
            block_size = (4, 4), periodic = (false, false), max_level = 2,
            refine = refine,
        ),
    )
    for mesh in meshes, Axis in (1, 2)
        incidences, global_to_local, volumes = TwoPhaseAMR._component_incidences(
            mesh, Val(Axis), Float64,
        )
        weights = TwoPhaseAMR._amr_flux_weights(
            mesh, Val(Axis), Float64, incidences, global_to_local, length(volumes),
        )
        @test weights isa TwoPhaseAMR._AMRFluxWeights{Float64}
        @test Base.return_types(
            TwoPhaseAMR._amr_flux_weights,
            (
                typeof(mesh), Val{Axis}, Type{Float64}, typeof(incidences),
                typeof(global_to_local), Int,
            ),
        ) == [TwoPhaseAMR._AMRFluxWeights{Float64}]
        expected = reference_amr_flux_weights(
            mesh, Axis, Float64, incidences, global_to_local,
        )
        @test !isempty(expected)
        # Same keys in the same order, zero sums skipped, and bitwise
        # identical sums.
        @test amr_flux_weight_entries(weights, length(volumes)) == expected
        @test_throws ArgumentError TwoPhaseAMR._amr_flux_weights(
            mesh, Val(Axis), Float64, incidences, global_to_local, 0,
        )
    end
end

"""Cell incidences of component `Axis` with the open-strip side fractions: the
inputs `_open_strip_phase_schedule` corrects, which the builder consumes."""
function open_strip_incidences(geometry, ::Type{T}, Axis) where {T}
    mesh = geometry.mesh
    incidences, global_to_local, _ = TwoPhaseAMR._component_incidences(
        mesh, Val(Axis), T,
    )
    nx, ny = mesh.block_size
    local_index = length(geometry.dual_volumes[Axis])
    for (boundary_index, face) in pairs(mesh.boundary_faces)
        face.axis == Axis || continue
        local_index += 1
        fraction = T(face.area / mesh.blocks[face.cell.block].spacing[3 - Axis])
        push!(
            incidences[TwoPhaseAMR._cell_index(face.cell, nx, ny)],
            TwoPhaseAMR.ComponentFaceIncidence(
                local_index, length(mesh.faces) + boundary_index, face.side,
                fraction,
            ),
        )
    end
    for (cell_index, cell_faces) in pairs(incidences)
        block = (cell_index - 1) ÷ (nx * ny) + 1
        local_cell = (cell_index - 1) % (nx * ny)
        ref = CellRef(block, local_cell % nx + 1, local_cell ÷ nx + 1)
        open = TwoPhaseAMR._get(geometry.open_fraction, ref)
        zero(T) < open < one(T) || continue
        for index in eachindex(cell_faces)
            incidence = cell_faces[index]
            incidence.fraction < one(T) || continue
            face = incidence.global_index <= length(mesh.faces) ?
                mesh.faces[incidence.global_index] :
                mesh.boundary_faces[incidence.global_index - length(mesh.faces)]
            lo, hi = TwoPhaseAMR._face_local_tangent_interval(mesh, ref, face)
            bounds = Axis == 1 ? (-0.5, 0.5, lo, hi) : (lo, hi, -0.5, 0.5)
            fraction = T(
                TwoPhaseAMR._open_subcell_area(geometry.solid, ref, bounds...) /
                    open,
            )
            cell_faces[index] = TwoPhaseAMR.ComponentFaceIncidence(
                incidence.local_index, incidence.global_index, incidence.side,
                fraction,
            )
        end
    end
    return incidences, global_to_local
end

@testset "open strip phase schedule matches a Dict accumulation" begin
    mesh = BlockTreeMesh(
        base_mesh = (8, 8), block_size = (4, 4), extent = (8.0, 8.0),
        max_level = 1,
        refine = (key, origin, extent) -> key == BlockKey(0, 1, 1),
    )
    geometry = GeometryField(
        mesh, (x, y) -> y - 5.3 - 0.2x; lipschitz = hypot(1, 0.2),
    )
    corrected = false
    for Axis in (1, 2)
        geometric = TwoPhaseAMR._geometry_phase_schedule(geometry, Val(Axis))
        schedule = TwoPhaseAMR._geometry_phase_schedule(geometry, Val(Axis))
        TwoPhaseAMR._open_strip_phase_schedule(schedule, geometry, Val(Axis))
        corrected |= schedule.flux_weights != geometric.flux_weights
        incidences, global_to_local = open_strip_incidences(
            geometry, Float64, Axis,
        )
        coefficients = reference_amr_flux_coefficients(
            mesh, Axis, Float64, incidences, global_to_local,
        )
        expected = fill(NaN, length(schedule.flux_weights))
        for face in schedule.faces
            for pointer in face.first_weight:(
                    face.first_weight + face.weight_count - 1
                )

                expected[pointer] = coefficients[
                    (face.minus, face.plus, schedule.flux_indices[pointer]),
                ]
            end
        end
        @test schedule.flux_weights == expected
    end
    # A cut coarse cell next to a split side has open strips that differ from
    # the geometric side fractions, so the corrected path ran.
    @test corrected
end

@testset "AMR staggered flux compatibility and stability" begin
    @test default_amr_interface_scheme(Upwind()) isa Upwind
    @test default_amr_interface_scheme(Centred()) isa MonotonizedCentral
    @test default_amr_interface_scheme(LaxWendroff()) isa MonotonizedCentral
    @test default_amr_interface_scheme(Minmod()) isa Minmod
    @test default_amr_interface_scheme(:upwind) isa Upwind
    @test default_amr_interface_scheme(:centered) isa MonotonizedCentral
    @test default_amr_interface_scheme(:lax_wendroff) isa MonotonizedCentral
    @test default_amr_interface_scheme(:superbee) === :superbee

    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
    )
    levels = sort!(unique(block.key.level for block in mesh.blocks))
    @test levels == [1, 2]

    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ?
            0.2 + sinpi(2x) * cospi(4y) :
            -0.1 + cospi(4x) * sinpi(2y),
    )
    cell_divergence = CellField(mesh)
    divergence!(cell_divergence, velocity)

    for Axis in (1, 2)
        constant_field = FaceField{Axis}(mesh; init = 1)
        tendency = FaceField{Axis}(mesh)
        workspace = StaggeredAdvectionWorkspace(constant_field)
        @test all(
            dual_face -> all(
                index -> mesh.faces[workspace.schedule.flux_indices[index]].axis ==
                    dual_face.axis,
                dual_face.first_weight:(
                    dual_face.first_weight +
                        dual_face.weight_count - 1
                )
            ),
            workspace.schedule.faces
        )
        expected_divergence =
            interpolate_cell_to_component_faces(cell_divergence, Axis)
        convection!(
            tendency, constant_field, velocity, 0.0, workspace;
            scheme = Centred()
        )

        # This is D_s I_F = I_C D. It is also the row sum of the
        # convection matrix and must hold at hanging faces, not only away
        # from the refinement interface.
        @test tendency.interior ≈ expected_divergence atol = 8.0e-14 rtol = 8.0e-14
        @test abs(integrate(tendency, workspace)) <= 8.0e-14

        n = length(constant_field.interior)
        basis = FaceField{Axis}(mesh)
        column = FaceField{Axis}(mesh)
        operator = zeros(n, n)
        for source in 1:n
            fill!(basis.interior, 0)
            basis.interior[source] = 1
            convection!(
                column, basis, velocity, 0.0, workspace;
                scheme = Centred(), amr_interface_scheme = Centred()
            )
            operator[:, source] .= column.interior
        end
        # A centred conservative flux puts half of its dual divergence on
        # the matrix diagonal. Recovering the primary divergence here is the
        # defining AMR compatibility test.
        @test 2diag(operator) ≈ expected_divergence atol = 8.0e-14 rtol = 8.0e-14

        smooth = FaceField{Axis}(
            mesh, (x, y) -> 1 + 0.2sinpi(2x) * cospi(2y)
        )
        for scheme in (Upwind(), Centred(), LaxWendroff())
            convection!(tendency, smooth, velocity, 0.01, workspace; scheme)
            @test abs(integrate(tendency, workspace)) <= 8.0e-14
        end
    end

    # The variable-coefficient LW correction changes the transported state,
    # not the conservative flux pairing. Exercise the complete AMR update for
    # both coordinate orders and both velocity components: every connection
    # must still contribute equal-and-opposite integrated momentum to its two
    # adjacent dual volumes, including at hanging faces.
    transported = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ?
            1 + 0.15sinpi(2x) * cospi(2y) :
            -0.5 + 0.1cospi(2x) * sinpi(2y),
    )
    configurations = (
        (LaxWendroff(), LaxWendroff()), (LaxWendroff(), MonotonizedCentral()), (Upwind(), Upwind()),
    )
    for (bulk_scheme, interface_scheme) in configurations,
            sweep_forward in (true, false)
        evolved = copy(transported)
        workspace = StaggeredAdvectionWorkspace(evolved)
        workspace.sweep_forward[] = sweep_forward
        initial_momentum = (
            integrate(FaceField{1}(evolved), workspace.xworkspace),
            integrate(FaceField{2}(evolved), workspace.yworkspace),
        )
        advect!(
            evolved, velocity, 0.01, workspace;
            scheme = bulk_scheme, amr_interface_scheme = interface_scheme
        )
        final_momentum = (
            integrate(FaceField{1}(evolved), workspace.xworkspace),
            integrate(FaceField{2}(evolved), workspace.yworkspace),
        )
        @test final_momentum[1] ≈ initial_momentum[1] atol = 2.0e-13
        @test final_momentum[2] ≈ initial_momentum[2] atol = 2.0e-13
    end

    # A constant velocity is discretely divergence free even when a coarse
    # cell face is represented by two fine owned segments. Centred transport
    # retains its weighted skew identity, while upwind damps an alternating
    # interface mode and is monotone under the returned CFL bound.
    constant_velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? 0.3 : -0.2
    )
    for Axis in (1, 2)
        field = FaceField{Axis}(
            mesh, (x, y) -> 1 + 0.2sinpi(2x) * cospi(2y)
        )
        workspace = StaggeredAdvectionWorkspace(field)
        volumes = dual_volumes(mesh, Axis)
        n = length(field.interior)
        basis = FaceField{Axis}(mesh)
        column = FaceField{Axis}(mesh)
        operator = zeros(n, n)
        for source in 1:n
            fill!(basis.interior, 0)
            basis.interior[source] = 1
            convection!(
                column, basis, constant_velocity, 0.0, workspace;
                scheme = Centred(), amr_interface_scheme = Centred()
            )
            operator[:, source] .= column.interior
        end
        weighted_operator = volumes .* operator
        @test weighted_operator ≈ -transpose(weighted_operator) atol = 8.0e-14 rtol = 8.0e-14

        alternating = FaceField{Axis}(
            mesh, (x, y) -> isodd(round(Int, 32x) + round(Int, 32y)) ? -1.0 : 1.0
        )
        damping = FaceField{Axis}(mesh)
        convection!(
            damping, alternating, constant_velocity, 0.0, workspace;
            scheme = Upwind()
        )
        @test staggered_inner_product(alternating, damping, workspace) > 0

        lower, upper = extrema(field.interior)
        dt = stable_timestep(constant_velocity, workspace; cfl = 0.8)
        advect!(field, constant_velocity, dt, workspace; scheme = Upwind())
        @test minimum(field.interior) >= lower - 16eps()
        @test maximum(field.interior) <= upper + 16eps()

        default_interface = FaceField{Axis}(mesh)
        mc_interface = FaceField{Axis}(mesh)
        lw_interface = FaceField{Axis}(mesh)
        symbol_lw_interface = FaceField{Axis}(mesh)
        inherited_upwind_interface = FaceField{Axis}(mesh)
        explicit_global_upwind_interface = FaceField{Axis}(mesh)
        convection!(default_interface, field, constant_velocity, dt, workspace)
        convection!(
            mc_interface, field, constant_velocity, dt, workspace;
            scheme = LaxWendroff(), amr_interface_scheme = MonotonizedCentral()
        )
        convection!(
            lw_interface, field, constant_velocity, dt, workspace;
            scheme = LaxWendroff(), amr_interface_scheme = LaxWendroff()
        )
        convection!(
            symbol_lw_interface, field, constant_velocity, dt,
            workspace; scheme = :lw, amr_interface_scheme = :lw
        )
        convection!(
            inherited_upwind_interface, field, constant_velocity, dt,
            workspace; scheme = Upwind()
        )
        convection!(
            explicit_global_upwind_interface, field, constant_velocity,
            dt, workspace; scheme = Upwind(),
            amr_interface_scheme = Upwind()
        )
        @test default_interface.interior == mc_interface.interior
        @test lw_interface.interior != default_interface.interior
        @test symbol_lw_interface.interior == lw_interface.interior
        @test inherited_upwind_interface.interior ==
            explicit_global_upwind_interface.interior

        default_step = copy(field)
        explicit_mc_step = copy(field)
        effective_rate = FaceField{Axis}(mesh)
        rate_step = copy(field)
        convection!(
            effective_rate, field, constant_velocity, dt, workspace;
            scheme = LaxWendroff(), amr_interface_scheme = MonotonizedCentral()
        )
        rate_step.interior .-= dt .* effective_rate.interior
        advect!(default_step, constant_velocity, dt, workspace)
        workspace.sweep_forward[] = true
        advect!(
            explicit_mc_step, constant_velocity, dt, workspace;
            scheme = LaxWendroff(), amr_interface_scheme = MonotonizedCentral()
        )
        @test default_step.interior == explicit_mc_step.interior
        @test rate_step.interior ≈ explicit_mc_step.interior atol = 8eps()

        @test_throws ArgumentError convection!(
            lw_interface, field, constant_velocity, dt, workspace;
            scheme = LaxWendroff(), amr_interface_scheme = Upwind()
        )

        @test_throws ArgumentError convection!(
            lw_interface, field, constant_velocity, dt, workspace;
            scheme = LaxWendroff(), amr_interface_scheme = :unknown
        )
    end

    transported_velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ?
            1 + 0.1sinpi(2x) * cospi(2y) :
            -0.5 + 0.2cospi(2x) * sinpi(2y),
    )
    default_vector_step = copy(transported_velocity)
    explicit_vector_step = copy(transported_velocity)
    default_vector_workspace =
        StaggeredAdvectionWorkspace(default_vector_step)
    explicit_vector_workspace =
        StaggeredAdvectionWorkspace(explicit_vector_step)
    vector_dt = stable_timestep(
        constant_velocity, default_vector_workspace; cfl = 0.5
    )
    advect!(
        default_vector_step, constant_velocity, vector_dt,
        default_vector_workspace
    )
    advect!(
        explicit_vector_step, constant_velocity, vector_dt,
        explicit_vector_workspace; scheme = LaxWendroff(),
        amr_interface_scheme = MonotonizedCentral()
    )
    @test default_vector_step.interior == explicit_vector_step.interior

    default_vector_rate = copy(transported_velocity)
    explicit_vector_rate = copy(transported_velocity)
    convection!(
        default_vector_rate, transported_velocity, constant_velocity,
        vector_dt, default_vector_workspace
    )
    convection!(
        explicit_vector_rate, transported_velocity, constant_velocity,
        vector_dt, explicit_vector_workspace; scheme = LaxWendroff(),
        amr_interface_scheme = MonotonizedCentral()
    )
    @test default_vector_rate.interior == explicit_vector_rate.interior
    @test_throws ArgumentError advect!(
        explicit_vector_step, constant_velocity, vector_dt,
        explicit_vector_workspace; scheme = LaxWendroff(), amr_interface_scheme = Upwind()
    )
end


@testset "stationary AMR-interface extremum probe" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
    )
    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? 0.3 : -0.2
    )

    for Axis in (1, 2)
        field = FaceField{Axis}(mesh)
        tendency = FaceField{Axis}(mesh)
        workspace = StaggeredAdvectionWorkspace(field)
        interface_incidence = zeros(Int, length(field.interior))
        for face in workspace.schedule.faces
            face.refinement_interface || continue
            interface_incidence[face.minus] += 1
            interface_incidence[face.plus] += 1
        end
        component_centres = [
            face.center for face in mesh.faces
                if face.axis == Axis
        ]
        inside(centre) = all(
            mesh.origin[d] < centre[d] < mesh.origin[d] + mesh.extent[d]
                for d in (1, 2)
        )
        # Most-coupled interface node away from the domain boundary; ties are
        # broken by face order, which the probe must not depend on.
        best = maximum(interface_incidence)
        source = findfirst(
            i -> interface_incidence[i] == best && inside(component_centres[i]),
            eachindex(interface_incidence),
        )
        @test !isnothing(source)
        @test interface_incidence[source] >= 4
        field.interior[source] = 1
        zero_nodes = findall(iszero, field.interior)

        # This is a frozen spatial-operator test. Since dq/dt = -C(q), a
        # positive convection rate at a zero node creates a new negative
        # extremum for every positive forward step, however small.
        convection!(
            tendency, field, velocity, 0.0, workspace;
            scheme = LaxWendroff(), amr_interface_scheme = LaxWendroff()
        )
        @test maximum(tendency.interior[zero_nodes]) > 0.5

        convection!(
            tendency, field, velocity, 0.0, workspace;
            scheme = LaxWendroff(), amr_interface_scheme = Fromm()
        )
        @test maximum(tendency.interior[zero_nodes]) > 0.25

        for interface_scheme in (Minmod(), MonotonizedCentral(), VanLeer(), Superbee())
            convection!(
                tendency, field, velocity, 0.0, workspace;
                scheme = LaxWendroff(), amr_interface_scheme = interface_scheme
            )
            @test maximum(tendency.interior[zero_nodes]) <= 32eps()
            @test abs(integrate(tendency, workspace)) <= 8.0e-14
        end

        symbol_tendency = FaceField{Axis}(mesh)
        convection!(
            symbol_tendency, field, velocity, 0.0, workspace;
            scheme = :lw, amr_interface_scheme = :mc
        )
        convection!(
            tendency, field, velocity, 0.0, workspace;
            scheme = LaxWendroff(), amr_interface_scheme = MonotonizedCentral()
        )
        @test symbol_tendency.interior == tendency.interior
    end
end


@testset "mixed-level staggered convection convergence" begin
    function convection_error(block_size, scheme, interface_scheme = scheme)
        mesh = BlockTreeMesh(
            block_size = (block_size, block_size), max_level = 2,
            refine = (key, origin, extent) ->
            key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
        )
        a, b = 0.3, -0.2
        velocity = FaceVelocity(
            mesh, (x, y, time, axis) -> axis == 1 ? a : b
        )
        field = FaceField{1}(mesh, (x, y) -> sinpi(2x) * cospi(2y))
        tendency = FaceField{1}(mesh)
        workspace = StaggeredAdvectionWorkspace(field)
        dt = 0.05 / block_size
        convection!(
            tendency, field, velocity, dt, workspace;
            scheme, amr_interface_scheme = interface_scheme
        )

        volumes = dual_volumes(mesh, 1)
        squared_error = zero(eltype(field))
        local_index = 0
        for face in mesh.faces
            face.axis == 1 || continue
            local_index += 1
            x, y = face.center
            exact = 2pi * (
                a * cospi(2x) * cospi(2y) -
                    b * sinpi(2x) * sinpi(2y)
            )
            squared_error += volumes[local_index] *
                (tendency.interior[local_index] - exact)^2
        end
        return sqrt(squared_error / sum(volumes))
    end

    for (scheme, minimum_rate) in ((LaxWendroff(), 1.3), (MonotonizedCentral(), 1.3))
        errors = [
            convection_error(block_size, scheme)
                for block_size in (4, 8, 16)
        ]
        rates = log2.(errors[1:2] ./ errors[2:3])
        @test all(>(minimum_rate), rates)
    end


    # MC is the representative metric-aware three-point closure. The other
    # limiters have focused formula, boundedness, and dispatch checks above.
    errors = [
        convection_error(block_size, LaxWendroff(), MonotonizedCentral())
            for block_size in (4, 8, 16)
    ]
    rates = log2.(errors[1:2] ./ errors[2:3])
    @test all(>(1.35), rates)
end


@testset "directionally split AMR amplification" begin
    adaptive_mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
    )
    uniform_mesh = BlockTreeMesh(block_size = (16, 16))

    function amplification_radius(
            mesh, ::Val{Axis}, scheme,
            interface_scheme, sweep_forward
        ) where {Axis}
        velocity = FaceVelocity(
            mesh, (x, y, time, axis) -> axis == 1 ? 0.3 : -0.2
        )
        basis = FaceField{Axis}(mesh)
        column = FaceField{Axis}(mesh)
        workspace = StaggeredAdvectionWorkspace(basis)
        dt = stable_timestep(velocity, workspace; cfl = 0.8)
        n = length(basis.interior)
        convection_matrix = zeros(n, n)
        workspace.sweep_forward[] = sweep_forward
        for source in 1:n
            fill!(basis.interior, 0)
            basis.interior[source] = 1
            convection!(
                column, basis, velocity, dt, workspace;
                scheme,
                amr_interface_scheme = interface_scheme
            )
            convection_matrix[:, source] .= column.interior
        end
        update_matrix = Matrix(I, n, n) - dt * convection_matrix
        return maximum(abs, eigvals(update_matrix))
    end

    for Axis in (1, 2), sweep_forward in (true, false)
        uniform_radius = amplification_radius(
            uniform_mesh, Val(Axis), LaxWendroff(), LaxWendroff(), sweep_forward
        )
        upwind_radius = amplification_radius(
            adaptive_mesh, Val(Axis), Upwind(), Upwind(), sweep_forward
        )
        lw_radius = amplification_radius(
            adaptive_mesh, Val(Axis), LaxWendroff(), LaxWendroff(), sweep_forward
        )
        @test uniform_radius <= 1 + 2.0e-12
        @test upwind_radius <= 1 + 2.0e-12
        @test lw_radius <= 1 + 2.0e-12
    end
end

@testset "centred convection matrix" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    velocity = FaceVelocity(
        mesh,
        (x, y, t, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y),
    )
    for Axis in (1, 2)
        basis = FaceField{Axis}(mesh)
        column = FaceField{Axis}(mesh)
        workspace = StaggeredAdvectionWorkspace(basis)
        n = length(basis.interior)
        operator = zeros(n, n)
        for source in 1:n
            fill!(basis.interior, 0)
            basis.interior[source] = 1
            convection!(column, basis, velocity, 0.0, workspace; scheme = Centred())
            operator[:, source] .= column.interior
        end
        @test operator ≈ -transpose(operator) atol = 2.0e-14 rtol = 2.0e-14
    end
end

@testset "Lax-Wendroff convergence" begin
    function translation_error(nx)
        mesh = BlockTreeMesh(block_size = (nx, 4))
        speed = 0.5
        velocity = FaceVelocity(
            mesh, (x, y, t, axis) -> axis == 1 ? speed : 0.0
        )
        field = FaceField{1}(mesh, (x, y) -> sinpi(2x))
        workspace = StaggeredAdvectionWorkspace(field)
        final_time = 0.2
        steps = ceil(Int, final_time / (0.4 / (nx * speed)))
        dt = final_time / steps
        for _ in 1:steps
            advect!(field, velocity, dt, workspace; scheme = LaxWendroff())
        end

        squared_error = 0.0
        local_index = 0
        for face in mesh.faces
            face.axis == 1 || continue
            local_index += 1
            exact = sinpi(2(face.center[1] - speed * final_time))
            squared_error += (field.interior[local_index] - exact)^2
        end
        return sqrt(squared_error / length(field.interior))
    end

    errors = translation_error.((16, 32, 64))
    @test log2(errors[1] / errors[2]) > 1.9
    @test log2(errors[2] / errors[3]) > 1.9
end

@testset "centred staggered convection is skew-symmetric" begin
    mesh = BlockTreeMesh(block_size = (16, 16))
    velocity = FaceVelocity(
        mesh,
        (x, y, t, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y),
    )
    divergence = CellField(mesh)
    divergence!(divergence, velocity)
    @test maximum(abs, interior(divergence, 1)) <= 64eps()

    for Axis in (1, 2)
        a = FaceField{Axis}(
            mesh, (x, y) -> sinpi(4x) + cospi(2y) + 0.3sinpi(2x + 2y)
        )
        b = FaceField{Axis}(
            mesh, (x, y) -> cospi(6x) - 0.2sinpi(4y) + 0.1cospi(2x - 4y)
        )
        Ca = FaceField{Axis}(mesh)
        Cb = FaceField{Axis}(mesh)
        workspace = StaggeredAdvectionWorkspace(a)
        convection!(Ca, a, velocity, 0.0, workspace; scheme = :centred)
        convection!(Cb, b, velocity, 0.0, workspace; scheme = Centred())

        lhs = staggered_inner_product(a, Cb, workspace)
        rhs = -staggered_inner_product(Ca, b, workspace)
        @test lhs ≈ rhs atol = 2.0e-14 rtol = 2.0e-14
        @test abs(staggered_inner_product(a, Ca, workspace)) <= 2.0e-14

        constant_field = FaceField{Axis}(mesh; init = 1)
        convection!(Cb, constant_field, velocity, 0.0, workspace; scheme = Centred())
        @test maximum(abs, Cb.interior) <= 64eps()
    end
end

@testset "staggered transport is independent of uniform block partitioning" begin
    oneblock = BlockTreeMesh(block_size = (8, 8))
    fourblocks = BlockTreeMesh(
        block_size = (4, 4), max_level = 1,
        refine = (key, origin, extent) -> true,
    )
    fourroots = BlockTreeMesh(base_mesh = (8, 8), block_size = (4, 4))
    velocity_function = (x, y, t, axis) -> axis == 1 ? 0.3 : -0.15
    initial = (x, y) -> 1 + 0.2sinpi(2x) * cospi(2y)

    for Axis in (1, 2), scheme in (LaxWendroff(), MonotonizedCentral())
        a = FaceField{Axis}(oneblock, initial)
        b = FaceField{Axis}(fourblocks, initial)
        c = FaceField{Axis}(fourroots, initial)
        va = FaceVelocity(oneblock, velocity_function)
        vb = FaceVelocity(fourblocks, velocity_function)
        vc = FaceVelocity(fourroots, velocity_function)
        wa = StaggeredAdvectionWorkspace(a)
        wb = StaggeredAdvectionWorkspace(b)
        wc = StaggeredAdvectionWorkspace(c)
        dt = 0.02
        advect!(a, va, dt, wa; scheme)
        advect!(b, vb, dt, wb; scheme)
        advect!(c, vc, dt, wc; scheme)
        @test face_values_by_coordinate(a) == face_values_by_coordinate(b)
        @test face_values_by_coordinate(a) == face_values_by_coordinate(c)
    end
end

@testset "staggered convection uses current boundary configuration" begin
    mesh = BlockTreeMesh(block_size = (6, 6), periodic = (false, false))
    velocity = FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? 0.75 : 0.25)
    field = FaceField{1}(mesh; init = 1)
    workspace = StaggeredAdvectionWorkspace(field)
    first_rate = FaceField{1}(mesh)
    second_rate = FaceField{1}(mesh)
    lid = NoSlip((x, y, t, axis) -> t)
    convection!(
        first_rate, field, velocity, 0.0, workspace;
        scheme = Upwind(), boundary = lid, time = 1.0
    )
    convection!(
        second_rate, field, velocity, 0.0, workspace;
        scheme = Upwind(), boundary = lid, time = 2.0
    )
    @test first_rate.interior != second_rate.interior
    convection!(
        second_rate, field, velocity, 0.0, workspace;
        scheme = Upwind(), boundary = lid, time = 1.0
    )
    @test first_rate.interior == second_rate.interior
    convection!(
        second_rate, field, velocity, 0.0, workspace;
        scheme = Upwind(), boundary = FreeSlip(), time = 1.0
    )
    @test first_rate.interior == second_rate.interior
    convection!(second_rate, field, velocity, 0.0, workspace; scheme = Upwind())
    @test first_rate.interior != second_rate.interior
end

using Test, TwoPhaseAMR, LinearAlgebra
include("helpers.jl")

const TID = TwoPhaseAMR

function _drag_flow(kind; interphase_drag = InterphaseDrag(), viscosity = (0.03, 0.002), velocity = nothing)
    geometry_case = kind in (:geometry, :mixed_geometry)
    mixed_case = kind in (:mixed, :mixed_geometry)
    mesh = BlockTreeMesh(
        block_size = geometry_case ? (6, 6) : (4, 4),
        max_level = mixed_case ? 2 : 0,
        periodic = geometry_case ? (false, false) : (true, true),
        refine = (key, origin, extent) -> mixed_case &&
            (key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0)),
    )
    geometry = geometry_case ? GeometryField(
            mesh, (x, y) -> y - 0.16 - 0.1x; lipschitz = hypot(1, 0.1),
        ) : nothing
    interface = InterfaceField(
        mesh, (x, y) -> (x - 0.5)^2 + (y - 0.55)^2 - 0.3^2;
        geometry, lipschitz = 2,
    )
    initial = isnothing(velocity) ? (
            (x, y, time, axis) -> sinpi(2x + y + axis / 3) + 0.1axis,
            (x, y, time, axis) -> 0.4 + cospi(x - 2y + axis / 5),
        ) : velocity
    flow = FlowSimulation(
        mesh,
        TwoPhase(interface; density = (5.0, 1.0), dynamic_viscosity = viscosity);
        velocity_model = TwoVelocityModel(; interphase_drag),
        geometry, velocity = initial, boundary = FreeSlip(),
        numerics = NavierStokesOptions(
            diffusion = LinearSolverOptions(preconditioner = nothing),
        ),
    )
    state = TID._two_velocity_flow_workspace!(
        flow.workspace.two_phase, flow.system.two_phase,
    )
    TID._copy_velocity!(state.convected, flow.velocity)
    return flow
end

function _drag_viscous_step!(flow, dt)
    TID._diffusion_substep!(
        flow.velocity, flow.system, flow.workspace, dt, 0.0,
        flow.system.two_phase; tolerance = 1.0e-12, max_cycles = 1000,
    )
    return (copy(flow.velocity[1].interior), copy(flow.velocity[2].interior))
end

function _drag_phase_masses(flow)
    model = flow.system.two_phase
    interface = model.interface
    return map(interface.mesh.faces) do face
        c = TID._staggered_liquid_fraction(interface, interface.geometry, face)
        (model.density[1] * c, model.density[2] * (1 - c))
    end
end

@testset "Interphase drag geometry" begin
    # Liquid part of the unit cell below a horizontal line and on the left of
    # a diagonal: area, first moments and the chord.
    half = 0.5
    square = ((-half, -half), (half, -half), (half, half), (-half, half), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    clipped, count, first, second = TID._clip_convex(square, 4, 0.0, 1.0, 0.1, true)
    area, moment_x, moment_y = TID._tuple_polygon_area_moments(clipped, count, Float64)
    @test count == 4 && area ≈ 0.6
    @test abs(moment_x) <= 1.0e-15
    @test moment_y ≈ 0.6 * (0.1 - 0.5) / 2
    @test hypot((second .- first)...) ≈ 1
    clipped, count, first, second = TID._clip_convex(square, 4, 1 / sqrt(2), 1 / sqrt(2), 0.0, true)
    area, moment_x, moment_y = TID._tuple_polygon_area_moments(clipped, count, Float64)
    @test area ≈ 0.5 && hypot((second .- first)...) ≈ sqrt(2)
    @test moment_x ≈ -1 / 12 && moment_y ≈ -1 / 12

    # A flat interface parallel to a face lies in its control volume with the
    # face length and a centroid distance of half a cell; one normal to it
    # gives half the chord at the distance of one cell, and on a grid line
    # (the periodic seam) the face itself at the cell-centre distance. The
    # tangential weight is one for the tangential and zero for the normal
    # component, also on the grid line.
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, true))
    h = 1 / 8
    for (levelset, axis, expected, tangential_weight) in (
            ((x, y) -> y - 0.5 - 0.3h, 1, (h, h / 2), 1),
            ((x, y) -> x - 0.5 - 0.3h, 2, (h, h / 2), 1),
            ((x, y) -> x - 0.5 - 0.3h, 1, (h / 2, h), 0),
        )
        interface = InterfaceField(mesh, levelset; lipschitz = 1)
        checked = 0
        for face in mesh.faces
            face.axis == axis || continue
            c = TID._staggered_liquid_fraction(interface, nothing, face)
            0 < c < 1 || continue
            grid_line = all(
                ref -> TID._conditional_fraction(interface, ref) in (0, 1),
                (face.minus, face.plus),
            )
            contact = TID._interphase_contact(interface, face, false)
            @test contact[1] ≈ (grid_line ? h : expected[1])
            @test contact[2] ≈ (grid_line ? h : expected[2])
            @test contact[4] ≈ c
            tangential = TID._interphase_contact(interface, face, true)
            @test tangential[3] ≈ tangential_weight atol = 1.0e-14
            checked += 1
        end
        @test checked > 0
    end

    # The decay rate of the horizontal interface: |Gamma| = h, delta = h/2,
    # harmonic mu_Gamma at the staggered fraction, masses rho c h^2.
    interface = InterfaceField(mesh, (x, y) -> y - 0.5 - 0.3h; lipschitz = 1)
    density, viscosity = (5.0, 1.0), (0.03, 0.002)
    face = mesh.faces[findfirst(f -> f.axis == 1 && 0 < TID._staggered_liquid_fraction(interface, nothing, f) < 1, mesh.faces)]
    c = TID._staggered_liquid_fraction(interface, nothing, face)
    mu = 1 / (c / viscosity[1] + (1 - c) / viscosity[2])
    expected = mu * h / (h / 2) / h^2 * (1 / (density[1] * c) + 1 / (density[2] * (1 - c)))
    for tangential in (false, true)
        @test TID._interphase_drag_rate(
            interface, face, c, h^2, density, viscosity, HarmonicViscosityAveraging(), tangential,
        ) ≈ expected rtol = 1.0e-12
    end

    # Orientation: per unit chord length, a tangential jump gets the same
    # coupling from the tangential weight as from the full component (the
    # weight sums t_k^2 w_k to one), and a normal jump none where the
    # interface is grid-aligned. The remaining dependence on the angle is the
    # centroid distance of the control volume, which makes the stress of a
    # resolved linear shear exact.
    function coupling(theta; n = 32)
        mesh = BlockTreeMesh(block_size = (n, n), periodic = (false, false))
        c, s = cos(theta), sin(theta)
        interface = InterfaceField(mesh, (x, y) -> -s * (x - 0.5) + c * (y - 0.5 - 0.013); lipschitz = 1)
        inside(p) = abs(p[1] - 0.5) < 0.3 && abs(p[2] - 0.5) < 0.3
        tangential = normal = full = 0.0
        for face in mesh.faces
            inside(face.center) && 0 < TID._staggered_liquid_fraction(interface, nothing, face) < 1 || continue
            t, m = face.axis == 1 ? (c, -s) : (s, c)
            L, d, w, _ = TID._interphase_contact(interface, face, true)
            Lf, df, _, _ = TID._interphase_contact(interface, face, false)
            tangential += L * w * t^2 / d
            normal += L * w * m^2 / d
            full += Lf * t^2 / df
        end
        return tangential / full, normal / full
    end
    for (theta, leak) in ((0.0, 0.0), (pi / 6, nothing), (pi / 4, 1.0))
        ratio, normal = coupling(theta)
        @test 0.95 <= ratio <= 1.05
        isnothing(leak) || @test normal ≈ leak atol = 1.0e-8
    end

    @test_throws ArgumentError InterphaseDrag(; component = :normal)
    @test_throws ArgumentError TwoVelocityModel(; interphase_drag = :on)
    @test TwoVelocityModel().interphase_drag == InterphaseDrag()
    @test TwoVelocityModel(; interphase_drag = InterphaseDrag()).interphase_drag == InterphaseDrag(; component = :tangential)
    @test isnothing(TwoVelocityModel(; interphase_drag = nothing).interphase_drag)
end

@testset "Interphase drag control volume on transitions, cut cells and small bubbles" begin
    # The control volume is the one of the staggered fraction: on a level
    # transition and next to a solid its liquid fraction is that fraction,
    # and every mixed face gets a contact.
    for kind in (:mixed, :geometry, :mixed_geometry)
        flow = _drag_flow(kind)
        interface = flow.system.two_phase.interface
        mixed = transitions = 0
        for face in interface.mesh.faces
            c = TID._staggered_liquid_fraction(interface, interface.geometry, face)
            0 < c < 1 || continue
            contact = TID._interphase_contact(interface, face, false)
            mixed += 1
            @test contact[1] > 0
            @test contact[4] ≈ c rtol = 1.0e-10
            transitions += interface.mesh.blocks[face.minus.block].spacing !=
                interface.mesh.blocks[face.plus.block].spacing
        end
        @test mixed > 0
        kind === :mixed && @test transitions > 0
    end

    # A horizontal interface through a column cut by a vertical wall: the open
    # width a of the cut cell and the full neighbour give |Gamma| = (a + h)/2
    # on the face between them, at the distance h/2.
    n = 8
    h = 1 / n
    mesh = BlockTreeMesh(block_size = (n, n), periodic = (false, false))
    geometry = GeometryField(mesh, (x, y) -> x - 0.25 - 0.4h; lipschitz = 1)
    interface = InterfaceField(mesh, (x, y) -> y - 0.5 - 0.3h; geometry, lipschitz = 1)
    face = only(
        filter(
            f -> f.axis == 1 && f.center[1] ≈ 0.375 && 0.5 < f.center[2] < 0.625,
            mesh.faces,
        ),
    )
    contact = TID._interphase_contact(interface, face, true)
    @test contact[1] ≈ (0.6h + h) / 2 rtol = 1.0e-8
    @test contact[2] ≈ h / 2 rtol = 1.0e-8
    @test contact[3] ≈ 1 rtol = 1.0e-8
    @test contact[4] ≈ TID._staggered_liquid_fraction(interface, geometry, face) rtol = 1.0e-10

    # A liquid film thinner than a cell: the two cells of an x-face hold
    # opposite interfaces, and the centroid distance is raised to the floor
    # of a quarter cell.
    mesh = BlockTreeMesh(block_size = (32, 32), periodic = (true, true))
    interface = InterfaceField(mesh, (x, y) -> abs(x - 0.5) - 0.3 / 32; lipschitz = 1)
    distances = Float64[]
    for face in mesh.faces
        0 < TID._staggered_liquid_fraction(interface, nothing, face) < 1 || continue
        push!(distances, TID._interphase_contact(interface, face, false)[2])
    end
    @test minimum(distances) ≈ 1 / 128
    @test count(d -> d ≈ 1 / 128, distances) > 0
end

@testset "Interphase drag relaxes the jump at fixed mixture momentum" begin
    dt = 0.002
    for kind in (:uniform, :mixed, :geometry, :mixed_geometry),
            drag in (InterphaseDrag(), InterphaseDrag(; component = :full))
        with = _drag_flow(kind; interphase_drag = drag)
        without = _drag_flow(kind; interphase_drag = nothing)
        before = (copy(with.velocity[1].interior), copy(with.velocity[2].interior))
        on = _drag_viscous_step!(with, dt)
        off = _drag_viscous_step!(without, dt)
        @test _phase_storage_contract(with.velocity, with.system.two_phase.interface)
        masses = _drag_phase_masses(with)
        coupled = untouched_changes = sign_changes = 0
        momentum_defect = jump_growth = 0.0
        jump_energy_on = jump_energy_off = jump_energy_before = 0.0
        for index in eachindex(masses)
            liquid_mass, gas_mass = masses[index]
            if !(isfinite(off[1][index]) && isfinite(off[2][index]))
                # A face with one phase only is untouched.
                untouched_changes += !isequal(on[1][index], off[1][index]) +
                    !isequal(on[2][index], off[2][index])
                continue
            end
            total = liquid_mass + gas_mass
            momentum_on = liquid_mass * on[1][index] + gas_mass * on[2][index]
            momentum_off = liquid_mass * off[1][index] + gas_mass * off[2][index]
            scale = liquid_mass * abs(off[1][index]) + gas_mass * abs(off[2][index])
            momentum_defect = max(momentum_defect, abs(momentum_on - momentum_off) / scale)
            jump_on = on[2][index] - on[1][index]
            jump_off = off[2][index] - off[1][index]
            jump_growth = max(jump_growth, abs(jump_on) - abs(jump_off))
            sign_changes += jump_on * jump_off < 0
            reduced = liquid_mass * gas_mass / total
            jump_energy_on += reduced * jump_on^2 / 2
            jump_energy_off += reduced * jump_off^2 / 2
            jump_energy_before += reduced *
                (before[2][index] - before[1][index])^2 / 2
            coupled += abs(jump_on) < abs(jump_off) * (1 - 1.0e-6)
        end
        @test untouched_changes == 0
        @test momentum_defect <= 8eps()
        @test jump_growth <= 8eps()
        @test sign_changes == 0
        @test coupled > 0
        # The mixture solve keeps the jump; the drag only removes its energy.
        @test jump_energy_off ≈ jump_energy_before rtol = 1.0e-13
        @test jump_energy_on < jump_energy_off
    end
end

@testset "Interphase drag is inactive at rest, for equal phases and without viscosity" begin
    common = ((x, y, time, axis) -> sinpi(2x) * cospi(2y + axis), (x, y, time, axis) -> sinpi(2x) * cospi(2y + axis))
    for kind in (:uniform, :mixed_geometry)
        @test isequal(
            _drag_viscous_step!(_drag_flow(kind; velocity = common), 0.002),
            _drag_viscous_step!(_drag_flow(kind; velocity = common, interphase_drag = nothing), 0.002),
        )
        rest = _drag_flow(kind; velocity = ((x, y, t, a) -> 0.0, (x, y, t, a) -> 0.0))
        result = _drag_viscous_step!(rest, 0.002)
        for phase in 1:2
            @test all(value -> isnan(value) || iszero(value), result[phase])
        end
        inviscid = _drag_flow(kind; viscosity = (0.0, 0.0))
        @test isequal(
            _drag_viscous_step!(inviscid, 0.002),
            _drag_viscous_step!(_drag_flow(kind; viscosity = (0.0, 0.0), interphase_drag = nothing), 0.002),
        )
    end
end

@testset "Interphase drag locks the phases as lambda dt grows" begin
    for kind in (:uniform, :mixed_geometry)
        flow = _drag_flow(kind)
        model = flow.system.two_phase
        operator = TID._update_dynamic_viscosity!(
            flow.workspace.two_phase.diffusion, model.interface, model.dynamic_viscosity,
        )
        before = (copy(flow.velocity[1].interior), copy(flow.velocity[2].interior))
        TID._interphase_drag!(flow.velocity, model.interphase_drag, model, operator, 1.0e12)
        masses = _drag_phase_masses(flow)
        interface = model.interface
        locked = 0
        for (index, face) in pairs(interface.mesh.faces)
            rate = TID._interphase_drag_rate(
                interface, face,
                TID._staggered_liquid_fraction(interface, interface.geometry, face),
                operator.volumes[index], model.density,
                model.dynamic_viscosity, operator.averaging,
                model.interphase_drag.component === :tangential,
            )
            liquid, gas = flow.velocity[1].interior[index], flow.velocity[2].interior[index]
            if !(rate > 0 && isfinite(before[1][index]) && isfinite(before[2][index]))
                @test isequal(liquid, before[1][index]) && isequal(gas, before[2][index])
                continue
            end
            liquid_mass, gas_mass = masses[index]
            scale = abs(before[1][index]) + abs(before[2][index])
            @test abs(gas - liquid) <= 8eps() * scale
            @test liquid_mass * liquid + gas_mass * gas ≈
                liquid_mass * before[1][index] + gas_mass * before[2][index] rtol = 8eps() atol = 8eps() * (liquid_mass + gas_mass) * scale
            locked += 1
        end
        @test locked > 0
    end
end

@testset "Interphase drag semigroup and zero step" begin
    for kind in (:mixed, :mixed_geometry)
        flow = _drag_flow(kind)
        model = flow.system.two_phase
        operator = TID._update_dynamic_viscosity!(
            flow.workspace.two_phase.diffusion, model.interface, model.dynamic_viscosity,
        )
        start = (copy(flow.velocity[1].interior), copy(flow.velocity[2].interior))
        once = (FaceVelocity(model.interface.mesh), FaceVelocity(model.interface.mesh))
        twice = (FaceVelocity(model.interface.mesh), FaceVelocity(model.interface.mesh))
        for (target, steps) in ((once, (0.4,)), (twice, (0.2, 0.2)))
            copyto!(target[1].interior, start[1]); copyto!(target[2].interior, start[2])
            for dt in steps
                TID._interphase_drag!(target, model.interphase_drag, model, operator, dt)
            end
        end
        scale = maximum(x -> isfinite(x) ? abs(x) : 0.0, start[1]) +
            maximum(x -> isfinite(x) ? abs(x) : 0.0, start[2])
        for phase in 1:2
            @test all(
                isequal(a, b) || abs(a - b) <= 16eps() * scale
                    for (a, b) in zip(once[phase].interior, twice[phase].interior)
            )
        end
        @test any(index -> !isequal(once[2].interior[index], start[2][index]), eachindex(start[2]))
        TID._interphase_drag!(flow.velocity, model.interphase_drag, model, operator, 0.0)
        @test isequal(flow.velocity[1].interior, start[1]) && isequal(flow.velocity[2].interior, start[2])
    end
end

@testset "Two-velocity viscous step without drag is bitwise the mixture step" begin
    for kind in (:uniform, :mixed_geometry)
        dt = 0.002
        default = _drag_flow(kind; interphase_drag = nothing)
        result = _drag_viscous_step!(default, dt)
        reference = _drag_flow(kind; interphase_drag = nothing)
        model = reference.system.two_phase
        phase = reference.workspace.two_phase
        state = TID._two_velocity_flow_workspace!(phase, model)
        @test any(
            index -> isfinite(state.convected[1].interior[index]) &&
                isfinite(state.convected[2].interior[index]) &&
                state.convected[1].interior[index] != state.convected[2].interior[index],
            eachindex(state.convected[1].interior),
        )
        diffusion = TID._update_dynamic_viscosity!(phase.diffusion, model.interface, model.dynamic_viscosity)
        TID._mass_weighted_one_velocity!(state.mixture_velocity, phase.density, state.convected, model)
        implicit_midpoint_diffusion!(
            state.diffused_mixture, state.mixture_velocity, diffusion, dt, phase.diffusion_solver;
            density = phase.density, tolerance = 1.0e-12, max_cycles = 1000,
        )
        TID._apply_two_velocity_diffusion_update!(
            reference.velocity, state.convected, state.diffused_mixture, state.mixture_velocity;
            dirichlet_faces = diffusion.dirichlet_faces,
        )
        TID._mark_inactive_phase_velocities!(reference.velocity, model.interface, phase.inactive_failures)
        @test isequal(result[1], reference.velocity[1].interior)
        @test isequal(result[2], reference.velocity[2].interior)
    end
end

@testset "Interphase drag slip law" begin
    # Re_j -> 0: the geometric decay exp(-lambda dt); Re_j -> Inf: no coupling.
    for decay in (1.0e-6, 0.3, 2.0, 40.0)
        @test TID._slip_jump(0.7, decay, 1.0e-12) ≈ 0.7exp(-decay) rtol = 1.0e-10
        @test abs(TID._slip_jump(-0.7, decay, 1.0e12) + 0.7) <= 2decay / 1.0e12
    end
    # The exact solution against a fine RK4 integration of dj/dt = -l j/(1 + a|j|).
    rhs(j, a) = -j / (1 + a * abs(j))
    for (j0, a, decay) in ((1.0, 3.0, 0.7), (-0.2, 50.0, 5.0), (2.0, 0.01, 1.5))
        j = j0
        steps = 20_000
        dt = decay / steps
        for _ in 1:steps
            k1 = rhs(j, a); k2 = rhs(j + dt / 2 * k1, a)
            k3 = rhs(j + dt / 2 * k2, a); k4 = rhs(j + dt * k3, a)
            j += dt / 6 * (k1 + 2k2 + 2k3 + k4)
        end
        @test TID._slip_jump(j0, decay, a) ≈ j rtol = 1.0e-10
        # Semigroup of the exact update.
        @test TID._slip_jump(TID._slip_jump(j0, decay / 2, a), decay / 2, a) ≈
            TID._slip_jump(j0, decay, a) rtol = 1.0e-13
    end
    # |j| is non-increasing in the decay and keeps its sign; zero decay keeps j.
    for a in (1.0e-3, 1.0, 1.0e3), j0 in (-1.3, 0.4)
        values = [TID._slip_jump(j0, decay, a) for decay in 0.0:0.05:30.0]
        @test values[1] ≈ j0 rtol = 4eps()
        @test all(diff(abs.(values)) .<= 4eps())
        @test all(v -> sign(v) == sign(j0) || iszero(v), values)
    end
end

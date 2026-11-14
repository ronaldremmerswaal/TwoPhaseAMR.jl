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
    # a diagonal: area, first moments and the chord vector.
    area, moment_x, moment_y, chord = TID._plic_rectangle_clip(0.0, 1.0, 0.1, -0.5, 0.5, -0.5, 0.5)
    @test area ≈ 0.6
    @test abs(moment_x) <= 1.0e-15
    @test moment_y ≈ 0.6 * (0.1 - 0.5) / 2
    @test abs.(chord) == (1.0, 0.0)
    area, moment_x, moment_y, chord = TID._plic_rectangle_clip(1 / sqrt(2), 1 / sqrt(2), 0.0, -0.5, 0.5, -0.5, 0.5)
    @test area ≈ 0.5 && hypot(chord...) ≈ sqrt(2)
    @test moment_x ≈ -1 / 12 && moment_y ≈ -1 / 12

    # A flat interface parallel to a face lies in its control volume with the
    # face length and a centroid distance of half a cell; one normal to it
    # gives half the chord at the distance of one cell, and on a grid line
    # (the periodic seam) the face itself at the cell-centre distance.
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
            tangential = TID._interphase_contact(interface, face, true)
            @test tangential[3] ≈ tangential_weight atol = 1.0e-14
            checked += 1
        end
        @test checked > 0
    end
    @test_throws ArgumentError InterphaseDrag(; component = :normal)
    @test_throws ArgumentError TwoVelocityModel(; interphase_drag = :on)
    @test isnothing(TwoVelocityModel().interphase_drag)
    @test TwoVelocityModel(; interphase_drag = InterphaseDrag()).interphase_drag == InterphaseDrag(; component = :tangential)
    @test isnothing(TwoVelocityModel(; interphase_drag = nothing).interphase_drag)
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
                interface, face, operator.volumes[index], model.density,
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

using Test, TwoPhaseAMR

struct AffineBodyForce{T}
    scale::T
end
(force::AffineBodyForce)(x, y, time, axis) =
    force.scale * (axis == 1 ? 1 + 2x - y + time : -2 + x + 3y - time)

mutable struct RecordedBodyForce{T}
    calls::Vector{NTuple{4, T}}
end
function (force::RecordedBodyForce{T})(x, y, time, axis) where {T}
    push!(force.calls, (T(x), T(y), T(time), T(axis)))
    return axis + time
end

function _body_force_allocation(velocity, force::F, dt, midpoint) where {F}
    TwoPhaseAMR._body_force_substep!(velocity, dt, midpoint, force)
    return @allocated TwoPhaseAMR._body_force_substep!(velocity, dt, midpoint, force)
end

@testset "Typed body force preserves phase data and callback semantics" begin
    for T in (Float32, Float64), periodic in ((true, true), (true, false)), mixed in (false, true)
        mesh = BlockTreeMesh(block_size = (4, 4), max_level = 2,
            origin = (zero(T), zero(T)), extent = (one(T), one(T)), periodic = periodic,
            refine = (key, origin, extent) -> mixed &&
                (key.level == 0 || (key.level == 1 && key.i == key.j == 0)))
        interface = InterfaceField(mesh, (x, y) -> y - T(0.47); T)
        phase_velocity = (FaceVelocity(mesh, T), FaceVelocity(mesh, T))
        for (phase, velocity) in pairs(phase_velocity)
            set_velocities!(velocity, (x, y, time, axis) -> T(phase + axis) + x - y)
        end
        TwoPhaseAMR._mark_inactive_phase_velocities!(phase_velocity, interface)
        shared_velocity = FaceVelocity(mesh, T)
        set_velocities!(shared_velocity, (x, y, time, axis) -> T(axis) + x - y)
        dt, midpoint = T(0.003), T(0.021)
        force = AffineBodyForce(T(0.07))
        for velocity in (shared_velocity, phase_velocity)
            fields = velocity isa FaceVelocity ? (velocity,) : velocity
            before = map(copy, fields)
            @test TwoPhaseAMR._body_force_substep!(velocity, dt, midpoint, force) === velocity
            for (actual, original) in zip(fields, before)
                @test isequal(actual.boundary, original.boundary)
                for (index, face) in pairs(mesh.faces)
                    old = original.interior[index]
                    if isnan(old)
                        @test isnan(actual.interior[index])
                    else
                        x, y = face.center
                        acceleration = force.scale * (face.axis == 1 ?
                            1 + 2x - y + midpoint : -2 + x + 3y - midpoint)
                        @test actual.interior[index] == T(old + dt * acceleration)
                    end
                end
            end
            # No-force is a true no-op: no traversal or changes even to NaNs
            # or signed zeros. This is not an absent-velocity extrapolation.
            for field in fields
                field.interior[firstindex(field.interior)] = -zero(T)
            end
            unchanged = map(copy, fields)
            @test TwoPhaseAMR._body_force_substep!(velocity, dt, midpoint, nothing) === velocity
            @test all(isequal(a.interior, b.interior) && isequal(a.boundary, b.boundary)
                for (a, b) in zip(fields, unchanged))
            @test _body_force_allocation(velocity, nothing, dt, midpoint) == 0
            @test _body_force_allocation(velocity, force, dt, midpoint) == 0
        end
        recorder = RecordedBodyForce(Tuple{T, T, T, T}[])
        original = map(copy, phase_velocity)
        TwoPhaseAMR._body_force_substep!(phase_velocity, dt, midpoint, recorder)
        expected = [(face.center[1], face.center[2], midpoint, T(face.axis))
            for velocity in original for (index, face) in pairs(mesh.faces)
            if !isnan(velocity.interior[index])]
        @test recorder.calls == expected
    end
end

@testset "Body-force midpoint update reaches shared and intrinsic stepping" begin
    for model in (OneVelocityModel(),
            TwoVelocityModel(pressure_projection = CMDGFMProjection()),
            TwoVelocityModel(pressure_projection = MimeticProjection()))
        mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, false))
        interface = InterfaceField(mesh, (x, y) -> y - 0.47)
        flow = FlowSimulation(mesh,
            TwoPhase(interface; density = (1000.0, 1.0), dynamic_viscosity = (0.0, 0.0));
            velocity_model = model, velocity = (0.01, 0.0), boundary = FreeSlip())
        force = (x, y, time, axis) -> axis == 1 ? 0.07 * (1 + 2time) : 0.0
        result = simulate!(flow, 0.004; dt = 0.001, force, tolerance = 1e-10)
        @test result.pressure.converged
        expected = 0.01 + 0.07 * (flow.time + flow.time^2)
        fields = flow.velocity isa FaceVelocity ? (flow.velocity,) : flow.velocity
        for velocity in fields, (index, face) in pairs(mesh.faces)
            value = velocity.interior[index]
            isnan(value) && continue
            @test value ≈ (face.axis == 1 ? expected : 0.0) atol = 2e-10
        end
    end
end

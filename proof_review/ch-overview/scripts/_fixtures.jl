# Small flows on the INSTRUMENTED package (`_trace.jl`) for the audit of ch:overview.
# Included by the check scripts; loads `_trace.jl` once.

isdefined(Main, :Tracer) || include(joinpath(@__DIR__, "_trace.jl"))
using .Tracer
using Test, Logging
const A = Tracer.AMR

unit_mesh(; block = (8, 8), max_level = 0, periodic = (false, false), kwargs...) =
    A.BlockTreeMesh(; block_size = block, max_level, periodic, kwargs...)

# solid {y < 0.2 + 0.1 x} (level set <= 0 is solid), clear of the drop below
solid_levelset(x, y) = y - 0.2 - 0.1x
drop_levelset(x, y) = hypot(x - 0.5, y - 0.6) - 0.2

"""Two-phase flow with a drop; keyword choices select the model of the stage lists."""
function drop_flow(;
        model = A.OneVelocityModel(), sigma = 0.07, gravity = (0.0, -1.0), viscosity = (1.0e-2, 1.0e-3),
        solid = false, boundary = A.NoSlip(), mesh = unit_mesh(), density = (1000.0, 1.0), dt = nothing,
        adaptivity = A.Adaptivity(), velocity = nothing, diagnostics = (),
    )
    geometry = solid ? A.GeometryField(mesh, solid_levelset; lipschitz = hypot(1, 0.1)) : nothing
    interface = solid ?
        A.InterfaceField(mesh, drop_levelset; geometry, contact_angle = pi / 2, lipschitz = 1) :
        A.InterfaceField(mesh, drop_levelset; lipschitz = 1)
    physics = A.TwoPhase(
        interface; density, dynamic_viscosity = viscosity,
        surface_tension = A.SurfaceTension(sigma), gravity,
    )
    return A.FlowSimulation(
        mesh, physics; velocity_model = model, boundary, dt, adaptivity, velocity, diagnostics,
    )
end

"""Single-phase flow with a smooth velocity (walls); cheap enough for run-sequence checks."""
function single_flow(;
        dt = nothing, adaptivity = A.Adaptivity(), diagnostics = (), viscosity = 1.0e-2, mesh = unit_mesh(),
        velocity = (x, y, t, axis) -> axis == 1 ? 0.2 * sinpi(y) : 0.0,
    )
    return A.FlowSimulation(
        mesh, A.SinglePhase(density = 1.0, dynamic_viscosity = viscosity);
        velocity, boundary = A.NoSlip(), dt, adaptivity, diagnostics,
    )
end

"""Names of the log entries (optionally filtered), as a Vector{Symbol}."""
names_of(log) = [e.name for e in log]

"""`log` restricted to the names in `keep`, dropping `_one_velocity_pressure_solver!` calls without `refresh = true`."""
function stage_log(log, keep)
    return [
        e for e in log if e.name in keep &&
            !(e.name === :_one_velocity_pressure_solver! && !get(e.args, :refresh, false))
    ]
end

"""One-line printable form of a log entry."""
show_entry(e) = isempty(e.args) ? string(e.name) : string(e.name, "(", join(("$k=$v" for (k, v) in pairs(e.args)), ", "), ")")

const FAILURES = Ref(0)
"""Print and count a check; the driver exits non-zero when any check fails."""
function check(label, ok::Bool, detail = "")
    ok || (FAILURES[] += 1)
    println(ok ? "PASS  " : "FAIL  ", label, isempty(detail) ? "" : "   [" * detail * "]")
    return ok
end

function finish(name)
    println("\n", name, ": ", FAILURES[] == 0 ? "all checks passed" : "$(FAILURES[]) FAILED")
    return exit(FAILURES[] == 0 ? 0 : 1)
end

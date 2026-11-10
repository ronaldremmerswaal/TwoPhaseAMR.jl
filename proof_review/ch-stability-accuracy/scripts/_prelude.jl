# Shared prelude of the code scripts of the audit of ch:stability-accuracy: loads TwoPhaseAMR and the fixtures/helpers of
# docs/techref/figures/stability.jl (the helpers of generate.jl that stability.jl uses are copied here, verbatim).
# Include with `isdefined(Main, :STAB_CFL) || include(joinpath(@__DIR__, "_prelude.jl"))`.

using Test, Random, Printf
using TwoPhaseAMR
const AMR = TwoPhaseAMR
const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const OUTPUT = joinpath(ROOT, "docs", "techref", "figures", "generated")

function fmt(x::Real)
    isinteger(x) && return string(Int(x))
    text = rstrip(rstrip(@sprintf("%.6f", x), '0'), '.')
    return text == "-0" ? "0" : text
end

function frac(x::Real)
    q = rationalize(Int, x; tol = 1.0e-12)
    @assert abs(Float64(q) - x) <= 1.0e-13 && denominator(q) <= 64
    denominator(q) == 1 && return string(numerator(q))
    sign = q < 0 ? "-" : ""
    return "$(sign)\\tfrac{$(abs(numerator(q)))}{$(denominator(q))}"
end

macro_line(name, value) = "\\newcommand\\$name{$value}\n"
sci(x) = iszero(x) ? "0" : begin
        exponent = floor(Int, log10(abs(x)))
        @sprintf("%.1f\\times10^{%d}", x / 10.0^exponent, exponent)
    end
fix3(x) = @sprintf("%.3f", x)

include(joinpath(ROOT, "docs", "techref", "figures", "stability.jl"))

# A uniform two-phase fixture (flat interface y = 1/2 on a 16 x 16 mesh, periodic in x, walls in y) with a uniform velocity (U, 0).
function uniform_two_phase(;
        sigma = 0.0, gravity = 0.0, U = 0.0, density = (1.0, 1.0e-3), cells = 16,
        velocity_model = OneVelocityModel(),
    )
    mesh = BlockTreeMesh(;
        base_mesh = (cells, cells), block_size = (8, 8), extent = (1.0, 1.0), periodic = (true, false),
    )
    interface = InterfaceField(mesh, (x, y) -> y - 0.5; lipschitz = 1)
    physics = TwoPhase(
        interface; density, dynamic_viscosity = (0.0, 0.0), surface_tension = sigma,
        gravity = (0.0, -gravity),
    )
    field = (x, y, time, axis) -> axis == 1 ? U : 0.0
    velocity = velocity_model isa TwoVelocityModel ? (field, field) : field
    flow = stab_track!(FlowSimulation(mesh, physics; velocity, velocity_model, boundary = FreeSlip()))
    return (; name = :uniform, flow, mesh, density, sigma, gravity)
end

two_velocity_model() = TwoVelocityModel(
    interface_transport = FrozenInterfaceFluxes(), pressure_projection = CMDGFMProjection(),
)

# Message of the ArgumentError of a rejected step, or "" if the step is accepted.
function step_message(flow, dt)
    try
        step!(flow.velocity, flow.pressure, flow.system, dt, flow.workspace; time = flow.time)
        return ""
    catch error
        error isa ArgumentError || rethrow()
        return error.msg
    end
end

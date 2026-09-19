using Test
using TwoPhaseAMR

const INTEGRATION_CASES = (
    "wave-equation" => "wave_equation.jl",
    "staggered-wave-equation" => "staggered_wave_equation.jl",
    "amr-manufactured-flow" => "amr_manufactured_flow.jl",
    "vortex-reversal" => "two_phase_vortex.jl",
    "two-velocity-transport" => "two_velocity_transport.jl",
    "two-velocity-rotation" => "two_velocity_rotation.jl",
    "two-velocity-vortex" => "two_velocity_vortex.jl",
    "two-velocity-instabilities" => "two_velocity_instabilities.jl",
    "two-phase-waves" => "two_phase_waves.jl",
    "floating-droplet" => "two_phase_droplet.jl",
    "sessile-droplet" => "two_phase_sessile_droplet.jl",
    "anisotropic-cells" => "anisotropic_cells.jl",
    "single-phase-cylinder" => "single_phase_cylinder.jl",
)

const OPTIONAL_INTEGRATION_CASES = (
    "mimeticfv-reference" => "mimeticfv_reference.jl",
)

const ALL_INTEGRATION_CASES = (
    INTEGRATION_CASES..., OPTIONAL_INTEGRATION_CASES...,
)

function _integration_case(name)
    index = findfirst(pair -> first(pair) == name, ALL_INTEGRATION_CASES)
    isnothing(index) && throw(
        ArgumentError(
            "unknown integration case '$name'; available cases are " *
                join(first.(ALL_INTEGRATION_CASES), ", "),
        ),
    )
    return last(ALL_INTEGRATION_CASES[index])
end

function _run_integration_file(name, filename)
    case_module = Module(gensym(:IntegrationCase))
    Core.eval(case_module, :(using Test, TwoPhaseAMR))
    return @testset "$name" begin
        Base.include(case_module, joinpath(@__DIR__, filename))
    end
end

selected_cases = isempty(ARGS) ? first.(INTEGRATION_CASES) : ARGS

@testset verbose = true showtiming = true "Integration" begin
    for name in selected_cases
        _run_integration_file(name, _integration_case(name))
    end
end

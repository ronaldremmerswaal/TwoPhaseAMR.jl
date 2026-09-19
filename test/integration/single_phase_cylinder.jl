using Test, TwoPhaseAMR

include("cylinder_reference_problem.jl")

function _parse_cylinder_resolutions(text)
    values = Tuple(parse.(Int, split(text, ',')))
    length(values) >= 3 || error(
        "CYLINDER_CELLS_PER_DIAMETER needs at least three entries",
    )
    return values
end

resolutions = _parse_cylinder_resolutions(
    get(
        ENV, "CYLINDER_CELLS_PER_DIAMETER", "4,8,16",
    )
)
final_time = parse(Float64, get(ENV, "CYLINDER_FINAL_TIME", "40"))
transient_time = parse(
    Float64,
    get(ENV, "CYLINDER_TRANSIENT_TIME", string(final_time / 2)),
)
study = cylinder_paper_validation(;
    cells_per_diameter = resolutions, final_time, transient_time,
    checkpoints = 5, live_dt = final_time + 1,
)
pressure_drag = getproperty.(study.studies, :mean_pressure_drag)
total_drag = getproperty.(study.studies, :mean_drag)
strouhal = getproperty.(study.studies, :strouhal)
maximum_lift = getproperty.(study.studies, :maximum_lift)
println(
    (;
        resolutions, pressure_drag, total_drag, strouhal, maximum_lift,
        pressure_drag_order = study.pressure_drag_order,
        reference = study.reference,
    )
)

@testset "Re=100 cylinder paper refinement study" begin
    @test all(case.result.pressure.converged for case in study.studies)
    @test all(isfinite, pressure_drag)
    @test all(isfinite, total_drag)
    @test all(isfinite, strouhal)
    @test all(isfinite, maximum_lift)
    # Figure 4 of the paper reports approximately second-order pressure
    # drag.  As in the other integration studies, allow the asymptotic
    # order a quarter-order pre-asymptotic margin.
    @test study.pressure_drag_order >= 1.75
    reference = study.reference
    comparisons = (
        (pressure_drag, reference.pressure_drag),
        (total_drag, reference.total_drag),
        (strouhal, reference.strouhal),
        (maximum_lift, reference.maximum_lift),
    )
    for (values, target) in comparisons
        @test abs(values[end] - target) < abs(values[1] - target)
    end
    @test all(study.studies) do case
        case.mean_drag ≈
            case.mean_pressure_drag + case.mean_viscous_drag
    end
end

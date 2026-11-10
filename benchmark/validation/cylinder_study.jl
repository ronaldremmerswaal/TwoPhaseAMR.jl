# Manual, intentionally long validation (not part of the CI test registry):
#   julia --project=. benchmark/validation/cylinder_study.jl
# A symmetric start can suppress shedding down to roundoff, making zero
# crossings and Strouhal meaningless. Use a symmetry-breaking transverse
# perturbation and run long enough for a wake and at least two post-transient
# positive-going lift crossings to develop. These defaults may need a longer
# final time or a different perturbation, especially at coarse resolutions.
using Test, TwoPhaseAMR

include(joinpath(@__DIR__, "../../test/integration/cylinder_reference_problem.jl"))

resolutions = Tuple(parse.(Int, split(get(ENV, "CYLINDER_CELLS_PER_DIAMETER", "4,8,16"), ',')))
final_time = parse(Float64, get(ENV, "CYLINDER_FINAL_TIME", "100"))
transient_time = parse(Float64, get(ENV, "CYLINDER_TRANSIENT_TIME", string(final_time / 2)))
perturbation = parse(Float64, get(ENV, "CYLINDER_PERTURBATION", "0.001"))

@testset "Re=100 cylinder paper validation" begin
    study = cylinder_paper_validation(;
        cells_per_diameter = resolutions, final_time, transient_time, perturbation,
        keyframe_count = 5, preview_dt = final_time + 1,
    )
    pressure_drag = getproperty.(study.studies, :mean_pressure_drag)
    total_drag = getproperty.(study.studies, :mean_drag)
    # Exclude crossings that do not connect measurable negative and positive
    # excursions; otherwise roundoff before shedding corrupts the period.
    strouhal = map(study.studies) do case
        cylinder_strouhal(
            case.times, case.lift; start_time = transient_time, minimum_lift = 1.0e-3,
        )
    end
    maximum_lift = getproperty.(study.studies, :maximum_lift)
    println(
        (;
            resolutions, pressure_drag, total_drag, strouhal, maximum_lift,
            pressure_drag_order = study.pressure_drag_order, reference = study.reference,
        )
    )
    @test all(case.result.pressure.converged for case in study.studies)
    @test all(isfinite, pressure_drag)
    @test all(isfinite, total_drag)
    @test all(isfinite, maximum_lift)
    # A finite thresholded estimate requires at least two resolved cycles.
    @test all(isfinite, strouhal)
    # NUMERICS CONCERN: the former unperturbed t=40 CI study measured -0.884,
    # not the approximately second-order pressure drag reported in the paper.
    @test study.pressure_drag_order >= 1.75
    reference = study.reference
    for (values, target) in (
            (pressure_drag, reference.pressure_drag),
            (total_drag, reference.total_drag),
            (strouhal, reference.strouhal),
            (maximum_lift, reference.maximum_lift),
        )
        @test abs(values[end] - target) < abs(values[1] - target)
    end
    @test all(study.studies) do case
        case.mean_drag ≈ case.mean_pressure_drag + case.mean_viscous_drag
    end
end

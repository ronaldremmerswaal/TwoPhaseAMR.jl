# Code check 01: staggered density (eq:forces:density) and pressure coefficients (eq:forces:coefficient).
#
# Label:  eq:forces:density, eq:forces:coefficient
# Source: docs/techref/chapters/08_forces.tex:82-136; src/two_phase/forces.jl (_staggered_liquid_fraction, _mixture_density!, PoissonOperator),
#         src/two_phase/projection/common.jl (_projection_pressure_operator), src/navier_stokes/system.jl (single phase)
# Checks (the five balance fixtures, the drop fixtures, the planar fixtures with solids):
#   D1  rho_f of the code equals the independent value (strip weights of the two cells with clipped open areas, conditional fractions, clamp);
#   D2  the clamp is inactive: the unclamped weighted fraction lies in [0, 1] to rounding (convex combination);
#   D3  coefficients, bitwise: one-velocity operator 1/rho (no solid) or a^2/rho (solid); two-velocity operator 1/rho or a/rho; the
#       single-phase operator 1/rho_0 (times a^2 with a solid); gamma = gamma^m a^2 with gamma^m = 1/rho;
#   D4  faces with both open strips empty (covered faces): fraction 0 -> rho = rho_g and coefficient 0 (a = 0);
#   D5  refreshing the coefficient keeps the metric: the template's gradient operator (hence K) is the same object, and K is unchanged (bitwise).
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-stages/scripts/code_01_density_coefficient.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ForcesAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "forces_audit.jl"))
using .ForcesAudit
using TwoPhaseAMR
const FA = ForcesAudit

fixtures = Any[FA.balance_fixtures()...]
push!(fixtures, :plane_amr_liquid => FA.plane_field(:amr; solid = :liquid))
push!(fixtures, :plane_uniform_gas => FA.plane_field(:uniform; solid = :gas))
d1 = d2lo = d2hi = 0.0; d3 = 0.0; d3tv = 0.0; ncovered = 0; covered_ok = true; d5 = true
d2lo = Inf; d2hi = -Inf
for (name, field) in fixtures
    mesh = field.mesh
    solid = !isnothing(field.geometry)
    a = FA.apertures(field)
    rho = AMR.mixture_density(field, FA.DENSITY)
    rho_ind = [FA.staggered_density_independent(field, f) for f in mesh.faces]
    global d1 = max(d1, maximum(abs.(rho .- rho_ind) ./ rho_ind))
    # D2: raw fraction
    for f in mesh.faces
        weighted = volume = 0.0
        for ref in (f.minus, f.plus)
            w = 0.5 * FA.open_strip_area(field, ref, f) * mesh.blocks[ref.block].volume
            weighted += w * AMR._conditional_fraction(field, ref); volume += w
        end
        if volume > 0
            c = weighted / volume
            global d2lo = min(d2lo, c); global d2hi = max(d2hi, c)
        else
            global ncovered += 1
            i = findfirst(x -> x === f, mesh.faces)
            global covered_ok &= a[i] == 0
        end
    end
    # D3
    op1 = FA.one_velocity_operator(field); optv = FA.two_velocity_operator(field)
    g1 = solid ? a .^ 2 ./ rho : 1 ./ rho
    gtv = solid ? a ./ rho : 1 ./ rho
    global d3 = max(d3, maximum(abs.(op1.face_coefficient .- g1)))
    global d3tv = max(d3tv, maximum(abs.(optv.face_coefficient .- gtv)))
    # D5
    template = FA.one_velocity_operator(field)                 # (the refresh overwrites the template's coefficient in place)
    K0 = copy(FA.metric_matrix(template).nzval)
    refreshed = isnothing(field.geometry) ? PoissonOperator(field; density = (500.0, 2.0), template) :
        PoissonOperator(field, field.geometry; density = (500.0, 2.0), template)
    global d5 &= refreshed.gradient_operator === template.gradient_operator && FA.metric_matrix(refreshed).nzval == K0 &&
        refreshed.face_coefficient != op1.face_coefficient
    println(rpad("fixture $name", 20), " rho gap ", round(maximum(abs.(rho .- rho_ind) ./ rho_ind), sigdigits = 2), "  coefficient gap (one/two-velocity) ", maximum(abs.(op1.face_coefficient .- g1)), "/", maximum(abs.(optv.face_coefficient .- gtv)))
end
# single phase
mesh = FA.mesh_of(:amr)
single = PoissonOperator(mesh; face_coefficient = inv(1.7))
single_gap = maximum(abs.(single.face_coefficient .- inv(1.7)))
geometry = GeometryField(mesh, FA.solid_low; lipschitz = hypot(1, 0.1))
single_cut = PoissonOperator(geometry; face_coefficient = inv(1.7))
single_cut_gap = maximum(abs.(single_cut.face_coefficient .- geometry.aperture .^ 2 ./ 1.7))
println("D1 ", d1, "; D2 raw fraction in [", d2lo, ", ", d2hi, "]; D3 ", d3, " / ", d3tv, "; D4 covered faces ", ncovered, " (aperture zero: ", covered_ok, "); D5 metric object kept: ", d5,
    "; single phase gaps ", single_gap, ", ", single_cut_gap)
@testset "staggered density and coefficients" begin
    @test d1 <= 1.0e-14
    @test d2lo >= -1.0e-14 && d2hi <= 1 + 1.0e-14
    @test d3 == 0 && d3tv == 0
    @test ncovered > 0 && covered_ok
    @test d5
    @test single_gap == 0 && single_cut_gap == 0
end

# Code check 01: eq:viscosity:rows (normal, shear and wall rows of E) and eq:viscosity:weights (stress weights), against the brute-force
# reimplementation of _shared/viscosity_audit.jl (mesh, conditional fractions and open fractions only).
#
# Label:  eq:viscosity:rows, eq:viscosity:weights (shared fixtures; the weights script of the weights folder repeats the weight part)
# Source: docs/techref/chapters/09_viscosity.tex:222-296; src/two_phase/diffusion/variable_diffusion_operator.jl (_variable_staggered_diffusion),
#         src/embedded/operators.jl (_embedded_variable_wall_stresses)
# Checks:
#   R1  normal + shear rows of the code's E equal the independent E on the three docs fixtures and 12 random composite meshes
#       (levels 1 and 2, walled / periodic / mixed, with and without solid, NoSlip and FreeSlip), entrywise to 1e-12 max|E|;
#   R2  stress weights 2 mu_c |c^o|, mu_v |omega_v| with log mu_v the open-volume mean, equal the code's (relative 1e-13);
#   R3  wall rows (solid): one entry 1 at the face, weights cond*mu_c(w) equal the independent list (cut: segment length, midpoint,
#       max(|f-p|, h/4); covered: 1), counts, FreeSlip omits the cut kind only;
#   R4  the sum of the vertex measures = the open volume of the domain; the corner vertices are the cell corners on the finest lattice.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-viscosity-rows/scripts/code_01_rows.jl
using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ViscosityAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "viscosity_audit.jl"))
using .ViscosityAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR
const MA = ViscosityAudit.MA

function compare_field(field, boundary; label)
    mesh = field.mesh
    op = StaggeredDiffusionOperator(field; viscosity = FIX_VISC, boundary)
    E, meas, logmu, used = independent_strain(field, FIX_VISC; boundary)
    n = size(E, 1)
    @test size(op.strain, 2) == size(E, 2)
    Ec = op.strain[1:n, :]
    dE = maximum(abs, Ec - E) / maximum(abs, E)
    w = meas .* exp.(logmu)
    dw = maximum(abs(w[i] - op.stress_weights[i]) / w[i] for i in 1:n if w[i] > 0)
    zero_ok = all(op.stress_weights[i] == 0 for i in 1:n if w[i] == 0)
    nwall = size(op.strain, 1) - n
    # R3 wall rows
    wall_ok = true; wall_diff = 0.0
    if field.geometry === nothing
        wall_ok = nwall == 0
    else
        indep = independent_wall_rows(field; include_cut = boundary isa NoSlip)
        wall_ok = nwall == length(indep)
        if wall_ok
            lm(cn) = begin
                ref = cells(mesh)[cn]
                c = cond_fraction(field, ref); c * log(FIX_VISC[1]) + (1 - c) * log(FIX_VISC[2])
            end
            code_list = sort!([(findfirst(!iszero, op.strain[n + j, :]), op.stress_weights[n + j]) for j in 1:nwall])
            ind_list = sort!([(i, c * exp(lm(cn))) for (i, c, cn, kind) in indep])
            wall_ok &= all(a[1] == b[1] for (a, b) in zip(code_list, ind_list))
            wall_diff = maximum(abs(a[2] - b[2]) / b[2] for (a, b) in zip(code_list, ind_list))
            wall_ok &= all(count(!iszero, op.strain[n + j, :]) == 1 && maximum(op.strain[n + j, :]) == 1 for j in 1:nwall)
        end
    end
    # R4: vertex measures sum to the open volume
    nv = n - 2 * length(cells(mesh))
    total_open = sum(open_fraction(field, ref) * cell_volume(mesh, ref) for ref in cells(mesh))
    vsum = sum(meas[(2 * length(cells(mesh)) + 1):n])
    println(rpad(label, 40), "dE=", round(dE, sigdigits = 2), "  dw=", round(dw, sigdigits = 2), "  rows: ", nv, " vertices, ", nwall, " wall; wall weight diff=", round(wall_diff, sigdigits = 2), "  vertex-measure sum / open volume - 1 = ", round(vsum / total_open - 1, sigdigits = 2))
    @test dE <= 1.0e-12
    @test dw <= 1.0e-13
    @test zero_ok
    @test wall_ok
    @test wall_diff <= 1.0e-12
    @test abs(vsum / total_open - 1) <= 1.0e-13
    return nwall
end

@testset "R1-R4 rows, weights, wall rows" begin
    for (name, fx) in ((:walled, fixture_walled), (:cut, fixture_cut), (:periodic, fixture_periodic))
        for boundary in (NoSlip(), FreeSlip())
            compare_field(fx(), boundary; label = "$name $(typeof(boundary).name.name)")
        end
    end
    rng = MersenneTwister(909)
    count = 0
    for periodic in ((true, true), (false, false), (true, false)), levels in (1, 2), solid in (nothing, :disc)
        count += 1
        field, _, _ = MA.random_two_phase(rng; periodic, levels, solid_kind = solid)
        compare_field(field, count % 2 == 0 ? NoSlip() : FreeSlip(); label = "random #$count periodic=$periodic L=$levels solid=$solid")
    end
end

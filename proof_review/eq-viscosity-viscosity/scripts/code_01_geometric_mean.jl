# Code check 01: eq:viscosity:viscosity -- geometric-mean viscosity with the conditional fraction; vertex value = geometric mean at
# the open-volume average of the conditional fractions (eq:viscosity:weights); refresh equals rebuild.
#
# Label:  eq:viscosity:viscosity (also the vertex statement of :weights)
# Source: docs/techref/chapters/09_viscosity.tex:208-216, 280-294; src/two_phase/diffusion/variable_diffusion_geometry.jl
#         (dynamic_viscosity, _phase_dynamic_viscosity), variable_diffusion_operator.jl (_update_dynamic_viscosity!),
#         src/interface/interface_field.jl (_refresh_conditional_fraction!: conditional = liquid volume / open volume)
# Checks:
#   V1  dynamic_viscosity(field; viscosity) = mu_l^c mu_g^(1-c) = exp(c log mu_l + (1-c) log mu_g), c = conditional fraction, every cell
#       (walled, cut, periodic fixtures and random composite fields incl. extreme ratio 1e5), mu_g <= mu <= mu_l,
#   V2  conditional fraction = liquid volume / open volume on cut cells,
#   V3  vertex weight / vertex measure = mu_g^(1-cbar) mu_l^cbar with cbar the open-volume-weighted mean of the conditional fractions,
#   V4  _update_dynamic_viscosity! (log-form refresh) reproduces the weights of a rebuild to round-off.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-viscosity-viscosity/scripts/code_01_geometric_mean.jl
using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ViscosityAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "viscosity_audit.jl"))
using .ViscosityAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR
const MA = ViscosityAudit.MA

@testset "V1-V4 geometric mean viscosity" begin
    rng = MersenneTwister(606)
    fields = Any[fixture_walled(), fixture_cut(), fixture_periodic()]
    for solid in (nothing, :disc), periodic in ((true, true), (false, false))
        push!(fields, MA.random_two_phase(rng; periodic, solid_kind = solid, levels = 1)[1])
    end
    for (n, field) in enumerate(fields), mus in (FIX_VISC, (1.0e3, 1.0e-2))
        mesh = field.mesh
        mu = dynamic_viscosity(field; viscosity = mus)
        worst = 0.0; lo = Inf; hi = 0.0
        for ref in cells(mesh)
            c = cond_fraction(field, ref)
            m = AMR._get(mu, ref)
            worst = max(worst, abs(m - mus[1]^c * mus[2]^(1 - c)) / m, abs(m - exp(c * log(mus[1]) + (1 - c) * log(mus[2]))) / m)
            lo = min(lo, m); hi = max(hi, m)
            # V2: conditional fraction = liquid volume / open volume
            if field.geometry !== nothing
                avail = open_fraction(field, ref)
                if avail > 0 && c > 0
                    liquid = AMR._get(field.fraction, ref)
                    @test c ≈ clamp(liquid, 0, avail) / avail atol = 1.0e-12
                end
            end
        end
        @test worst <= 1.0e-14
        @test lo >= min(mus...) * (1 - 1.0e-14) && hi <= max(mus...) * (1 + 1.0e-14)
        # V3/V4 through the operator
        op = StaggeredDiffusionOperator(field; viscosity = mus)
        E, meas, logmu, used = independent_strain(field, mus)
        ncell2 = 2 * length(cells(mesh))
        # independent vertex values: cbar
        lat = lattice_of(mesh)
        verts = Dict{NTuple{2, Int}, Vector{Tuple{CellRef, Float64}}}()
        for ref in cells(mesh)
            x = AMR.cellcenter(mesh, ref.block, ref.i, ref.j); h = cell_spacing(mesh, ref)
            for sx in (-1, 1), sy in (-1, 1)
                key = ntuple(2) do k
                    val = round(Int, (x[k] + (k == 1 ? sx : sy) * h[k] / 2 - mesh.origin[k]) / mesh.extent[k] * lat[k])
                    mesh.periodic[k] ? mod(val, lat[k]) : val
                end
                push!(get!(verts, key, Tuple{CellRef, Float64}[]), (ref, cell_volume(mesh, ref) / 4))
            end
        end
        wv = 0.0
        for (j, key) in enumerate(used)
            cl = verts[key]
            vol = sum(w * open_fraction(field, ref) for (ref, w) in cl)
            cbar = sum(w * open_fraction(field, ref) * cond_fraction(field, ref) for (ref, w) in cl) / vol
            muv = mus[1]^cbar * mus[2]^(1 - cbar)
            code = op.stress_weights[ncell2 + j] / op.stress_measures[ncell2 + j]
            wv = max(wv, abs(code - muv) / muv)
        end
        @test wv <= 1.0e-13
        # refresh vs rebuild with different viscosities
        other = (mus[1] * 1.7, mus[2] * 0.6)
        AMR._update_dynamic_viscosity!(op, field, other)
        op2 = StaggeredDiffusionOperator(field; viscosity = other)
        dw = maximum(abs, op.stress_weights .- op2.stress_weights) / maximum(op2.stress_weights)
        dS = maximum(abs, op.stiffness - op2.stiffness) / maximum(abs, op2.stiffness)
        println("field #", n, " mus=", mus, ": mean form defect ", worst, ", vertex defect ", wv, ", refresh-vs-rebuild weights ", dw, ", stiffness ", dS)
        @test dw <= 1.0e-13 && dS <= 1.0e-12
    end
end

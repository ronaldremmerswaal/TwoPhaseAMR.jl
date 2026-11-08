# Code check 02: the identity at walls (boundary dual faces), the two documented exceptions, and the operator connection.
#
# Label:  eq:momentum:identity  (eq:momentum:boundary, "Limitations at walls", operator connection of the divergence identity)
# Source: docs/techref/chapters/06_momentum_transport.tex:189-215 (identity, operator connection), 309-361 (boundary dual faces,
#         limitations (i), (ii));  src/operators/advection/staggered_schedules.jl (_boundary_dual_faces),
#         src/operators/advection/staggered_fluxes.jl (_add_boundary_convection!)
# Checks (centred operator assembled with convection!, q = 1 including its boundary values, NoSlip walls, at dt = 0):
#   A1  uncut walled meshes, non-solenoidal random velocity: C(v)1 equals the printed right-hand side (cell sums over the
#       interior faces; wall faces carry no flux) at every unknown except the T-incomplete ones (limitation (i), issue 60), and
#       every T-incomplete unknown fails;  without the boundary dual faces every unknown that violates the hypothesis H0 fails.
#   A2  walled meshes with a solid (sloped wall across the box, discs): the failing unknowns are exactly the T-incomplete ones
#       and the unknowns of wall cells whose face aperture is below one (limitation (ii), issue 61): no further failure.
#   A3  solenoidal velocities (streamfunction vanishing on the walls and on solid vertices): C(v)1 = 0 outside (i), (ii).
#   B1  periodic composite meshes with an interior solid, random non-solenoidal velocity: the operator connection
#       C(v)1 = H(div^o v) holds exactly at the unknowns that satisfy condition (b) of eq:momentum:mass-identity (every cell of f
#       uncut or f the single face of its side), and elsewhere the residual is exactly
#           V_f [C1 - H div^o]_f = (1/2) sum_c ( varpi_{c,f} |c^o| - |Lambda_{c,f} cap c^o| ) (div^o v)_c .
# The hard checks cover the claim under its hypotheses.  The two exceptions (i) (issue 60) and (ii) (issue 61) and the failure of
# condition (b) (issue 63) are being fixed: what the current code does there is REPORTED, never asserted, so that the run keeps
# passing after the fixes and shows what changed.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-momentum-identity/scripts/code_02_walls_and_operator_connection.jl

using Test, Random, LinearAlgebra
isdefined(Main, :MomentumAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "momentum_audit.jl"))
using .MomentumAudit
using TwoPhaseAMR
const MA = MomentumAudit
const D = MA.Docs
const AMR = TwoPhaseAMR

"""Random fixture; the solenoidal velocity needs a solid whose open fluxes telescope, so the solid is redrawn until it does."""
function fixture(rng; periodic, solid_kind, solenoidal)
    for attempt in 1:50
        mesh = MA.random_mesh(rng; periodic, levels = 1, refine_probability = 0.5)
        solid = isnothing(solid_kind) ? nothing : MA.random_solid(rng, mesh.extent; kind = solid_kind)
        geometry = isnothing(solid) ? nothing : GeometryField(mesh, solid; lipschitz = 2)
        velocity = try
            if solenoidal
                psi = MA.random_streamfunction(rng, mesh.extent; walls = !all(periodic))
                isnothing(geometry) ? MA.stream_velocity(mesh, psi) :
                    D.solid_streamfunction_velocity(mesh, geometry, solid, psi)
            else
                MA.random_velocity(rng, mesh)
            end
        catch err
            err isa AssertionError || rethrow()
            continue
        end
        isnothing(geometry) || enforce_geometry_no_slip!(velocity, geometry)
        return mesh, geometry, velocity
    end
    error("no admissible fixture found")
end

"""One fixture: failing unknowns of C(v)1 against the printed right-hand side, with and without the boundary dual faces."""
function wall_case(rng; periodic, solid_kind, solenoidal)
    mesh, geometry, velocity = fixture(rng; periodic, solid_kind, solenoidal)
    nf = length(mesh.faces)
    aper = isnothing(geometry) ? ones(nf) : Float64.(geometry.aperture)
    Phi = [mesh.faces[i].area * aper[i] * velocity.interior[i] for i in 1:nf]
    cell_faces = MA.faces_of_cells(mesh)
    scale = maximum(abs, Phi)
    stats = (;
        total = Ref(0), failing = Ref(0), tinc = Ref(0), tinc_failing = Ref(0), cutwall = Ref(0), unexplained = Ref(0),
        nonh0 = Ref(0), nonh0_failing_noboundary = Ref(0), residual_explained = Ref(0.0), solenoidal_max = Ref(0.0),
    )
    for k in (1, 2)
        inc = MA.component_incidence(mesh, k)
        one = FaceField{k}(mesh; init = 1)
        ws = D.staggered_workspace(one, geometry)
        V = ws.schedule.volumes
        P = MA.identity_rhs_matrix(mesh, inc, nf, 1) .+ MA.identity_rhs_matrix(mesh, inc, nf, 2)
        rhs = (P * Phi) ./ V
        closed = FaceField{k}(mesh)
        convection!(closed, one, velocity, 0.0, ws; scheme = Centred(), amr_interface_scheme = Centred(), boundary = NoSlip())
        open_ = FaceField{k}(mesh)
        convection!(open_, one, velocity, 0.0, ws; scheme = Centred(), amr_interface_scheme = Centred())
        for (l, g) in enumerate(inc.interior)
            f = mesh.faces[g]
            stats.total[] += 1
            tolerance = 1.0e-12 * scale / V[l]
            r = abs(closed.interior[l] - rhs[l])
            tinc = !MA.t_complete(mesh, inc, g; cell_faces)
            wallcell = !MA.hypothesis_h0(mesh, inc, g)
            cutwall = wallcell && aper[g] < 1
            tinc && (stats.tinc[] += 1)
            if r > tolerance
                stats.failing[] += 1
                tinc && (stats.tinc_failing[] += 1)
                cutwall && (stats.cutwall[] += 1)
                tinc || cutwall || (stats.unexplained[] += 1)
            end
            # the unknowns outside (i), (ii) satisfy the closed identity (this also covers the wall cells)
            if !(tinc || cutwall)
                stats.residual_explained[] = max(stats.residual_explained[], r * V[l] / scale)
                solenoidal && (stats.solenoidal_max[] = max(stats.solenoidal_max[], abs(closed.interior[l]) * V[l] / scale))
            end
            if wallcell
                stats.nonh0[] += 1
                abs(open_.interior[l] - rhs[l]) > tolerance && (stats.nonh0_failing_noboundary[] += 1)
            end
        end
    end
    return stats
end

function summarize(stats_list)
    keys_ = (:total, :failing, :tinc, :tinc_failing, :cutwall, :unexplained, :nonh0, :nonh0_failing_noboundary)
    return Dict(key => sum(getfield(s, key)[] for s in stats_list) for key in keys_)
end

@testset "eq:momentum:identity: walls, boundary dual faces, exceptions (i) and (ii)" begin
    rng = MersenneTwister(2026)
    @testset "A1 uncut walled meshes" begin
        runs = [wall_case(rng; periodic = (false, false), solid_kind = nothing, solenoidal = false) for _ in 1:12]
        append!(runs, [wall_case(rng; periodic = (true, false), solid_kind = nothing, solenoidal = false) for _ in 1:4])
        s = summarize(runs)
        MA.excluded(
            "A1 limitation (i) (issue 60)", (; tincomplete = s[:tinc], of_which_failing = s[:tinc_failing], failing_total = s[:failing]),
            note = "pre-fix: every T-incomplete unknown fails and nothing else does"
        )
        @test s[:unexplained] == 0                                     # every failing unknown is T-incomplete or a cut wall unknown
        @test s[:nonh0_failing_noboundary] == s[:nonh0] > 100          # without the boundary dual faces every wall-adjacent unknown fails
        @test maximum(r.residual_explained[] for r in runs) <= 1.0e-13 # the others (incl. all wall cells) satisfy the closed identity
    end
    @testset "A2/A3 walled meshes with a solid" begin
        runs = Any[]
        for kind in (:wall, :disc), solenoidal in (false, true), _ in 1:5
            push!(runs, wall_case(rng; periodic = (false, false), solid_kind = kind, solenoidal))
        end
        s = summarize(runs)
        MA.excluded(
            "A2/A3 limitations (i), (ii) (issues 60, 61)", (; tincomplete = s[:tinc], cutwall_failing = s[:cutwall], failing_total = s[:failing]),
            note = "pre-fix: cut wall cells next to a wall fail"
        )
        @test s[:unexplained] == 0
        @test maximum(r.residual_explained[] for r in runs) <= 1.0e-13
        @test maximum(r.solenoidal_max[] for r in runs) <= 1.0e-13
    end
end

@testset "eq:momentum:identity: operator connection on cut meshes (B1)" begin
    rng = MersenneTwister(7)
    worst_b = 0.0
    worst_formula = 0.0
    n_b = 0
    n_not_b = 0
    n_not_b_nonzero = 0
    for trial in 1:8
        mesh, geometry, velocity = fixture(rng; periodic = (true, true), solid_kind = :disc, solenoidal = false)
        div = CellField(mesh)
        divergence!(div, velocity, geometry)
        nf = length(mesh.faces)
        flux_scale = maximum(abs(mesh.faces[i].area * geometry.aperture[i] * velocity.interior[i]) for i in 1:nf)
        for k in (1, 2)
            inc = MA.component_incidence(mesh, k)
            one = FaceField{k}(mesh; init = 1)
            dest = FaceField{k}(mesh)
            ws = D.staggered_workspace(one, geometry)
            convection!(dest, one, velocity, 0.0, ws; scheme = Centred(), amr_interface_scheme = Centred())
            V = ws.schedule.volumes
            interp = D.interpolation_weights(mesh, geometry, D.component_dofs(mesh, k))
            for (l, g) in enumerate(inc.interior)
                f = mesh.faces[g]
                # V_f (H div)_f = sum_c (1/2)|Lambda_{c,f} cap c^o| div_c  (the weights are (1/2)|Lambda cap c^o| themselves)
                VHdiv = sum(w * AMR._get(div, c) for (c, w) in interp[l])
                residual = V[l] * dest.interior[l] - VHdiv
                predicted = 0.0
                condition_b = true
                term_scale = flux_scale   # rounding scale: |c| |div^o| in small open cells (clipping error of the open area times 1/|c^o|)
                for c in (f.minus, f.plus)
                    varpi = f.area / mesh.blocks[c.block].spacing[3 - k]
                    open_volume = AMR._get(geometry.open_fraction, c) * mesh.blocks[c.block].volume
                    lambda = D.strip_open_area(geometry, c, f)[1] * mesh.blocks[c.block].volume
                    predicted += (varpi * open_volume - lambda) / 2 * AMR._get(div, c)
                    term_scale = max(term_scale, mesh.blocks[c.block].volume * abs(AMR._get(div, c)))
                    uncut = AMR._get(geometry.open_fraction, c) >= 1
                    condition_b &= uncut || isapprox(varpi, 1; atol = 1.0e-12)
                end
                residual /= term_scale
                predicted /= term_scale
                worst_formula = max(worst_formula, abs(residual - predicted))
                if condition_b
                    n_b += 1
                    worst_b = max(worst_b, abs(residual))
                else
                    n_not_b += 1
                    abs(residual) > 1.0e-10 && (n_not_b_nonzero += 1)      # relative
                end
            end
        end
    end
    println("B1 (residuals relative to the rounding scale): unknowns with (b): ", n_b, " (max |residual| ", worst_b, ")")
    MA.excluded(
        "B1 condition (b) fails (issue 63)", (; unknowns = n_not_b, nonzero_residual = n_not_b_nonzero, defect_formula_gap = worst_formula),
        note = "pre-fix: residual = (1/2) sum_c (varpi |c^o| - |Lambda cap c^o|) div^o_c"
    )
    @test worst_b <= 1.0e-13
    @test n_not_b > 0
end

# Code check 02: the skew identity with walls ("and with walls where the boundary values vanish").
#
# Label:  eq:momentum:skew  (eq:momentum:boundary, eq:momentum:dual-divergence)
# Source: docs/techref/chapters/06_momentum_transport.tex:486-512 (the parenthesis on walls), 309-346 (boundary dual faces);
#         src/operators/advection/staggered_fluxes.jl (_add_boundary_convection!), src/operators/advection/staggered_schedules.jl
# Setting: walled composite meshes (uncut, and with a sloped solid wall across the box), NoSlip walls, Centred bulk and interface
# scheme, the operator assembled from convection!(...; boundary = NoSlip()) at dt = 0 with all boundary values q^partial_b = 0.
#   W1  the off-diagonal part of V C is skew (the boundary dual faces only touch the diagonal when q^partial = 0);
#   W2  the diagonal: 2 V_f C_ff = V_f (div Upsilon)_f + beta_f with beta_f = sum_{b in B(f)} sigma_b Upsilon_b (boundary dual faces
#       of eq:momentum:boundary, uncut measures).  beta_f vanishes away from the walls and is NOT small in a wall cell, so the printed
#       right-hand side sum V q psi (div Upsilon) of eq:momentum:skew (interior dual faces only) is wrong at wall cells: the
#       displayed identity and "2 dC_ff/dq_f = div Upsilon" hold with the CLOSED dual divergence div Upsilon + V^{-1} beta;
#   W3  non-zero boundary values: A(q) = C q + (affine part), and <psi, A q> + <q, A psi> = sum V q psi (closed div) +
#       sum_b sigma_b Upsilon_b (q^partial_b psi_f + psi^partial_b q_f)/2;
#   W4  energy: for a solenoidal velocity (streamfunction vanishing on the walls and the solid) the kinetic energy rate -q^T V C q
#       is (1/2) sum V q^2 (closed div): zero at every unknown outside the two configurations of the wall limitations, which are
#       reported (not asserted: issues 60, 61 are being fixed).
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-momentum-skew/scripts/code_02_skew_walls.jl

using Test, Random, LinearAlgebra
isdefined(Main, :MomentumAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "momentum_audit.jl"))
using .MomentumAudit
using TwoPhaseAMR
const MA = MomentumAudit
const D = MA.Docs
const AMR = TwoPhaseAMR

function wall_fixture(rng; solid_kind, solenoidal)
    for attempt in 1:50
        mesh = MA.random_mesh(rng; periodic = (false, false), levels = 1, refine_probability = 0.5)
        solid = solid_kind === nothing ? nothing : MA.random_solid(rng, mesh.extent; kind = solid_kind)
        geometry = solid === nothing ? nothing : GeometryField(mesh, solid; lipschitz = 2)
        velocity = try
            if solenoidal
                psi = MA.random_streamfunction(rng, mesh.extent; walls = true)
                geometry === nothing ? MA.stream_velocity(mesh, psi) : D.solid_streamfunction_velocity(mesh, geometry, solid, psi)
            else
                MA.random_velocity(rng, mesh)
            end
        catch err
            err isa AssertionError || rethrow()
            continue
        end
        geometry === nothing || enforce_geometry_no_slip!(velocity, geometry)
        return mesh, geometry, velocity
    end
    error("no admissible fixture")
end

"""Normal-component boundary dual faces: `(unknown, boundary face, sigma_b Upsilon_b)` of eq:momentum:boundary (uncut measure)."""
function normal_boundary_dual_faces(mesh, inc, velocity)
    list = Tuple{Int, Int, Float64}[]
    for (b, bf) in pairs(mesh.boundary_faces)
        bf.axis == inc.k || continue
        entry = get(inc.sides, bf.cell, nothing)
        entry === nothing && continue
        for (l, varpi) in entry[-bf.side]
            f = mesh.faces[inc.interior[l]]
            push!(list, (l, b, bf.side * f.area / 2 * (velocity.boundary[b] + velocity.interior[inc.interior[l]])))
        end
    end
    return list
end

@testset "eq:momentum:skew: walls" begin
    rng = MersenneTwister(99)
    stats = Dict(
        :unknowns => 0, :printed_fails => 0, :printed_fails_max => 0.0, :beta_gap_clean => 0.0, :skew_offdiag => 0.0,
        :affine_gap => 0.0, :energy_clean => 0.0, :cases => 0,
    )
    energy_excluded = 0.0
    for (solid_kind, solenoidal) in ((nothing, false), (nothing, true), (:wall, false), (:wall, true), (:disc, true)), trial in 1:3
        mesh, geometry, velocity = wall_fixture(rng; solid_kind, solenoidal)
        nf = length(mesh.faces)
        aper = geometry === nothing ? ones(nf) : Float64.(geometry.aperture)
        Phi = [mesh.faces[i].area * aper[i] * velocity.interior[i] for i in 1:nf]
        cell_faces = MA.faces_of_cells(mesh)
        for k in (1, 2)
            stats[:cases] += 1
            inc = MA.component_incidence(mesh, k)
            n = inc.n
            cls = [MA.unknown_class(mesh, inc, g, aper; cell_faces) for g in inc.interior]
            clean = [c in (:regular, :wall) for c in cls]
            C, offset, ws = MA.assemble_convection(mesh, geometry, velocity, k; boundary = NoSlip(), boundary_value = 0.0)
            @test maximum(abs, offset) == 0                                        # zero boundary values: a linear operator
            V = MA.schedule_volumes(ws, n)
            VdivU = MA.dual_divergence(mesh, inc, Phi)
            Bd = MA.boundary_dual_divergence(mesh, inc, velocity)
            VC = V .* C
            scale = maximum(abs, VC)
            # W1 off-diagonal skew
            off = VC .+ VC'
            for i in 1:n
                off[i, i] = 0.0
            end
            stats[:skew_offdiag] = max(stats[:skew_offdiag], maximum(abs, off) / scale)
            @test maximum(abs, off) / scale <= 1.0e-13
            # W2 diagonal: measured boundary part beta_f = 2 V C_ff - V div Upsilon
            beta = 2 .* V .* diag(C) .- VdivU
            for l in 1:n
                stats[:unknowns] += 1
                if clean[l]
                    gap = abs(beta[l] - Bd[l]) / scale
                    stats[:beta_gap_clean] = max(stats[:beta_gap_clean], gap)
                    @test gap <= 1.0e-13
                    if cls[l] == :wall && abs(beta[l]) > 1.0e-3 * scale
                        stats[:printed_fails] += 1                                  # the printed formula (without beta) is wrong here
                        stats[:printed_fails_max] = max(stats[:printed_fails_max], abs(beta[l]) / scale)
                    end
                    cls[l] == :regular && @test abs(beta[l]) <= 1.0e-13 * scale
                end
            end
            # W3 non-zero boundary values (supported on the clean unknowns so that excluded configurations do not enter)
            boundary_faces = [b for (b, bf) in pairs(mesh.boundary_faces) if bf.axis == k]
            bloc = Dict(b => i for (i, b) in enumerate(boundary_faces))
            qv = randn(rng, n) .* clean
            pv = randn(rng, n) .* clean
            qb = randn(rng, length(boundary_faces)); pb = randn(rng, length(boundary_faces))
            q = FaceField{k}(mesh); psi = FaceField{k}(mesh)
            q.interior .= qv; psi.interior .= pv
            q.boundary .= qb; psi.boundary .= pb
            Aq = FaceField{k}(mesh); Apsi = FaceField{k}(mesh)
            convection!(Aq, q, velocity, 0.0, ws; scheme = Centred(), amr_interface_scheme = Centred(), boundary = NoSlip())
            convection!(Apsi, psi, velocity, 0.0, ws; scheme = Centred(), amr_interface_scheme = Centred(), boundary = NoSlip())
            lhs = sum(V .* pv .* Aq.interior) + sum(V .* qv .* Apsi.interior)
            closed = sum(qv .* pv .* (VdivU .+ Bd))
            extra = sum(sigmaU * (qb[bloc[b]] * pv[l] + pb[bloc[b]] * qv[l]) / 2 for (l, b, sigmaU) in normal_boundary_dual_faces(mesh, inc, velocity))
            gap = abs(lhs - closed - extra) / (sum(abs.(V .* pv .* Aq.interior)) + 1.0e-300)
            stats[:affine_gap] = max(stats[:affine_gap], gap)
            @test gap <= 1.0e-12
            # W4 energy rate for solenoidal velocities, per unknown class
            if solenoidal
                vals = randn(rng, n)
                rate_clean = sum(vals[l]^2 * VC[l, l] for l in 1:n if clean[l])
                rate_excluded = sum(vals[l]^2 * VC[l, l] for l in 1:n if !clean[l]; init = 0.0)
                stats[:energy_clean] = max(stats[:energy_clean], abs(rate_clean) / (scale * sum(vals .^ 2)))
                energy_excluded = max(energy_excluded, abs(rate_excluded) / (scale * sum(vals .^ 2)))
                @test abs(rate_clean) <= 1.0e-12 * scale * sum(vals .^ 2)
            end
        end
    end
    println("skew with walls: ", stats)
    MA.excluded(
        "energy rate carried by the unknowns of limitations (i), (ii) (relative to scale * |q|^2)", energy_excluded;
        note = "pre-fix: non-zero where a level transition meets a wall (issue 60) or in cut wall cells (issue 61)"
    )
    @test stats[:printed_fails] > 50                    # the printed right-hand side (interior dual faces only) fails at wall cells
    @test stats[:beta_gap_clean] <= 1.0e-13
end

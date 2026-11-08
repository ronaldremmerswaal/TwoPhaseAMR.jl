# Code check 01: the dilation bound of the projected-liquid carrier, dt * projtol * sum_{chi=1} |c^o|.
#
# Label:  none (suggested eq:vof:projected-dilation-bound); Section 5.9, sentences before (5.31): "A velocity produced by a
#         projection is divergence free only to its solver tolerance ... For the projected carrier every cell with chi = 1 is
#         active, so the dilation is at most dt projtol sum_{chi=1} |c^o|"; stopping rule in Section 5.8.2 after (5.27)
#         ("stops when max_{c: C_c>0} |(div^o proj u)_c| <= max(projtol, .), the largest strong residual ... a round-off floor")
# Source: docs/techref/chapters/05_vof_transport.tex:757-776 (stopping rule), 806-813 (bound), 740-755; code
#         src/two_phase/two_velocity/transport_workspace.jl (_project_liquid_velocity!), src/solvers/multigrid/elliptic_solver.jl
#         (_solve_krylov_with_stats!: target = max(atol, rtol*..., roundoff_floor), roundoff floor = 128 eps * scale, scale >= 1).
# Claims: (i)   every cell with chi = 1 is active (C > 0), so the active-cell residual bound covers all dilating cells;
#         (ii)  the max over active cells of the open divergence of the projected velocity is at most the solver's target
#               max(projtol, round-off floor) (it equals the solver's strong residual to round-off);
#         (iii) the volume change of a step without snaps equals the dilation dt sum_{chi=1} |c^o| (div^o proj u)_c;
#         (iv)  hence |dV| <= dt * max(projtol, floor) * sum_{chi=1} |c^o|; with the printed bound (projtol only) this fails
#               when the round-off floor exceeds projtol (projtol below ~1e-13);
#         (v)   the chain |sum| <= sum |.| <= tau sum |c^o| is tight for a divergence equal to tau on every chi = 1 cell
#               (equality example with a one-velocity carrier); for solver output the observed ratio is far below 1.
# Method: random two-phase velocities on graded AMR meshes with and without a random wall, ProjectedLiquidVelocity with
#         tolerances 1e-2 ... 1e-15; the divergence is evaluated independently from the faces (sum o |f| a_f v_f / |c^o|).
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/vof-projected-dilation/scripts/code_01_projection_bound.jl
#         (from the repository root)

using TwoPhaseAMR
using Test, Random, Printf
import LinearAlgebra
const AMR = TwoPhaseAMR
isdefined(Main, :VofAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "vof_audit.jl"))
using .VofAudit
const D = VofAudit.D

"""Open divergence `|c^o| (div^o v)_c = sum_f o |f| a_f v_f` per cell, from the face data (independent of the package)."""
function open_divergence_volumes(field, v)
    mesh = field.mesh
    incidence = D.cell_faces(mesh)
    return Dict(
        ref => sum(
            (
                o * mesh.faces[index].area * D.aperture_of(field, index) * v.interior[index]
                    for (index, o) in get(incidence, ref, Tuple{Int, Int}[])
            ); init = 0.0
        ) for ref in D.all_cells(mesh)
    )
end

function projection_case(rng; kind, tolerance, order = (1, 2))
    field, _ = VofAudit.random_field(rng; cut = kind == :cut, levels = 1)
    mesh = field.mesh
    ul, ug = VofAudit.random_phase_velocity(rng, field; noise = 0.4, opposite = 0.2)
    reconstruct!(field)
    prepared = try
        VofAudit.prepare_carrier(:projected, field, (ul, ug); tolerance)
    catch err
        err isa ErrorException || rethrow()
        return (; skipped = first(split(sprint(showerror, err), '\n')))
    end
    dt = prepared.dt
    work = prepared.work
    state = work.method_workspace[]
    stats = state.solver.stats
    projected = prepared.carrier.velocity
    cells = D.all_cells(mesh)
    chi = Dict(ref => D.cond(field, ref) > 0.5 for ref in cells)
    active = Dict(ref => D.cond(field, ref) > 0 for ref in cells)
    m = open_divergence_volumes(field, projected)
    open = Dict(ref => D.open_volume_of(field, ref) for ref in cells)
    maxdiv = maximum(abs(m[ref]) / open[ref] for ref in cells if active[ref] && open[ref] > 0)
    chi_volume = sum(open[ref] for ref in cells if chi[ref])
    dilation = dt * sum(m[ref] for ref in cells if chi[ref])
    dilation_abs = dt * sum(abs(m[ref]) for ref in cells if chi[ref])
    # run the package step and the independent replica (snaps)
    replica = try
        D.wy_replica(field, prepared.carrier, dt, order)
    catch err
        (err isa AssertionError || err isa ErrorException) || rethrow()
        return (; skipped = "replica: " * first(split(sprint(showerror, err), '\n')))
    end
    trial = copy(field)
    work2 = WYAdvectionWorkspace(trial, (ul, ug); interface_transport = ProjectedLiquidVelocity(; tolerance, max_cycles = 1000))
    before = D.liquid_volume(trial)
    advect!(trial, (ul, ug), dt, work2; sweep_order = order)
    change = D.liquid_volume(trial) - before
    return (;
        dt, stats, maxdiv, chi_volume, dilation, dilation_abs, change, snaps = replica.snaps,
        all_chi_active = all(active[ref] for ref in cells if chi[ref]),
        replica_dilation = replica.dilation_total, snapped = replica.snapped,
        reported = phase_flux_diagnostics(work2).projection_final_divergence, tolerance,
    )
end

@testset "vof projected dilation bound" begin
    rng = MersenneTwister(20261112)
    rows = NamedTuple[]
    for kind in (:plain, :cut), tolerance in (1.0e-2, 1.0e-4, 1.0e-8, 1.0e-12, 1.0e-15), rep in 1:2
        res = projection_case(rng; kind, tolerance, order = rep == 1 ? (1, 2) : (2, 1))
        if hasproperty(res, :skipped)
            println("  skipped ($kind, tol=$tolerance): ", res.skipped)
            continue
        end
        push!(rows, merge(res, (; kind)))
    end
    @test length(rows) >= 12
    for r in rows
        @printf(
            "  %-5s tol=%.0e  max|div|=%.2e  solver target=%.2e  final=%.2e  |dV|=%.2e  dilation=%.2e  bound(projtol)=%.2e  bound(target)=%.2e  snaps=%d\n",
            r.kind, r.tolerance, r.maxdiv, r.stats.target_linf, r.stats.final_linf, abs(r.change), r.dilation,
            r.dt * r.tolerance * r.chi_volume, r.dt * r.stats.target_linf * r.chi_volume, r.snaps
        )
    end
    # (i)
    @test all(r -> r.all_chi_active, rows)
    # (ii) the independent divergence respects the solver's target and equals its final residual to round-off
    @test all(r -> r.stats.converged, rows)
    @test all(r -> r.maxdiv <= r.stats.target_linf + 1.0e-13, rows)
    @test all(r -> abs(r.maxdiv - r.stats.final_linf) <= 1.0e-12 * max(1, r.stats.target_linf) + 1.0e-13, rows)
    @test all(r -> abs(r.maxdiv - r.reported) <= 1.0e-13, rows)
    # (iii) volume change = dilation (+ snaps), snaps absent in these runs
    nosnap = filter(r -> r.snaps == 0, rows)
    @test length(nosnap) >= 10
    @test all(r -> abs(r.change - r.dilation) <= 1.0e-13, nosnap)
    @test all(r -> abs(r.replica_dilation - r.dilation) <= 1.0e-13, nosnap)
    # (iv) bound with max(projtol, floor): always; with projtol only: false below the floor
    @test all(r -> abs(r.change) <= r.dt * r.stats.target_linf * r.chi_volume + 1.0e-14, nosnap)
    @test all(r -> r.dilation_abs <= r.dt * r.stats.target_linf * r.chi_volume + 1.0e-14, nosnap)
    above = filter(r -> r.tolerance >= 1.0e-8, nosnap)
    @test all(r -> r.stats.target_linf <= r.tolerance * (1 + 1.0e-12), above)       # printed bound valid above the floor
    below = filter(r -> r.tolerance <= 1.0e-15, nosnap)
    println("tolerance 1e-15: solver target / projtol = ", [r.stats.target_linf / r.tolerance for r in below])
    @test !isempty(below) && all(r -> r.stats.target_linf > 10 * r.tolerance, below)  # the round-off floor takes over
    @test any(r -> r.dilation_abs > r.dt * r.tolerance * r.chi_volume, below)         # sum of |dilation| exceeds projtol bound
    ratios = [abs(r.change) / (r.dt * r.stats.target_linf * r.chi_volume) for r in nosnap if r.stats.target_linf > 0]
    println("observed |dV| / (dt * target * sum_{chi=1}|c^o|): ", extrema(ratios))
end

# (v) equality example: a one-velocity carrier with open divergence exactly tau on every chi = 1 cell attains the bound
@testset "vof projected dilation bound: the chain is tight (one-velocity carrier with div = tau on chi = 1)" begin
    field, _ = D.vof_fixture(:uniform)
    mesh = field.mesh
    reconstruct!(field)
    cells = D.all_cells(mesh)
    index = Dict(ref => k for (k, ref) in enumerate(cells))
    chi = [D.cond(field, ref) > 0.5 for ref in cells]
    tau = 1.0e-3
    n1 = count(chi); n0 = length(cells) - n1
    d = [chi[k] ? tau : -tau * n1 / n0 for k in eachindex(cells)]
    L = zeros(length(cells), length(cells))                      # graph Laplacian over the owned faces (unit cells, |f| = 1)
    for face in mesh.faces
        i, j = index[face.minus], index[face.plus]
        L[i, i] -= 1; L[j, j] -= 1; L[i, j] += 1; L[j, i] += 1
    end
    L[1, :] .= 0; L[1, 1] = 1; d[1] = 0                          # pin one cell (the data sum to zero)
    phi = L \ d
    velocity = FaceVelocity(mesh)
    for (k, face) in pairs(mesh.faces)
        velocity.interior[k] = phi[index[face.plus]] - phi[index[face.minus]]
    end
    m = open_divergence_volumes(field, velocity)
    max_deviation = maximum(abs(m[ref] - (chi[index[ref]] ? tau : -tau * n1 / n0)) for ref in cells if index[ref] != 1)
    work = WYAdvectionWorkspace(field)
    dt = stable_timestep(velocity, work)
    before = D.liquid_volume(field)
    replica = D.wy_replica(copy(field), D.one_velocity_carrier(velocity), dt, (1, 2))
    advect!(field, velocity, dt, work)
    change = D.liquid_volume(field) - before
    bound = dt * tau * n1                                         # unit cells: sum_{chi=1} |c^o| = n1
    println(
        "equality example: divergence error ", max_deviation, "; dV = ", change, ", bound dt tau n1 = ", bound,
        ", ratio ", change / bound, ", snaps ", replica.snaps
    )
    @test max_deviation <= 1.0e-12
    @test replica.snaps == 0
    @test isapprox(change / bound, 1; atol = 1.0e-9)
end

# Code check 04: search for a violation of 0 <= C <= 1 where the disjointness argument does not apply.
#
# Label:  none (paragraph "Boundedness." of Section 5.5)
# Source: docs/techref/chapters/05_vof_transport.tex:296-322: "For a cut coarse donor with two segments on a side neither argument
#         applies"; and (by the same reasoning) the frozen carrier in cut donors, whose subface rectangles are not full width.
# Question: is the boundedness of a sweep actually lost there (conservative-phase outflow larger than the phase volume because two
#         rectangles overlap in the open conservative phase), or only the proof?  Method: randomised search over STRAIGHT walls and
#         STRAIGHT liquid interfaces (so that the geometry is consistent across cells and realisable by the package) on a 4x4 mesh,
#         with one donor cell [1,2]x[2,3] (class "hanging": its right side is two segments next to a refined block; class "uniform":
#         the same cell on a uniform mesh) and random outgoing face velocities at the stable step; the pre-clamp fraction of the donor
#         after the x sweep is evaluated with the independent polygon fluxes (Docs.replica_fluxes), the step is the package's
#         stable_timestep.  Candidates are then refined by hill-climbing.  A found violation is confirmed with the package's
#         `advect!` (error "WY conditional fraction out of bounds").
# Carriers: one-velocity (full-width rectangles) and frozen subfaces (random independent liquid/gas face velocities).
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/vof-boundedness/scripts/code_04_overlap_search.jl
#         (from the repository root)

using TwoPhaseAMR
using Test, Random, Printf
const AMR = TwoPhaseAMR
isdefined(Main, :VofAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "vof_audit.jl"))
using .VofAudit
const D = VofAudit.D

include(joinpath(@__DIR__, "search_helpers.jl"))

"""
Evaluate velocity samples on a built field.  Returns the worst violation `v` (positive = out of bounds), the sample, and counters.
carrier = :one or :frozen.
"""
function best_velocity(rng, field, donor, carrier; samples = 20)
    mesh = field.mesh
    cell_open = D.open_volume_of(field, donor)
    c0 = D.cond(field, donor)
    (0 < D.solid_fraction_of(field, donor) < 1 && 0 < c0 < 1) || return (; v = -Inf, reason = :not_mixed_cut)
    chi = c0 > 0.5 ? 1.0 : 0.0
    right, left = donor_faces(mesh, donor)
    best = (; v = -Inf, reason = :none)
    stats = (; strip = 0, evaluated = 0, overlap = 0)
    strip = evaluated = overlap_count = phase_overlap_count = 0
    closest = Inf
    for _ in 1:samples
        velocity = FaceVelocity(mesh)
        velocity2 = FaceVelocity(mesh)
        magnitudes = Float64[]
        for index in right
            m = rand(rng) < 0.15 ? 0.0 : exp(log(0.05) * rand(rng)); velocity.interior[index] = m; push!(magnitudes, m)
        end
        for index in left
            m = rand(rng) < 0.15 ? 0.0 : exp(log(0.05) * rand(rng)); velocity.interior[index] = -m; push!(magnitudes, m)
        end
        if carrier == :frozen
            for index in vcat(right, left)
                s = mesh.faces[index].minus == donor ? 1.0 : -1.0       # outgoing direction for the donor
                m = rand(rng) < 0.15 ? 0.0 : exp(log(0.05) * rand(rng))
                velocity2.interior[index] = s * m * (rand(rng) < 0.25 ? -1.0 : 1.0)
            end
            for index in vcat(right, left)                              # liquid: outgoing mostly
                velocity.interior[index] *= (rand(rng) < 0.25 ? -1.0 : 1.0)
            end
        end
        prepared = try
            carrier == :one ? VofAudit.prepare_carrier(:one, field, velocity) :
                VofAudit.prepare_carrier(:frozen, field, (velocity, velocity2))
        catch err
            err isa Union{ArgumentError, ErrorException} || rethrow()
            continue
        end
        dt = prepared.dt
        (isfinite(dt) && dt > 0) || continue
        flux_result = nothing
        try
            flux_result = D.replica_fluxes(field, prepared.carrier, dt, 1)
            evaluated += 1
        catch err
            err isa ErrorException || rethrow()
            strip += 1
            continue
        end
        volumes, liquids, _, regions = flux_result
        incidence = D.cell_faces(mesh)
        V = L = 0.0
        for (index, o) in incidence[donor]
            mesh.faces[index].axis == 1 || continue
            V += o * volumes[index]; L += o * liquids[index]
        end
        cpre = c0 + (chi * V - L) / cell_open
        v = chi == 0 ? -cpre : cpre - 1
        # overlap bookkeeping: open overlap, and overlap inside the conservatively moved phase (double counting)
        rs = [r for r in regions if r.result.donor == donor]
        counted = false; phase_overlap = 0.0
        for a in 1:length(rs), b in (a + 1):length(rs)
            inter = VofAudit.rect_intersection(rs[a].result.rectangle, rs[b].result.rectangle)
            isempty(inter) && continue
            op = D.open_part(field, donor, inter)
            isempty(op) && continue
            counted || (overlap_count += 1; counted = true)
            n, aa = D.liquid_plane(field, donor)
            ph = G.clip_halfplane(op, n, aa; keep_less = chi == 0)
            phase_overlap = max(phase_overlap, G.shoelace(ph))
        end
        if phase_overlap > 1.0e-12
            phase_overlap_count += 1
            out_cons = 0.0
            for r in rs
                carrier_volume = carrier == :frozen ?
                    abs(
                        dt * (r.phase == 1 ? velocity : velocity2).interior[r.index] * mesh.faces[r.index].area *
                        prepared.carrier.apertures[r.phase][r.index]
                    ) : abs(volumes[r.index])
                l = abs(r.result.liquid)
                out_cons += chi == 0 ? l : carrier_volume - l
            end
            v_cons = cell_open * (chi == 1 ? 1 - c0 : c0)
            closest = min(closest, (v_cons - out_cons) / cell_open)
        end
        if v > best.v
            best = (; v, cpre, chi, dt, velocity, velocity2, rects = length(rs), reason = :ok)
        end
    end
    return merge(best, (; strip, evaluated, overlap_count, phase_overlap_count, closest))
end

function search(rng, kind, carrier; geometries, samples, artifact = false)
    mesh = build_mesh(kind)
    centre = kind == :uniform ? (1.5, 1.5) : (1.5, 2.5)
    donor = find_cell(mesh, centre...)
    results = NamedTuple[]
    total = (; geometries = 0, mixed = 0, evaluated = 0, strip = 0, overlap = 0)
    counts = Dict{Symbol, Any}(
        :geometries => 0, :mixed => 0, :evaluated => 0, :strip => 0, :overlap_samples => 0, :overlap_geoms => 0,
        :phase_overlap_samples => 0, :closest => Inf
    )
    for _ in 1:geometries
        p = random_params(rng)
        artifact && (p = merge(p, (; ow = 1.5 * p.ow)))
        field = try
            build_fields(mesh, centre, p; artifact)
        catch err
            err isa Union{ArgumentError, ErrorException} || rethrow(); continue
        end
        counts[:geometries] += 1
        r = best_velocity(rng, field, donor, carrier; samples)
        r.v == -Inf && continue
        counts[:mixed] += 1
        counts[:evaluated] += r.evaluated; counts[:strip] += r.strip; counts[:overlap_samples] += r.overlap_count
        counts[:phase_overlap_samples] += r.phase_overlap_count
        counts[:closest] = min(counts[:closest], r.closest)
        r.overlap_count > 0 && (counts[:overlap_geoms] += 1)
        push!(results, merge((; p), r))
    end
    return results, counts, mesh, donor, centre
end

function hill_climb(rng, mesh, donor, centre, carrier, start; steps = 150, samples = 20, artifact = false)
    best = start
    for step in 1:steps
        scale = 0.25 * 0.97^step
        p = (;
            tw = best.p.tw + scale * randn(rng), ow = best.p.ow + 0.3 * scale * randn(rng),
            tl = best.p.tl + scale * randn(rng), ol = best.p.ol + 0.4 * scale * randn(rng),
        )
        field = try
            build_fields(mesh, centre, p; artifact)
        catch err
            err isa Union{ArgumentError, ErrorException} || rethrow(); continue
        end
        r = best_velocity(rng, field, donor, carrier; samples)
        r.v == -Inf && continue
        r.v > best.v && (best = merge((; p), r))
    end
    return best
end

function confirm(mesh_kind, carrier, centre, candidate; artifact = false)
    mesh = build_mesh(mesh_kind)
    field = build_fields(mesh, centre, candidate.p; artifact)
    dt = candidate.dt
    velocity = FaceVelocity(mesh); copyto!(velocity.interior, candidate.velocity.interior)
    velocity2 = FaceVelocity(mesh); copyto!(velocity2.interior, candidate.velocity2.interior)
    work = carrier == :one ? WYAdvectionWorkspace(field) :
        WYAdvectionWorkspace(field, (velocity, velocity2); interface_transport = FrozenInterfaceFluxes())
    try
        advect!(field, carrier == :one ? velocity : (velocity, velocity2), dt, work; check_timestep = false)
        return :no_error
    catch err
        err isa ErrorException || rethrow()
        return occursin("out of bounds", sprint(showerror, err)) ? :out_of_bounds : Symbol("other: " * first(split(sprint(showerror, err), '\n')))
    end
end

@testset "vof boundedness: search for overlap-induced violations" begin
    rng = MersenneTwister(20261114)
    summary = Dict{Tuple{Symbol, Symbol, Bool}, Any}()
    for (kind, carrier, geometries, artifact) in (
            (:hanging, :one, 3000, false), (:uniform, :one, 600, false), (:uniform, :frozen, 3000, false),
            (:hanging, :frozen, 3000, false), (:hanging, :one, 3000, true), (:uniform, :one, 1500, true), (:hanging, :frozen, 2000, true),
        )
        results, counts, mesh, donor, centre = search(rng, kind, carrier; geometries, samples = 14, artifact)
        sort!(results; by = r -> -r.v)
        best = isempty(results) ? nothing : results[1]
        refined = best
        if !isempty(results)
            # hill-climb the most promising candidates
            for start in results[1:min(3, length(results))]
                candidate = hill_climb(rng, mesh, donor, centre, carrier, start; steps = 100, samples = 14, artifact)
                candidate.v > refined.v && (refined = candidate)
            end
        end
        outcome = best === nothing ? :none : (refined.v > 1.0e-9 ? confirm(kind, carrier, centre, refined; artifact) : :no_violation)
        label = string(kind, "/", carrier, artifact ? "/artifact" : "/straight walls")
        println(
            "search ", label, ": ", counts, "; best violation v after random search ", best === nothing ? nothing : best.v,
            ", after hill climbing ", refined === nothing ? nothing : refined.v, "; code outcome for the best candidate: ", outcome
        )
        if refined !== nothing && refined.v > -Inf
            println("   best candidate: ", refined.p, ", cpre = ", refined.cpre, ", chi = ", refined.chi, ", dt = ", refined.dt)
            if refined.v > 1.0e-9
                right, left = donor_faces(mesh, donor)
                println(
                    "   donor face velocities (right segments, left face): ", [refined.velocity.interior[i] for i in right], " ",
                    [refined.velocity.interior[i] for i in left], carrier == :frozen ? " gas: " : "",
                    carrier == :frozen ? ([refined.velocity2.interior[i] for i in right], [refined.velocity2.interior[i] for i in left]) : ""
                )
            end
        end
        summary[(kind, carrier, artifact)] = (; counts, best = refined === nothing ? nothing : refined.v, outcome)
    end
    # sanity: the search explores the relevant regime (rectangles of the donor overlapping in open parts occur)
    @test summary[(:hanging, :one, false)].counts[:mixed] > 200
    @test summary[(:hanging, :one, true)].counts[:mixed] > 200
    # classes covered by the proof: no violation (controls), with straight walls and with the artefact apertures
    @test summary[(:uniform, :one, false)].best === nothing || summary[(:uniform, :one, false)].best <= 1.0e-12
    @test summary[(:uniform, :one, true)].best === nothing || summary[(:uniform, :one, true)].best <= 1.0e-12
end

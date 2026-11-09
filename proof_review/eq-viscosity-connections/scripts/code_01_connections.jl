# Code check 01: eq:viscosity:connections -- measures and distances of the dual-face connections of the single-phase operator.
#
# Label:  eq:viscosity:connections
# Source: docs/techref/chapters/09_viscosity.tex:38-60; src/operators/staggered_diffusion.jl (_dual_diffusion_connections)
# Checks (components 1 and 2):
#   C1  uniform periodic mesh: every connection has measure h and distance h; each unknown has 4 connections (2 per axis);
#   C2  composite meshes (random, levels 1-2, periodic and walled): for regular connections (not flagged as refinement interface)
#       meas * dist = V_minus = V_plus (the dual face is the length of the geometric dual face), distance = |position difference
#       along the axis| (nearest periodic image);
#   C3  closure: for every unknown away from walls the measures of its connections sum, per axis, to the perimeter projection of the
#       dual cell: 2|f| along the face axis k (two sides of length |f|) and 2 V_f/|f| along the other axis (top and bottom, each
#       of length V_f/|f| = mean width), irrespective of the minus/plus orientation of a connection (which is not
#       consistent across a periodic seam).  Holds for unknowns whose connections are all regular; at an unknown with a
#       refinement-interface connection the net measures |sum_g' w |f'|| are SMALLER than the geometric dual-face lengths
#       (the transverse rule spreads the weights over several partner unknowns with cancellation): the defect is reported;
#   C4  the flag `refinement_interface` equals the non-regularity of Chapter 6 (eq:momentum:regular: weight count != 2, |weight| != 1/2,
#       or V_minus != V_plus);
#   C5  a connection pair can have BOTH axes (split connections): count of pairs with two connections, and of connections with a
#       tangential offset |(x+ - x-)_s| > 0 (the widened-derivative rows of eq:viscosity:gradient).
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-viscosity-connections/scripts/code_01_connections.jl
using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ViscosityAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "viscosity_audit.jl"))
using .ViscosityAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR
const MA = ViscosityAudit.MA

function connections_of(mesh, axis)
    field = FaceField{axis}(mesh)
    schedule = AMR.StaggeredAdvectionWorkspace(field).schedule
    g2l, l2g = AMR._component_local_to_global(mesh, axis)
    volumes = schedule isa AMR.UniformDualAdvectionSchedule ? fill(schedule.volume, length(l2g)) : schedule.volumes
    return AMR._dual_diffusion_connections(mesh, schedule, l2g), schedule, l2g, volumes
end

@testset "C1: uniform periodic mesh" begin
    mesh = BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (true, true))
    for axis in 1:2
        conns, schedule, l2g, volumes = connections_of(mesh, axis)
        @test length(conns) == 2 * length(l2g)
        @test all(c -> c.measure ≈ 0.5 && c.distance ≈ 0.5, conns)
        deg = zeros(Int, length(l2g))
        for c in conns
            deg[c.minus] += 1; deg[c.plus] += 1
        end
        @test all(==(4), deg)
    end
end

@testset "C2-C5: composite meshes" begin
    rng = MersenneTwister(555)
    nsplit = 0; nboth = 0; nreg = 0; nnon = 0; worst_prod = 0.0; worst_close = 0.0; defect_nonreg = 0.0; nunk_nonreg = 0; nunk = 0; nonreg_unknown = Set{Int}()
    for periodic in ((true, true), (false, false), (true, false)), levels in (1, 2), trial in 1:2
        mesh = MA.random_mesh(rng; periodic, levels, refine_probability = 0.5)
        for axis in 1:2
            conns, schedule, l2g, volumes = connections_of(mesh, axis)
            tot = zeros(2, length(l2g)); nonreg_unknown = Set{Int}()
            pairs_seen = Dict{Tuple{Int, Int}, Int}()
            for c in conns
                fm = mesh.faces[l2g[c.minus]]; fp = mesh.faces[l2g[c.plus]]
                d = abs(wrapped(fp.center[c.axis] - fm.center[c.axis], mesh.extent[c.axis], mesh.periodic[c.axis]))
                @test c.distance ≈ d atol = 1.0e-13
                tot[c.axis, c.minus] += c.measure; tot[c.axis, c.plus] += c.measure
                pairs_seen[(c.minus, c.plus)] = get(pairs_seen, (c.minus, c.plus), 0) + 1
                s = 3 - c.axis
                off = abs(wrapped(fp.center[s] - fm.center[s], mesh.extent[s], mesh.periodic[s]))
                off > 1.0e-12 && (nsplit += 1)
                c.refinement_interface && (push!(nonreg_unknown, c.minus); push!(nonreg_unknown, c.plus))
                if !c.refinement_interface
                    nreg += 1
                    worst_prod = max(worst_prod, abs(c.measure * c.distance - volumes[c.minus]) / volumes[c.minus], abs(c.measure * c.distance - volumes[c.plus]) / volumes[c.plus])
                else
                    nnon += 1
                end
            end
            nboth += count(==(2), values(pairs_seen))
            # closure for unknowns whose dual cell does not touch a wall: transverse extent
            for i in eachindex(l2g)
                f = mesh.faces[l2g[i]]
                touches_wall = !all(mesh.periodic) && any(
                    abs(f.center[a] - mesh.origin[a]) < 1.1 * maximum(maximum(b.spacing) for b in mesh.blocks) ||
                        abs(f.center[a] - mesh.origin[a] - mesh.extent[a]) < 1.1 * maximum(maximum(b.spacing) for b in mesh.blocks) for a in 1:2 if !mesh.periodic[a]
                )
                touches_wall && continue
                expected = axis == 1 ? (2 * f.area, 2 * volumes[i] / f.area) : (2 * volumes[i] / f.area, 2 * f.area)
                for a in 1:2
                    defect = abs(tot[a, i] - expected[a]) / expected[a]
                    if i in nonreg_unknown
                        defect_nonreg = max(defect_nonreg, defect); a == 1 && (nunk_nonreg += 1)
                    else
                        worst_close = max(worst_close, defect); a == 1 && (nunk += 1)
                    end
                end
            end
        end
    end
    println("regular connections ", nreg, " (meas*dist=V defect ", worst_prod, "), non-regular ", nnon, ", pairs with two connections ", nboth, ", connections with a tangential offset ", nsplit, ", closure defect (all-regular unknowns, ", nunk, ") ", worst_close, "; at unknowns with a refinement-interface connection (", nunk_nonreg, ") up to ", defect_nonreg)
    @test worst_prod <= 1.0e-12
    @test worst_close <= 1.0e-12
    @test nsplit > 0
end

@testset "C4: flag = non-regular of Chapter 6" begin
    rng = MersenneTwister(556)
    mesh = MA.random_mesh(rng; periodic = (true, true), levels = 2, refine_probability = 0.5)
    for axis in 1:2
        conns, schedule, l2g, volumes = connections_of(mesh, axis)
        mismatches = 0
        # one flag per dual face of the schedule; connections inherit the flag of the (minus, plus) group (|=)
        flags = Dict{Tuple{Int, Int}, Bool}()
        for face in schedule.faces
            weights = schedule.flux_weights[face.first_weight:(face.first_weight + face.weight_count - 1)]
            nonreg = face.weight_count != 2 || any(w -> !isapprox(abs(w), 0.5; atol = 1.0e-14), weights) || !isapprox(volumes[face.minus], volumes[face.plus]; atol = 1.0e-13)
            flags[(face.minus, face.plus)] = get(flags, (face.minus, face.plus), false) | nonreg
            mismatches += nonreg == face.refinement_interface ? 0 : 1
        end
        @test mismatches == 0
        for c in conns
            @test c.refinement_interface == flags[(c.minus, c.plus)]
        end
    end
end

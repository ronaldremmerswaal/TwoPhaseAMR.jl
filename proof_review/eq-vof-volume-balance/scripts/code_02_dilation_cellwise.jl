# Code check 02: when is the liquid volume conserved?  The two bullets after (5.30): the carrier is discretely divergence
# free in every cell with chi = 1, and no fraction is clamped or snapped.
#
# Label:  eq:vof:volume-balance
# Source: docs/techref/chapters/05_vof_transport.tex:797-815
# Claims: (i)   a divergence-free carrier (streamfunction face velocities, exactly solenoidal on composite meshes; open
#               streamfunction on cut meshes) conserves the liquid volume to round-off;
#         (ii)  divergence-freeness is needed only in the cells with chi = 1: a perturbation on faces whose two cells have
#               chi = 0 (so the field is no longer solenoidal there) leaves the volume change at round-off;
#         (iii) a perturbation touching a cell with chi = 1 changes the volume by the dilation of (5.30), which the
#               independent replica predicts (change - dilation = round-off);
#         (iv)  the dilation of a single sweep does not vanish individually, only the sum over the two sweeps does
#               (Section 5.3: "cancels over the two sweeps").
# Method: docs fixtures :uniform, :amr (periodic, drop), :cut, :film (wall, open streamfunction) and random perturbations;
#         the independent replica of docs/techref/figures/vof.jl (`step_balance_check`).
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-vof-volume-balance/scripts/code_02_dilation_cellwise.jl
#         (from the repository root)

using TwoPhaseAMR
using Test, Random, Printf
const AMR = TwoPhaseAMR
isdefined(Main, :VofAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "vof_audit.jl"))
using .VofAudit
const D = VofAudit.D

"""Add `amplitude*randn` to the faces selected by `select(face_index, chi_minus, chi_plus)`."""
function perturbed(rng, field, base, select; amplitude)
    reconstruct!(field)
    mesh = field.mesh
    velocity = copy(base)
    touched = 0
    for (index, face) in pairs(mesh.faces)
        chim = D.cond(field, face.minus) > 0.5
        chip = D.cond(field, face.plus) > 0.5
        if select(index, chim, chip)
            velocity.interior[index] += amplitude * randn(rng)
            touched += 1
        end
    end
    return velocity, touched
end


"""Volume balance of one step with dt = dtwy / 2^attempt; `nothing` if the independent replica or the package refuses the step."""
function balance_at(field, velocity, order, attempt)
    trial = copy(field)
    work = WYAdvectionWorkspace(trial)
    dt = stable_timestep(velocity, work) / 2^attempt
    before = D.liquid_volume(trial)
    replica = try
        D.wy_replica(trial, D.one_velocity_carrier(velocity), dt, order)
    catch err
        (err isa AssertionError || err isa ErrorException) || rethrow()
        return nothing
    end
    try
        advect!(trial, velocity, dt, work; sweep_order = order)
    catch err
        err isa ErrorException || rethrow()
        return nothing
    end
    change = D.liquid_volume(trial) - before
    return (;
        change, before, dilation = replica.dilation_total, snapped = replica.snapped, snaps = replica.snaps,
        residual = abs(change - replica.dilation_total - replica.snapped),
    )
end

@testset "eq:vof:volume-balance: cellwise dilation" begin
    rng = MersenneTwister(31415)
    @testset "(i) solenoidal carriers conserve volume to round-off" begin
        halvings = Dict{Tuple{Symbol, Tuple{Int, Int}}, Int}()
        for kind in (:uniform, :amr, :cut, :film)
            field, velocity = D.vof_fixture(kind)
            if kind in (:cut, :film)
                velocity = D.open_streamfunction_velocity(field, D.vof_psi_walls)
            end
            for order in ((1, 2), (2, 1))
                result = nothing
                for attempt in 0:6
                    result = balance_at(field, velocity, order, attempt)
                    result === nothing || (halvings[(kind, order)] = attempt; break)
                end
                @test result !== nothing
                result === nothing && continue
                @test result.snaps == 0
                @test abs(result.change) <= 1.0e-13 * result.before
                @test result.residual <= 1.0e-13 * result.before
            end
        end
        println("(i) halvings of the stable step needed (cut-donor strip limitation, see vof-boundedness): ", halvings)
    end
    @testset "(ii)/(iii) perturbations: only chi = 1 cells matter" begin
        gas_dilation = Float64[]; liquid_dilation = Float64[]; liquid_change = Float64[]
        for kind in (:uniform, :amr), rep in 1:3
            field, velocity = D.vof_fixture(kind)
            # (ii) faces whose cells both have chi = 0
            v2, touched2 = perturbed(rng, field, velocity, (i, m, p) -> !m && !p; amplitude = 0.2)
            s2 = D.step_balance_check(copy(field), v2)
            @test touched2 > 100
            @test s2.snaps == 0
            @test abs(s2.change) <= 1.0e-13 * s2.before
            @test abs(s2.dilation) <= 1.0e-13 * s2.before
            push!(gas_dilation, abs(s2.change))
            # (iii) faces next to a chi = 1 cell
            v3, touched3 = perturbed(rng, field, velocity, (i, m, p) -> m || p; amplitude = 0.05)
            s3 = D.step_balance_check(copy(field), v3)
            @test touched3 > 20
            @test s3.snaps == 0
            @test s3.residual <= 1.0e-13 * s3.before                 # change = dilation (replica) to round-off
            @test abs(s3.dilation) > 1.0e-4 * s3.before              # and the dilation is not small
            push!(liquid_dilation, abs(s3.dilation) / s3.before); push!(liquid_change, abs(s3.change) / s3.before)
        end
        println("(ii) |volume change| with chi=0-only perturbation, worst: ", maximum(gas_dilation))
        println("(iii) relative dilation / change with chi=1 perturbation: ", extrema(liquid_dilation), extrema(liquid_change))
    end
    @testset "(iv) per-sweep dilation does not vanish for a solenoidal carrier, the sum does" begin
        field, velocity = D.vof_fixture(:uniform)
        work = WYAdvectionWorkspace(field)
        dt = stable_timestep(velocity, work)
        replica = D.wy_replica(field, D.one_velocity_carrier(velocity), dt, (1, 2))
        chi = replica.chi
        first = replica.states[2].dilation; second = replica.states[3].dilation
        sums = maximum(abs(first[ref] + second[ref]) for ref in keys(first) if chi[ref] == 1)
        singles = maximum(abs(first[ref]) for ref in keys(first) if chi[ref] == 1)
        println("(iv) max single-sweep dilation / cell volume ", singles, ", max |sum over both sweeps| ", sums)
        @test singles > 1.0e-2
        @test sums <= 1.0e-14
    end
end

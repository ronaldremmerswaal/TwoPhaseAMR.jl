# Code check 02: the step-selection rule eq:stab:selection, the last-step claim, the retained state and the reference step against the
# package (simulate!), through a custom observer that reads the reference step the loop hands to step!.
#
# Label:  eq:stab:selection (ch:stability-accuracy)
# Source: docs/techref/chapters/13_stability_accuracy.tex:146-233; src/navier_stokes/simulate.jl:90-172, 262-282;
#         test/navier_stokes/high_level_api.jl:620-640 (the ReferenceStepObserver pattern), 255-303 (adaptive cylinder)
# Checks:
#   D1  the recorded steps and reference steps of real runs (decaying Taylor--Green with constant proposal; Taylor--Green with a force that lowers the
#       bound; chunked calls; prescribed dt followed by an automatic call; the adaptive cylinder with regrids) equal the replica of _replica.jl
#       (same bounds), to rounding; the replica is the literal loop, whose rules step02_selection_replica.jl checks;
#   D2  stretched / shortened / extra last steps constructed from a probe run: ratios, reference = max(proposal, previous step, step), the third
#       term is the maximum only for the stretched last step; the last step lies in [1e-6, 1 + 1e-6) of its proposal (random targets, 60 runs);
#   D3  retained state after a call: proposal (not the shortened/stretched step), last step, defect step; none after a prescribed-dt call;
#       a regrid sets the defect step to the proposal of the step just taken (adaptive cylinder).
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/ch-stability-accuracy/scripts/code_02_selection_real.jl

isdefined(Main, :STAB_CFL) || include(joinpath(@__DIR__, "_prelude.jl"))
isdefined(Main, :replica_call) || include(joinpath(@__DIR__, "_replica.jl"))

mutable struct SelectionObserver <: AbstractObserver
    flow::Any
    cfl::Float64
    procedure::Any
    bounds::Vector{Float64}       # bound at the start of step n = value at event n - 1 (event 0 = initial)
    dts::Vector{Float64}
    refs::Vector{Float64}
    regrid::Vector{Bool}
end
SelectionObserver(flow; cfl = STAB_CFL) = SelectionObserver(flow, cfl, nothing, Float64[], Float64[], Float64[], Bool[])

function current_bound(flow, cfl)
    AMR._enforce_current_velocity_boundary!(flow.velocity, flow.system, flow.time)
    return stable_timestep(flow; cfl)
end

function AMR.observe!(o::SelectionObserver, event::Symbol, state, info::StepInfo)
    if event === :initial
        o.procedure = info.step_procedure
        push!(o.bounds, current_bound(o.flow, o.cfl))
    elseif event === :step
        push!(o.dts, info.dt)
        push!(o.refs, o.procedure.reference_dt)
        push!(o.regrid, info.topology_changed)
        push!(o.bounds, current_bound(o.flow, o.cfl))
    end
    return o
end

# One call; returns the observer record and the replica of the same call.
function observed_call!(flow, T; cfl = STAB_CFL, retained = Inf, defect0 = 0.0, dt = nothing, kwargs...)
    t0 = flow.time
    observer = SelectionObserver(flow; cfl)
    options = (; cfl, tolerance = 1.0e-9, max_cycles = 1000)
    isnothing(dt) || (options = merge(options, (; dt)))
    result = simulate!(flow, T; observer, options..., kwargs...)
    flow.time = result.time
    replica = replica_call(
        t0, T, n -> observer.bounds[n]; dt, retained, defect0, regrid = n -> observer.regrid[n],
    )
    return (; observer, replica, result, t0, T)
end

function agree(call; rtol = 1.0e-12)
    obs, rec = call.observer, call.replica.rec
    length(obs.dts) == length(rec) || return false
    return all(
        isapprox(obs.dts[n], rec[n].step; rtol) && isapprox(obs.refs[n], rec[n].reference; rtol) for n in eachindex(rec)
    ) && call.result.time == call.T
end

falling_force = (x, y, time, axis) -> axis == 1 ? 60.0 : 0.0

@testset "ch13 code: step selection against the package" begin
    @testset "D1/D2 Taylor--Green, constant proposal" begin
        probe = stab_taylor_green()
        p = stable_timestep(probe; cfl = STAB_CFL)
        for (label, extra, expect_n) in (("stretched", 0.5e-6, 3), ("extra step", 2.0e-6, 4), ("shortened", 0.3, 4), ("aligned", 0.0, 3))
            flow = stab_taylor_green()
            call = observed_call!(flow, 3p + extra * p)
            @test agree(call)
            n = length(call.observer.dts)
            @test n == expect_n
            ratio = call.observer.dts[end] / p
            refs = call.observer.refs
            if label == "stretched"
                @test 1 < ratio < 1 + 1.0e-6 && refs[end] == call.observer.dts[end] > p       # third term active
            elseif label == "extra step"
                @test ratio ≈ 2.0e-6 rtol = 1.0e-5
                @test refs[end] ≈ p rtol = 1.0e-13           # ref = max(proposal, previous step)
            elseif label == "shortened"
                @test ratio ≈ 0.3 rtol = 1.0e-9
                @test refs[end] ≈ p rtol = 1.0e-13
            end
            @test all(refs[1:(end - 1)] .≈ p) && refs[1] ≈ p                                    # regular steps: ref = proposal
            println("D1 ", label, ": steps ", n, " last ratio ", ratio, " ref/p ", refs[end] / p)
            sel = flow.workspace.simulation_step
            @test sel.step ≈ p rtol = 1.0e-13                                                  # retained: the proposal, not the last step
            @test sel.last_dt == call.observer.dts[end] && sel.defect_dt == call.observer.dts[end]
        end
    end

    @testset "D1/D2 falling proposals (force), probe-constructed targets and random targets" begin
        probe_flow = stab_taylor_green()
        probe = observed_call!(probe_flow, 0.25; force = falling_force)
        @test agree(probe)
        props = getfield.(probe.replica.rec, :proposed)
        @test issorted(props; rev = true) && props[end] < 0.8props[1]                           # the bound falls
        @test any(r -> r.reference > r.proposed * (1 + 1.0e-9), probe.replica.rec)              # previous step above the new proposal is the reference
        times = cumsum(getfield.(probe.replica.rec, :step))
        count_checked = 0
        for k in 2:5, delta in (0.5e-6, 0.999e-6, 1.5e-6, 0.3, 0.9999)
            T = times[k] + delta * props[k + 1]
            # the probe sequence is independent of T up to the last step
            flow = stab_taylor_green()
            call = observed_call!(flow, T; force = falling_force)
            @test agree(call)
            dts = call.observer.dts
            n = length(dts)
            ratio = dts[end] / call.replica.rec[end].proposed
            n >= 2 && @test 1.0e-6 * (1 - 1.0e-6) <= ratio < 1 + 1.0e-6 + 1.0e-9
            @test all(dts[i] ≈ call.replica.rec[i].proposed for i in 1:(n - 1))
            @test all(r.reference >= r.step for r in call.replica.rec)
            third = [i for i in 1:n if call.observer.refs[i] ≈ dts[i] && dts[i] > max(call.replica.rec[i].proposed, call.replica.rec[i].defect) * (1 + 1.0e-12)]
            @test all(==(n), third)                                                              # only the last step
            count_checked += 1
        end
        Random.seed!(2026)
        for trial in 1:60
            T = 0.0 + 0.02 + 0.2rand()
            flow = stab_taylor_green()
            call = observed_call!(flow, T; force = falling_force)
            @test agree(call)
            n = length(call.observer.dts)
            r = call.replica.rec[end]
            n >= 2 && @test 1.0e-6 * (1 - 1.0e-6) <= r.step / r.proposed < 1 + 1.0e-6 + 1.0e-9
        end
        println("D2 falling: ", count_checked, " constructed + 60 random targets agree with the replica; last-step ratio bounds hold")
    end

    @testset "D1/D3 chunked calls, prescribed dt, carried defect" begin
        p = stable_timestep(stab_taylor_green(); cfl = STAB_CFL)
        flow = stab_taylor_green()
        first_call = observed_call!(flow, 2.5p)
        @test agree(first_call)
        sel = flow.workspace.simulation_step
        @test sel.step ≈ p rtol = 1.0e-13
        @test sel.last_dt ≈ 0.5p rtol = 1.0e-9
        second = observed_call!(flow, 5p; retained = first_call.replica.retained, defect0 = first_call.observer.dts[end])
        @test agree(second)
        @test second.observer.dts[1] ≈ p rtol = 1.0e-13
        @test second.observer.refs[1] ≈ p rtol = 1.0e-13   # whole proposal, ref = max(p, 0.5 p, p)
        # prescribed dt: no retained proposal afterwards, defect step retained
        flow = stab_taylor_green()
        dt = 0.3p
        fixed = observed_call!(flow, 1.0p; dt)
        @test agree(fixed)
        @test isinf(flow.workspace.simulation_step.step) && flow.workspace.simulation_step.defect_dt == fixed.observer.dts[end]
        automatic = observed_call!(flow, flow.time + 2p; retained = Inf, defect0 = fixed.observer.dts[end])
        @test agree(automatic)
        @test automatic.observer.refs[1] ≈ max(automatic.observer.bounds[1], fixed.observer.dts[end]) rtol = 1.0e-13   # max(proposal = bound, previous step, step)
        @test automatic.observer.refs[1] > automatic.observer.bounds[1] * (1 - 1.0e-13)
    end

    @testset "D1/D3 adaptive cylinder with regrids" begin
        flow = let
            mesh = BlockTreeMesh(;
                base_mesh = (56, 32), block_size = (4, 4), origin = (-4.0, -4.0), extent = (14.0, 8.0),
                periodic = (false, false), max_level = 2, refine = (key, origin, extent) -> key.level < 1,
                adaptation = MeshAdaptationOptions(update_interval = 2, coarsen_delay = 2, minimum_level = 1),
            )
            geometry = GeometryField(mesh, (x, y) -> hypot(x, y) - 0.5; lipschitz = 1, atol = 1.0e-7, maxdepth = 24)
            inflow = (x, y, time, axis) -> axis == 1 ? 1.0 : 0.0
            velocity = FaceVelocity(mesh, (x, y, time, axis) -> axis == 1 ? 1.0 : 0.005exp(-((x - 0.75)^2 + y^2)))
            boundary = VelocityBoundaryConditions(
                left = Inflow(inflow), right = Outflow(), bottom = ZeroGradientVelocity(), top = ZeroGradientVelocity(),
            )
            criteria = (
                GeometryRefinementIndicator(geometry; level = 2),
                VorticityRefinementIndicator(velocity; refine_tolerance = 0.2, coarsen_tolerance = 0.2 / 3, maximum_level = 2),
            )
            FlowSimulation(
                mesh, SinglePhase(density = 1.0, dynamic_viscosity = 1 / 200);
                velocity, geometry, boundary, adaptivity = Adaptivity(; criteria),
            )
        end
        call = observed_call!(flow, 0.05; cfl = 0.25)
        obs = call.observer
        @test any(obs.regrid)
        @test agree(call; rtol = 1.0e-10)
        rec = call.replica.rec
        # after a regrid at step n the next defect is the proposal of step n (not the step)
        for n in findall(obs.regrid)
            n < length(rec) && @test rec[n + 1].defect == rec[n].proposed
        end
        println("D3 cylinder: ", length(rec), " steps, regrids after steps ", findall(obs.regrid), ", refs agree with the replica")
    end
end

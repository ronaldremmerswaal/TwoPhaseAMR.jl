# Fresh process: julia --heap-size-hint=3G -t 1 --project=benchmark benchmark/latency/inference.jl stokes3_wave
using LinearAlgebra
BLAS.set_num_threads(1)
using SnoopCompileCore
using TwoPhaseAMR

length(ARGS) == 1 && ARGS[1] in ("stokes3_wave", "capillary_wave") ||
    error("usage: inference.jl {stokes3_wave|capillary_wave}")
const example_name = ARGS[1]
module Example end
Base.include(Example, joinpath(@__DIR__, "..", "..", "examples", example_name * ".jl"))
include(joinpath(@__DIR__, "cases.jl"))

function inference_workload()
    flow = setup_case(Val(Symbol(example_name)), Val(false); max_level = 0, dt = 0.001)
    example_name == "stokes3_wave" && initial_projection!(flow)
    one_step!(flow, 0.001)
    return nothing
end

# SnoopCompile itself is intentionally loaded *after* recording.
const inference = @snoop_inference inference_workload()
using SnoopCompile
using Printf

function summarize_inference(inference)
    frames = SnoopCompile.flatten(inference; sortby = nothing)
    # Retain the library's standard inclusive aggregation as a cross-check. Its
    # default inclusive times also contain LLVM; the table below excludes it.
    aggregated = SnoopCompile.accumulate_by_source(frames; by = SnoopCompile.inclusive)
    pure_inference = Dict{Any, Float64}()
    instances = Set{Core.MethodInstance}()
    for frame in frames
        mi = Core.MethodInstance(frame)
        push!(instances, mi)
        method = mi.def isa Method ? mi.def : mi
        method isa Method && method.name === :ROOT && continue # synthetic tree root
        pure_inference[method] = get(pure_inference, method, 0.0) +
            SnoopCompile.inclusive(frame; include_llvm = false)
    end
    ranked = sort!(collect(pure_inference); by = last, rev = true)
    println("Example=", example_name, " Julia=", VERSION, " max_level=0, threads=", Threads.nthreads())
    println("Recorded inference nodes=", length(frames), ", distinct method instances=", length(instances),
        ", aggregate methods (including thunks)=", length(aggregated))
    println("Top 20 methods: aggregate inclusive inference seconds, excluding LLVM (nested times overlap)")
    for (rank, (method, seconds)) in enumerate(Iterators.take(ranked, 20))
        if method isa Method
            @printf("%2d %9.3f %s.%s %s:%d\n", rank, seconds, method.module,
                method.name, basename(String(method.file)), method.line)
        else
            @printf("%2d %9.3f %s\n", rank, seconds, string(method))
        end
    end
    targets = (TwoPhaseAMR.IncompressibleNavierStokesSolver,
        TwoPhaseAMR.NavierStokesWorkspace, TwoPhaseAMR.FlowSimulation)
    found = Dict(target => Set{Any}() for target in targets)
    function collect_types(t)
        if t isa UnionAll
            collect_types(t.body)
        elseif t isa DataType
            for target in targets
                t.name.wrapper === target && isconcretetype(t) && push!(found[target], t)
            end
            foreach(collect_types, t.parameters)
        elseif t isa Union
            collect_types(t.a)
            collect_types(t.b)
        elseif t isa TypeVar
            collect_types(t.lb)
            collect_types(t.ub)
        end
        return nothing
    end
    for mi in instances
        collect_types(mi.specTypes)
    end
    for target in targets
        println("Distinct concrete ", nameof(target), " types in inference signatures: ", length(found[target]))
    end
    kw_sorters = count(mi -> mi.def isa Method && mi.def.name === :kwcall, instances)
    println("Distinct Core.kwcall sorter method instances: ", kw_sorters)
    keyword_instances = filter(instances) do mi
        method = mi.def
        method isa Method || return false
        name = String(method.name)
        return name == "kwcall" || (startswith(name, "#") && occursin("#", name[2:end]))
    end
    println("Keyword-like method instances (name heuristic, also counts closures): ", length(keyword_instances))
    println("Stale instances after loading SnoopCompile for analysis: ", length(SnoopCompile.staleinstances(inference)),
        " (may include invalidations caused by the analyzer)")
end
summarize_inference(inference)

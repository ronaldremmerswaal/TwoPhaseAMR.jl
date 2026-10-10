#!/usr/bin/env julia
module BlockInteriorMakeProbe

include(joinpath(@__DIR__, "..", "..", "Support.jl"))
using .ScalingSupport

const ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
const RELATIVE = joinpath("src","operators","advection","cell_advection.jl")
const DRIVER = joinpath(ROOT,"benchmark","scaling","stage3","main_driver.jl")

function split_region(text, first_anchor, last_anchor)
    first_range = findfirst(first_anchor,text)
    first_range === nothing && error("missing source anchor: $first_anchor")
    last_range = findnext(last_anchor,text,last(first_range)+1)
    last_range === nothing && error("missing source anchor: $last_anchor")
    before = first(first_range) == firstindex(text) ? "" : text[firstindex(text):prevind(text,first(first_range))]
    body = text[first(first_range):prevind(text,first(last_range))]
    after = text[first(last_range):lastindex(text)]
    return before,body,after
end

function insert_once(body, anchor, addition)
    parts = split(body,anchor;limit=3)
    length(parts) == 2 || error("expected exactly one anchor: $anchor")
    return parts[1] * anchor * addition * parts[2]
end

counter(name) = "        @inbounds _STAGE3_PASS_COUNTERS[\"$name\"][Threads.threadid()] = 1\n"

function instrument(original)
    prefix,advect,suffix = split_region(original,
        "function advect!(\n        scalar::CellField", "\nfunction stable_timestep(")
    names = String[]
    if occursin("function _cell_advection_regular_fluxes!(",prefix)
        before,regular,after = split_region(prefix,
            "function _cell_advection_regular_fluxes!(",
            "\nfunction _cell_advection_regular_rates!(")
        regular = insert_once(regular,
            "    _parallel_block_columns(mesh) do b, first_j, last_j\n",
            counter("regular_flux"))
        prefix = before * regular * after
        advect = insert_once(advect,
            "    @_parallel for index in eachindex(workspace.interior_faces)\n",
            counter("residual_flux"))
        append!(names,["regular_flux","residual_flux","gather"])
    else
        anchors = ("    @_parallel for index in eachindex(workspace.interior_faces)\n",
                   "    @_parallel for index in 1:length(mesh.faces)\n")
        matches = filter(anchor -> length(split(advect,anchor)) == 2,anchors)
        length(matches) == 1 || error("unrecognized baseline owned-face pass")
        advect = insert_once(advect,only(matches),counter("owned_flux"))
        append!(names,["owned_flux","gather"])
    end
    advect = insert_once(advect,
        "    _parallel_block_columns(mesh) do b, first_j, last_j\n",
        counter("gather"))
    declarations = "# Untimed copied-source per-pass participation counters.\n" *
                   "const _STAGE3_PASS_COUNTERS = Dict(\n" *
                   join(("    \"$name\" => zeros(UInt8, Threads.nthreads(:default)),\n"
                         for name in names)) * ")\n\n"
    return prefix * declarations * advect * suffix, names
end

function instrument_driver(original)
    before,_,after = split_region(original,
        "    if isdefined(TwoPhaseAMR,:_STAGE3_PROBE_FLUX)\n",
        "    scatter!(field,q0)\n    GC.gc()\n")
    replacement = "    if isdefined(TwoPhaseAMR,:_STAGE3_PASS_COUNTERS)\n" *
        "        counters = getfield(TwoPhaseAMR,:_STAGE3_PASS_COUNTERS)\n" *
        "        for seen in values(counters)\n" *
        "            fill!(seen,0)\n" *
        "        end\n" *
        "        scatter!(field,q0)\n" *
        "        step!()\n" *
        "        for (name,seen) in counters\n" *
        "            record[\"active_\" * name * \"_worker_ids\"] = findall(!iszero,seen)\n" *
        "        end\n" *
        "    end\n"
    return before * replacement * after
end

function main(args=ARGS)
    if any(arg -> arg in ("-h","--help"),args)
        println("usage: make_worker_probe.jl --source SOURCE --output NEW_OUTPUT")
        return
    end
    length(args) == 4 || error("usage: make_worker_probe.jl --source SOURCE --output NEW_OUTPUT")
    options = Dict(args[i] => args[i+1] for i in 1:2:length(args))
    source = abspath(options["--source"])
    output = abspath(options["--output"])
    !ispath(output) || error("probe output already exists")
    clean = joinpath(source,RELATIVE)
    original = read(clean,String)
    changed,names = instrument(original)
    driver_changed = instrument_driver(read(DRIVER,String))
    mkpath(output)
    cp(joinpath(source,"src"),joinpath(output,"src"))
    target = joinpath(output,RELATIVE)
    write(target,changed)
    probe_driver = joinpath(output,"probe_driver.jl")
    write(probe_driver,driver_changed)
    diff = run_capture(["diff","-u","--label",RELATIVE,"--label",RELATIVE*".probe",clean,target])
    diff.exitcode == 1 || error("expected one copied-source probe diff")
    patch = joinpath(output,"probe.patch")
    write(patch,diff.stdout)
    receipt = Dict("clean_source" => source, "clean_source_sha256" => sha256file(clean),
                   "probe_source_sha256" => sha256file(target),
                   "driver_sha256" => sha256file(probe_driver),
                   "patch_sha256" => sha256file(patch), "passes" => names,
                   "policy" => "copied source and driver only; single writer per thread slot")
    writejson(joinpath(output,"probe.json"),receipt)
    println("created probe: ",output," (",join(names,", "),")")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module

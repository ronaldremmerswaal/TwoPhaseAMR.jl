"""Create an untimed instrumented copy of the general upwind source."""
module Stage3WorkerProbe

include(joinpath(@__DIR__, "..", "Support.jl"))
using .ScalingSupport: writejson, sha256file, run_capture

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const RELATIVE = joinpath("src", "operators", "advection", "cell_advection.jl")

function replace_once(source, before, after)
    pieces = split(source, before; limit=3)
    length(pieces) == 2 || error("expected exactly one worker probe anchor")
    return pieces[1] * after * pieces[2]
end

function patch_source(original)
    start = findfirst("function advect!(\n        scalar::CellField", original)
    start === nothing && error("general cell advect! anchor missing")
    finish = findnext("\nfunction stable_timestep(", original, last(start)+1)
    finish === nothing && error("stable_timestep anchor missing")
    body = original[first(start):first(finish)-1]
    gather = "    _parallel_block_columns(mesh) do b, first_j, last_j\n"
    body = replace_once(body, gather,
        gather * "        @inbounds _STAGE3_PROBE_GATHER[Threads.threadid()] = 1\n")
    patched = original[begin:first(start)-1] * body * original[first(finish):end]
    regular = "function _cell_advection_regular_fluxes!("
    if occursin(regular, original)
        helper_start = findfirst(regular, patched)
        helper_end = findnext("function _cell_advection_regular_rates!", patched,
                              last(helper_start)+1)
        helper_end === nothing && error("regular rate helper anchor missing")
        helper = patched[first(helper_start):first(helper_end)-1]
        callback = "    _parallel_block_columns(mesh) do b, first_j, last_j\n"
        helper = replace_once(helper, callback,
            callback * "        @inbounds _STAGE3_PROBE_FLUX[Threads.threadid()] = 1\n")
        patched = patched[begin:first(helper_start)-1] * helper *
                  patched[first(helper_end):end]
        scope = "regular block-interior flux tile callback"
    else
        face_anchors = ("    @_parallel for index in 1:length(mesh.faces)\n",
                        "    @_parallel for index in eachindex(workspace.interior_faces)\n")
        matches = [anchor for anchor in face_anchors if length(split(patched, anchor)) == 2]
        length(matches) == 1 || error("unrecognized legacy owned-face flux pass")
        patched = replace_once(patched, only(matches),
            only(matches) * "        @inbounds _STAGE3_PROBE_FLUX[Threads.threadid()] = 1\n")
        scope = "owned-face flux pass"
    end
    declarations = "# Untimed Stage 3 worker probe; private to this copied source.\n" *
        "const _STAGE3_PROBE_FLUX = zeros(UInt8, Threads.nthreads(:default))\n" *
        "const _STAGE3_PROBE_GATHER = zeros(UInt8, Threads.nthreads(:default))\n\n"
    return declarations * patched, scope
end

function make_probe(source, output)
    !ispath(output) || error("probe output already exists")
    source_file = joinpath(source, RELATIVE)
    isfile(source_file) || error("advection source missing: $source_file")
    original = read(source_file, String)
    instrumented, scope = patch_source(original)
    mkpath(output)
    cp(joinpath(source, "src"), joinpath(output, "src"))
    target = joinpath(output, RELATIVE)
    write(target, instrumented)
    patch = joinpath(output, "worker-probe.patch")
    result = run_capture(["diff", "-u", "--label", RELATIVE,
                          "--label", "$RELATIVE.instrumented", source_file, target])
    result.exitcode == 1 || error("diff did not produce an instrumented patch")
    write(patch, result.stdout)
    receipt = Dict{String,Any}(
        "source"=>source, "source_sha256"=>sha256file(source_file),
        "instrumented_sha256"=>sha256file(target), "instrumented_file"=>target,
        "injected_diff"=>patch, "injected_diff_sha256"=>sha256file(patch),
        "injected_counters"=>[scope, "cell gather tile callback"],
        "policy"=>"untimed copy only; each thread writes its own UInt8 slot")
    writejson(joinpath(output, "worker-probe.json"), receipt)
    return receipt
end

function main(args=ARGS)
    source, output = ROOT, nothing
    iseven(length(args)) || error("usage: julia make_worker_probe.jl [--source DIR] --output DIR")
    for index in 1:2:length(args)
        if args[index] == "--source"
            source = abspath(args[index+1])
        elseif args[index] == "--output"
            output = abspath(args[index+1])
        else
            error("unknown option $(args[index])")
        end
    end
    output === nothing && error("--output is required")
    println(make_probe(source, output))
end

end # module

abspath(PROGRAM_FILE) == (@__FILE__) && Stage3WorkerProbe.main()

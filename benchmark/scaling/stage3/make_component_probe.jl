"""Create an untimed copied-source diagnostic of general upwind pass costs."""
module Stage3ComponentProbe

include(joinpath(@__DIR__, "..", "Support.jl"))
using .ScalingSupport: readjson, writejson, sha256file, run_capture

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const RELATIVE = joinpath("src", "operators", "advection", "cell_advection.jl")

function replace_once(source, before, after)
    pieces = split(source, before; limit=3)
    length(pieces) == 2 || error("expected exactly one component probe anchor: $before")
    return pieces[1] * after * pieces[2]
end

function patch_source(original)
    start = findfirst("function advect!(\n        scalar::CellField", original)
    start === nothing && error("general cell advect! anchor missing")
    finish = findnext("\nfunction stable_timestep(", original, last(start)+1)
    finish === nothing && error("stable_timestep anchor missing")
    preceding = original[begin:first(start)-1]
    body = original[first(start):first(finish)-1]
    following = original[first(finish):end]
    face_anchors = ("    @_parallel for index in 1:length(mesh.faces)\n",
                    "    @_parallel for index in eachindex(workspace.interior_faces)\n")
    matches = [anchor for anchor in face_anchors if length(split(body, anchor)) == 2]
    length(matches) == 1 || error("unrecognized interior-face pass")
    face_anchor = only(matches)
    boundary = "    @_parallel for index in 1:length(mesh.boundary_faces)\n"
    gather = "    _parallel_block_columns(mesh) do b, first_j, last_j\n"
    body = replace_once(body, face_anchor,
        "    _stage3_component_start = time_ns()\n" * face_anchor)
    body = replace_once(body, boundary,
        "    _STAGE3_COMPONENT_FLUX_NS[] += time_ns() - _stage3_component_start\n" *
        "    _stage3_component_start = time_ns()\n" * boundary)
    body = replace_once(body, gather,
        "    _STAGE3_COMPONENT_BOUNDARY_NS[] += time_ns() - _stage3_component_start\n" *
        "    _stage3_component_start = time_ns()\n" * gather)
    body = replace_once(body, "    end\n    return scalar\nend",
        "    end\n" *
        "    _STAGE3_COMPONENT_GATHER_NS[] += time_ns() - _stage3_component_start\n" *
        "    return scalar\nend")
    declarations = "# Stage 3 private component timing diagnostic; caller thread alone writes.\n" *
        "const _STAGE3_COMPONENT_FLUX_NS = Ref{UInt64}(0)\n" *
        "const _STAGE3_COMPONENT_BOUNDARY_NS = Ref{UInt64}(0)\n" *
        "const _STAGE3_COMPONENT_GATHER_NS = Ref{UInt64}(0)\n\n"
    return preceding * declarations * body * following
end

function patch_driver(original)
    before = "    t0 = time_ns()\n    for _ in 1:steps\n        step!()\n" *
             "    end\n    elapsed = (time_ns()-t0)/1e9\n"
    after = "    for counter in (TwoPhaseAMR._STAGE3_COMPONENT_FLUX_NS,\n" *
            "                    TwoPhaseAMR._STAGE3_COMPONENT_BOUNDARY_NS,\n" *
            "                    TwoPhaseAMR._STAGE3_COMPONENT_GATHER_NS)\n" *
            "        counter[] = 0\n" *
            "    end\n" * before *
            "    record[\"component_flux_seconds\"] = " *
            "TwoPhaseAMR._STAGE3_COMPONENT_FLUX_NS[] / 1e9\n" *
            "    record[\"component_boundary_seconds\"] = " *
            "TwoPhaseAMR._STAGE3_COMPONENT_BOUNDARY_NS[] / 1e9\n" *
            "    record[\"component_gather_seconds\"] = " *
            "TwoPhaseAMR._STAGE3_COMPONENT_GATHER_NS[] / 1e9\n"
    return replace_once(original, before, after)
end

function diff_text(source, target, label)
    command = ["diff", "-u", "--label", label, "--label", "$label.instrumented",
               source, target]
    result = run_capture(command)
    result.exitcode == 1 || error("diff did not produce an instrumented patch")
    return result.stdout
end

function make_probe(source, output)
    !ispath(output) || error("component probe output already exists")
    source_file = joinpath(source, RELATIVE)
    driver_source = joinpath(@__DIR__, "main_driver.jl")
    isfile(source_file) && isfile(driver_source) || error("source or driver missing")
    mkpath(output)
    cp(joinpath(source, "src"), joinpath(output, "src"))
    target = joinpath(output, RELATIVE)
    write(target, patch_source(read(source_file, String)))
    driver_target = joinpath(output, "component_driver.jl")
    write(driver_target, patch_driver(read(driver_source, String)))
    patch = joinpath(output, "component-probe.patch")
    write(patch, diff_text(source_file, target, RELATIVE) *
                 diff_text(driver_source, driver_target, "main_driver.jl"))
    receipt = Dict{String,Any}(
        "source"=>source, "source_sha256"=>sha256file(source_file),
        "component_source_sha256"=>sha256file(target),
        "driver_sha256"=>sha256file(driver_source),
        "component_driver_sha256"=>sha256file(driver_target),
        "patch_sha256"=>sha256file(patch),
        "passes"=>["owned-face residual and regular flux", "boundary flux",
                     "ordered cell gather/update"],
        "policy"=>"diagnostic copy only; one time_ns pair per pass per step")
    writejson(joinpath(output, "component-probe.json"), receipt)
    return receipt
end

function main(args=ARGS)
    if any(arg -> arg in ("-h", "--help"), args)
        println("usage: julia make_component_probe.jl [--source DIR] --output DIR")
        return nothing
    end
    source, output = ROOT, nothing
    iseven(length(args)) || error("usage: julia make_component_probe.jl [--source DIR] --output DIR")
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

abspath(PROGRAM_FILE) == (@__FILE__) && Stage3ComponentProbe.main()

#!/usr/bin/env julia
module RecreateNarrowIDControl

include(joinpath(@__DIR__, "study.jl"))
using .BlockInteriorStudy
const Support = BlockInteriorStudy.ScalingSupport

const ROOT = BlockInteriorStudy.ROOT
const REVISION = BlockInteriorStudy.CONTROL_REVISION
const ENV_SOURCE = joinpath(@__DIR__, "numerical_project")
const PROJECT_SHA = "01d49519e37d146088a17f8830bff96f2e941e07c1f79ea895c2df199f1f1e91"
const MANIFEST_SHA = "d2bfdb90a95191cd2e7ed74a32cc827e3413f05ef231de44edac5260e92f90d0"
const CONTROL_TREE_SHA = "962e6fb10d96e4bf513f7f1da30fa2a7e39cd4acc3abb55ccc8caa7545f89cdb"

function files_under(root)
    found = String[]
    for (directory, _, names) in walkdir(root)
        append!(found, joinpath(directory, name) for name in names)
    end
    return sort!(found)
end

function verify_recreated(output)
    receipt = Support.readjson(joinpath(output, "freeze.json"))
    receipt["schema"] == 2 && receipt["git_revision"] == REVISION &&
        receipt["numerical_project_source_strategy"] == "copy_committed_control_src" ||
        error("recreated freeze identity differs")
    actual = Set(relpath(path, output) for path in files_under(output)
                 if basename(path) != "freeze.json")
    Set(keys(receipt["files"])) == actual || error("recreated file set differs")
    for (relative, hash) in receipt["files"]
        Support.sha256file(joinpath(output, relative)) == hash ||
            error("recreated file hash differs: $relative")
    end
    control_src = joinpath(output, "src")
    project_src = joinpath(output, "benchmark-project", "src")
    control_files = Set(relpath(path, control_src) for path in files_under(control_src))
    project_files = Set(relpath(path, project_src) for path in files_under(project_src))
    control_files == project_files || error("schema-2 numerical source file set differs")
    for relative in control_files
        Support.sha256file(joinpath(control_src, relative)) ==
            Support.sha256file(joinpath(project_src, relative)) ||
            error("schema-2 numerical source copy differs: $relative")
    end
    Support.source_hash(output) == receipt["source_tree_sha256"] == CONTROL_TREE_SHA ||
        error("committed control source hash differs")
    relative = joinpath("src", "operators", "advection", "cell_advection.jl")
    Support.sha256file(joinpath(output, relative)) ==
        BlockInteriorStudy.CONTROL_ADVECTION_SHA256 || error("advection source differs")
    relative = joinpath("src", "indicator", "embedded_operators.jl")
    Support.sha256file(joinpath(output, relative)) ==
        BlockInteriorStudy.CONTROL_EMBEDDED_SHA256 || error("embedded source differs")
    for name in ("Project.toml", "Manifest.toml")
        hash = name == "Project.toml" ? PROJECT_SHA : MANIFEST_SHA
        Support.sha256file(joinpath(output, "benchmark-project", name)) == hash ||
            error("numerical $name differs")
    end
    for (path, key) in ((BlockInteriorStudy.DRIVER, "main_driver_sha256"),
                         (joinpath(ROOT, "benchmark", "scaling", "Support.jl"),
                          "support_sha256"),
                         (joinpath(ROOT, "benchmark", "scaling", "Project.toml"),
                          "tooling_project_sha256"),
                         (joinpath(ROOT, "benchmark", "scaling", "Manifest.toml"),
                          "tooling_manifest_sha256"),
                         (BlockInteriorStudy.JULIA, "numerical_julia_sha256"))
        Support.sha256file(path) == receipt[key] || error("tooling hash differs: $path")
    end
    return receipt
end

function main(args=ARGS)
    if any(arg -> arg in ("-h", "--help"), args)
        println("usage: recreate_control.jl [NEW_OUTPUT_DIRECTORY]")
        println("default: ", BlockInteriorStudy.BASELINE)
        return
    end
    length(args) <= 1 || error("usage: recreate_control.jl [NEW_OUTPUT_DIRECTORY]")
    output = isempty(args) ? BlockInteriorStudy.BASELINE : abspath(only(args))
    !ispath(output) || error("refusing to overwrite existing control: $output")
    for (name, hash) in (("Project.toml", PROJECT_SHA), ("Manifest.toml", MANIFEST_SHA))
        Support.sha256file(joinpath(ENV_SOURCE, name)) == hash ||
            error("pinned numerical $name differs")
    end
    Support.sha256file(BlockInteriorStudy.JULIA) ==
        "1c7ea3ba2e49e6c4a96ca8b80e34f984f65c1dbcdc1423f33a63f2bc14a69a5b" ||
        error("numerical Julia executable differs from accepted study")
    revision = readchomp(Cmd(["git", "-C", ROOT, "rev-parse", REVISION]))
    revision == REVISION || error("committed control revision unavailable")
    mkpath(output)
    archive = Cmd(["git", "-C", ROOT, "archive", REVISION, "src", "test"])
    extract = Cmd(["tar", "-xf", "-", "-C", output])
    run(pipeline(archive, extract))
    project = joinpath(output, "benchmark-project")
    mkpath(project)
    for name in ("Project.toml", "Manifest.toml")
        cp(joinpath(ENV_SOURCE, name), joinpath(project, name))
    end
    cp(joinpath(output, "src"), joinpath(project, "src"))
    # This environment src is a project layout copy. Drivers explicitly load
    # the source passed by `source=`, and the accepted schema-1 environment
    # contained a different historical src copy; do not imply byte identity.
    file_hashes = Dict(relpath(path, output) => Support.sha256file(path)
                       for path in files_under(output))
    receipt = Dict{String,Any}(
        "schema" => 2,
        "git_revision" => REVISION,
        "extraction" => "git archive committed src test; copy pinned Project/Manifest",
        "numerical_project_source_strategy" => "copy_committed_control_src",
        "numerical_project_source_role" => "Julia project layout; measured driver includes source= explicitly",
        "files" => file_hashes,
        "source_tree_sha256" => Support.source_hash(output),
        "main_driver_sha256" => Support.sha256file(BlockInteriorStudy.DRIVER),
        "support_sha256" => Support.sha256file(joinpath(ROOT, "benchmark", "scaling", "Support.jl")),
        "tooling_project_sha256" => Support.sha256file(joinpath(ROOT, "benchmark", "scaling", "Project.toml")),
        "tooling_manifest_sha256" => Support.sha256file(joinpath(ROOT, "benchmark", "scaling", "Manifest.toml")),
        "tooling_julia_version" => string(VERSION),
        "numerical_julia_path" => BlockInteriorStudy.JULIA,
        "numerical_julia_sha256" => Support.sha256file(BlockInteriorStudy.JULIA),
        "numerical_julia_version" => readchomp(Cmd([BlockInteriorStudy.JULIA,
                                                     "--startup-file=no", "-e", "print(VERSION)"])),
        "numerical_depot" => BlockInteriorStudy.DEPOT)
    Support.writejson(joinpath(output, "freeze.json"), receipt)
    verify_recreated(output)
    println("created and verified schema-2 committed control at $output")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module

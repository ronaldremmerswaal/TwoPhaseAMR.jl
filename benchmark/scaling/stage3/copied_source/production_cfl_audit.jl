#!/usr/bin/env julia
"""Independently audit a production-only, CFL-inclusive Stage 3 receipt."""
module ProductionCFLAudit

include(joinpath(@__DIR__, "..", "..", "Support.jl"))
using .ScalingSupport
using JSON
using TOML

const ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
const COHORT = joinpath(ROOT, "benchmark", "scaling", "results",
                    "20261010-narrow-ids-final")
const RELATIVE = joinpath("src", "operators", "advection", "cell_advection.jl")
const ADVECTION_SHA = "047457e224baa1c5b5dfa26376eff71acd61de02b225ca7e13360df6c3184541"
const TREE_SHA = "0bb8826fb52d143ae0fa5f84e043f4ed4d4feeadb4c7328d6fbda33cbf7095a2"
const WORKERS = (1, 4, 8)

equal(actual, expected, label) =
    actual == expected || error("$label differs: $actual versus $expected")
valid(condition, label) = condition || error("$label differs")

function check_files(locked)
    for (path, hash) in locked
        valid(isfile(path), "locked file missing: $path")
        equal(sha256file(path), hash, "locked $path")
    end
end

function copied_source(output)
    source = joinpath(output, "control")
    equal(source_hash(source), TREE_SHA, "copied production tree")
    equal(sha256file(joinpath(source, RELATIVE)), ADVECTION_SHA,
          "copied production advection")
    equal(source_hash(joinpath(COHORT, "current-source")), TREE_SHA,
          "frozen production tree")
    return source
end

function setup_command(output, source, driver, julia, project, n, block, p)
    return [julia, "--startup-file=no", "--project=$project", "--threads=$p",
            "--gcthreads=1", driver, "source=$source", "mode=setup",
            "kernel=upwind_cfl", "n=$n", "block=$block",
            "report=$(joinpath(output,"setup-p$p.toml"))"]
end

function audit_setup(output)
    output = abspath(output)
    gate = readjson(joinpath(output, "setup-summary.json"))
    equal(gate["schema"], 1, "setup schema")
    equal(gate["status"], "passed", "setup status")
    equal(gate["workers"], collect(WORKERS), "setup worker coverage")
    n, block = gate["case"]
    valid(n == 1024 && block in (8, 1024), "setup case")
    equal(gate["source_advection_sha256"], ADVECTION_SHA, "setup advection")
    equal(gate["source_tree_sha256"], TREE_SHA, "setup tree")
    source = copied_source(output)
    driver = joinpath(output, "checked_driver.jl")
    equal(sha256file(driver), gate["driver_sha256"], "setup driver")
    equal(sha256file(joinpath(COHORT,"placement.json")),
          gate["placement_sha256"], "setup placement")
    equal(sha256file(joinpath(output,"setup-rows.jsonl")),
          gate["setup_rows_sha256"], "setup rows")
    equal(sha256file(joinpath(output,"commands.jsonl")),
          gate["commands_sha256"], "setup commands")
    check_files(gate["locked_files_sha256"])
    rows = readjsonl(joinpath(output,"setup-rows.jsonl"))
    commands = readjsonl(joinpath(output,"commands.jsonl"))
    equal(length(rows), length(WORKERS), "setup row count")
    equal(length(commands), length(WORKERS), "setup child count")
    placement = readjson(joinpath(COHORT,"placement.json"))
    julia = commands[1]["command"][1]
    project = split(commands[1]["command"][3], '='; limit=2)[2]
    equal(sha256file(julia), gate["julia_sha256"], "setup Julia")
    equal(sha256file(joinpath(project,"Project.toml")),
          gate["project_toml_sha256"], "setup project")
    equal(sha256file(joinpath(project,"Manifest.toml")),
          gate["project_manifest_sha256"], "setup manifest")
    for (i,p) in enumerate(WORKERS)
        entry, row = commands[i], rows[i]
        equal(entry["command"], setup_command(output,source,driver,julia,
                                               project,n,block,p), "setup p$p command")
        equal(entry["cpus"], placement[string(p)], "setup p$p placement")
        equal(entry["exit_code"], 0, "setup p$p exit")
        equal(row, TOML.parsefile(joinpath(output,"setup-p$p.toml")),
              "setup p$p report copy")
        for (key,expected) in (("source",source),("mode","setup"),
                               ("kernel","upwind_cfl"),("n",n),
                               ("block",block),("threads",p),
                               ("blocks",(n÷block)^2),("faces",2n*n),
                               ("julia_version","1.10.12"))
            equal(row[key], expected, "setup p$p $key")
        end
        valid(isfinite(row["mesh_setup_seconds"]) &&
              isfinite(row["workspace_setup_seconds"]) &&
              row["mesh_setup_seconds"] >= 0 &&
              row["workspace_setup_seconds"] >= 0, "setup p$p durations")
    end
    return gate
end

function timed_command(output, source, driver, config, p, stem)
    n, block = config["case"]
    cpus = config["placement"][string(p)]
    return [config["julia"], "--startup-file=no",
            "--project=$(config["project"])", "--threads=$p", "--gcthreads=1",
            driver, "source=$source", "mode=measure_cfl",
            "kernel=upwind_cfl", "n=$n", "block=$block",
            "steps=$(config["steps"])", "cpus=$(join(cpus,','))",
            "input=$(joinpath(output,"initial.bin"))",
            "output=$(joinpath(output,"$stem.bin"))",
            "report=$(joinpath(output,"$stem.toml"))"]
end

function audit_measure(output)
    output = abspath(output)
    config = readjson(joinpath(output,"configuration.json"))
    equal(config["schema"], 1, "measure schema")
    equal(config["repeats"], 5, "measure repetitions")
    equal(config["workload"], "cfl", "measure workload")
    equal((config["mode"],config["kernel"]),
          ("measure_cfl","upwind_cfl"), "measure mode/kernel")
    equal(config["source_advection_sha256"], ADVECTION_SHA, "measure advection")
    equal(config["source_tree_sha256"], TREE_SHA, "measure source tree")
    setup = audit_setup(config["setup_gate"])
    equal(setup["case"], config["case"], "setup/measure case")
    equal(sha256file(joinpath(config["setup_gate"],"setup-summary.json")),
          config["setup_gate_sha256"], "setup gate receipt")
    source = copied_source(output)
    check_files(config["locked_files_sha256"])
    equal(config["placement"], readjson(joinpath(COHORT,"placement.json")),
          "measure placement")
    n, block = config["case"]
    valid(n == 1024 && block in (8,1024), "measure case")
    valid(config["steps"] > 0, "measure steps")
    equal(sha256file(joinpath(output,"initial.bin")),
          sha256file(joinpath(COHORT,"initial-n$n.bin")), "frozen input")
    driver = joinpath(output,"checked_driver.jl")
    oracle = joinpath(output,"oracle.bin")
    equal(sha256file(oracle), config["oracle_sha256"], "oracle file")
    oracle_report = TOML.parsefile(joinpath(output,"oracle.toml"))
    equal(oracle_report["field_sha256"], config["oracle_sha256"],
          "oracle report field")
    commands = readjsonl(joinpath(output,"commands.jsonl"))
    jobs = readjson(joinpath(output,"job_order.json"))
    rows = readjsonl(joinpath(output,"raw.jsonl"))
    equal(length(commands), 16, "measure child count")
    equal(length(jobs), 15, "measure job count")
    equal(length(rows), 15, "measure row count")
    equal(Set((job["workers"],job["repeat"]) for job in jobs),
          Set((p,r) for p in WORKERS for r in 0:4), "measure job coverage")
    equal(commands[1]["command"], timed_command(output,source,driver,config,
                                                  1,"oracle"), "oracle command")
    equal(commands[1]["exit_code"], 0, "oracle exit")
    equal(commands[1]["cpus"], config["placement"]["1"], "oracle placement")
    equal(oracle_report["source"], source, "oracle source")
    for (key,expected) in (("n",n),("block",block),
                           ("steps",config["steps"]),("threads",1),
                           ("mode","measure_cfl"),("kernel","upwind_cfl"))
        equal(oracle_report[key], expected, "oracle $key")
    end
    equal(oracle_report["cfl_scans_per_timed_step"], 1, "oracle CFL scans")
    equal(oracle_report["dt"], 0.4oracle_report["cfl_bound"], "oracle CFL dt")
    for (i,job) in enumerate(jobs)
        p,r = job["workers"],job["repeat"]
        stem = "control-n$n-b$block-p$p-r$r"
        command,row = commands[i+1],rows[i]
        equal(command["command"], timed_command(output,source,driver,config,
                                                 p,stem), "$stem command")
        equal(command["exit_code"], 0, "$stem exit")
        equal(command["cpus"], config["placement"][string(p)],
              "$stem placement")
        for (key,expected) in (("source",source),("variant","control"),
                               ("threads",p),("repeat",r),("n",n),
                               ("block",block),("steps",config["steps"]),
                               ("mode","measure_cfl"),("kernel","upwind_cfl"),
                               ("blocks",(n÷block)^2),("faces",2n*n))
            equal(row[key],expected,"$stem $key")
        end
        report = TOML.parsefile(joinpath(output,"$stem.toml"))
        for (key,value) in report
            equal(row[key], value, "$stem report $key")
        end
        equal(row["source_sha256"], ADVECTION_SHA, "$stem advection")
        equal(row["initial_sha256"], sha256file(joinpath(output,"initial.bin")),
              "$stem input")
        equal(row["oracle_sha256"], config["oracle_sha256"], "$stem oracle")
        equal(row["field_sha256"], config["oracle_sha256"], "$stem field")
        valid(row["bitwise_equal"] && row["tolerance_violations"] == 0 &&
              row["max_abs_vs_oracle"] == 0, "$stem full field")
        equal(row["cfl_scans_per_timed_step"], 1, "$stem CFL scans")
        equal(row["cfl_bound"], oracle_report["cfl_bound"], "$stem CFL bound")
        equal(row["dt"], 0.4row["cfl_bound"], "$stem CFL dt")
        equal(row["carrier_policy"], "prescribed constant owned-face velocity",
              "$stem carrier")
        equal(row["dt_policy"],
              "0.4 * stable_timestep(velocity, workspace; cfl=0.9) each step",
              "$stem dt policy")
        valid(row["time_seconds"] >= config["minimum_seconds"] &&
              isfinite(row["ns_per_unknown_step"]) &&
              isapprox(row["ns_per_unknown_step"],
                       1e9row["time_seconds"]/(n*n*config["steps"]);rtol=1e-12),
              "$stem time normalization")
        equal([a["cpu"] for a in row["affinity"]],
              config["placement"][string(p)], "$stem CPUs")
        valid(all(a["cpu"] == a["observed_cpu"] for a in row["affinity"]),
              "$stem observed CPUs")
    end
    summary = readjson(joinpath(output,"summary.json"))
    equal((summary["case"],summary["steps"],summary["repeats"],
           summary["workload"],summary["samples"]),
          (config["case"],config["steps"],5,"cfl",15), "measure summary")
    equal(summary["minimum_seconds"], config["minimum_seconds"],
          "measure interval floor")
    samples = Dict(p => [only(row["ns_per_unknown_step"] for row in rows if
                        row["threads"] == p && row["repeat"] == r) for r in 0:4]
                   for p in WORKERS)
    serial = median_value(samples[1])
    results = Any[]
    for p in WORKERS
        group = only(g for g in summary["groups"] if g["workers"] == p)
        equal(group["samples"], samples[p], "p$p samples")
        median = median_value(samples[p])
        equal(group["median_ns_per_unknown_complete_step"], median,
              "p$p median")
        efficiency = serial/(p*median)
        equal(group["parallel_efficiency"], efficiency, "p$p efficiency")
        push!(results,Dict("workers"=>p,
                           "median_ns_per_cell_complete_step"=>median,
                           "efficiency"=>efficiency,
                           "efficiency_95_interval"=>p == 1 ? nothing :
                               bootstrap_ratio(samples[1],samples[p],p;
                                               seed=20261015+p)))
    end
    return Dict("status"=>"passed","phase"=>"measure","case"=>config["case"],
                "steps"=>config["steps"],"children"=>16,"rows"=>15,
                "results"=>results,"output"=>output)
end

function main(args=ARGS)
    length(args) == 2 && args[1] in ("setup","measure") ||
        error("usage: production_cfl_audit.jl setup|measure RECEIPT_DIR")
    result = args[1] == "setup" ?
        Dict("status"=>"passed","phase"=>"setup","case"=>
             audit_setup(args[2])["case"],"output"=>abspath(args[2])) :
        audit_measure(args[2])
    println(JSON.json(result, 2))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end

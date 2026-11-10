# Runs every check of the audit of the five-row catalogue spot check of Chapter 14 (slug sec-verif-catalogue-spotcheck) and exits non-zero if any fails.
#
#   julia --project=@claude-symbolics proof_review/sec-verif-catalogue-spotcheck/scripts/run_all.jl
#
# Audit of five random rows of the catalogue tables of docs/techref/chapters/14_verification.tex against the cited tests (one is rerun).
# The checks need the TwoPhaseAMR implementation, so they run as sequential child processes under the package environment
# (one Julia process busy at a time; this driver itself loads no packages):
#
#   1. code_01_capillary_row.jl
# Flags for the shared machine: --heap-size-hint=3G -t 1, OPENBLAS_NUM_THREADS=1, JULIA_NUM_PRECOMPILE_TASKS=1.

const SCRIPTS = @__DIR__
const ROOT = normpath(joinpath(SCRIPTS, "..", "..", ".."))
const CODE_PROJECT = isfile(joinpath(ROOT, "Manifest.toml")) ? ROOT : joinpath(ROOT, "docs", "techref", "figures")

function run_group(name, project, script)
    cmd = Cmd(
        `$(Base.julia_cmd()) --heap-size-hint=3G -t 1 --project=$project $(joinpath(SCRIPTS, script))`;
        dir = ROOT,
    )
    cmd = addenv(cmd, "OPENBLAS_NUM_THREADS" => "1", "JULIA_NUM_PRECOMPILE_TASKS" => "1")
    println("\n=== ", name, " ===")
    ok = success(pipeline(cmd; stdout = stdout, stderr = stderr))
    println("=== ", name, ": ", ok ? "PASS" : "FAIL", " ===")
    return ok
end

results = [
    run_group("code_01_capillary_row.jl", CODE_PROJECT, "code_01_capillary_row.jl"),
]
println("\nrun_all: ", all(results) ? "all checks passed" : "FAILURES")
exit(all(results) ? 0 : 1)

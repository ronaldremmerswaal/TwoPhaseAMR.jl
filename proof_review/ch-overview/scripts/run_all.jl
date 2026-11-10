# Runs every check of the audit of ch:overview (stage lists, run sequence, options table) and exits non-zero if any fails.
#
#   julia --project=docs/techref/figures proof_review/ch-overview/scripts/run_all.jl
#
# Each group is a child process (one Julia process busy at a time; this driver loads no packages). Every check script loads an
# instrumented copy of src/ (see _trace.jl) next to the package of the active project, so the project must provide TwoPhaseAMR and its
# dependencies: docs/techref/figures does (its [sources] path is the repository root). Flags for the shared machine: --heap-size-hint=3G -t 1,
# OPENBLAS_NUM_THREADS=1, JULIA_NUM_PRECOMPILE_TASKS=1. About 3 x 1-4 minutes (compilation dominates).

const SCRIPTS = @__DIR__
const ROOT = normpath(joinpath(SCRIPTS, "..", "..", ".."))
const PROJECT = joinpath(ROOT, "docs", "techref", "figures")

function run_group(name, script)
    cmd = Cmd(
        `$(Base.julia_cmd()) --heap-size-hint=3G -t 1 --project=$PROJECT $(joinpath(SCRIPTS, script))`;
        dir = ROOT,
    )
    cmd = addenv(cmd, "OPENBLAS_NUM_THREADS" => "1", "JULIA_NUM_PRECOMPILE_TASKS" => "1")
    println("\n=== ", name, " ===")
    ok = success(pipeline(cmd; stdout = stdout, stderr = stderr))
    println("=== ", name, ": ", ok ? "PASS" : "FAIL", " ===")
    return ok
end

results = [
    run_group("one step: stage lists (sec:overview:step-one, step-two)", "step01_stage_order.jl"),
    run_group("a run: simulate! sequence, observer events, stop polling (sec:overview:run)", "step02_run_sequence.jl"),
    run_group("options table defaults and solver choices (tab:overview:options)", "step03_options_defaults.jl"),
]
println("\nrun_all: ", all(results) ? "all checks passed" : "FAILURES")
exit(all(results) ? 0 : 1)

# Runs every check of the audit of eq:interface:contact-row and exits non-zero if any fails.
#
#   julia --project=@claude-symbolics proof_review/eq-interface-contact-row/scripts/run_all.jl
#
# The two groups need different environments, so they run as two sequential child processes
# (one Julia process busy at a time; this driver itself loads no packages):
#   1. run_symbolic.jl -- Symbolics.jl                      (--project=@claude-symbolics)
#   2. run_code.jl     -- the TwoPhaseAMR implementation    (--project=<repository root>)
# Flags for the shared machine: --heap-size-hint=3G -t 1, OPENBLAS_NUM_THREADS=1,
# JULIA_NUM_PRECOMPILE_TASKS=1.

const SCRIPTS = @__DIR__
const ROOT = normpath(joinpath(SCRIPTS, "..", "..", ".."))

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
    run_group("symbolic checks", "@claude-symbolics", "run_symbolic.jl"),
    run_group("TwoPhaseAMR implementation", ROOT, "run_code.jl"),
]
println("\nrun_all: ", all(results) ? "all checks passed" : "FAILURES")
exit(all(results) ? 0 : 1)

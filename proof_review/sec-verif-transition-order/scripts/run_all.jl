# Runs every check of the audit of the order-1/2 argument of Chapter 14 (slug sec-verif-transition-order; no label) and exits non-zero if any fails.
#
#   julia --project=@claude-symbolics proof_review/sec-verif-transition-order/scripts/run_all.jl
#
# Audit of the closing paragraph of sec:verif:orders (docs/techref/chapters/14_verification.tex:473-483).
# The checks need the TwoPhaseAMR implementation, so they run as sequential child processes under the package environment
# (one Julia process busy at a time; this driver itself loads no packages):
#
#   1. code_01_difference_scaling.jl
#   2. code_02_truncation_vs_solution.jl
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
    run_group("code_01_difference_scaling.jl", CODE_PROJECT, "code_01_difference_scaling.jl"),
    run_group("code_02_truncation_vs_solution.jl", CODE_PROJECT, "code_02_truncation_vs_solution.jl"),
]
println("\nrun_all: ", all(results) ? "all checks passed" : "FAILURES")
exit(all(results) ? 0 : 1)

# Runs every check of the audit of eq:viscosity:stiffness and exits non-zero if any fails.
#
#   julia --project=@claude-symbolics proof_review/eq-viscosity-stiffness/scripts/run_all.jl
#
# The driver loads no packages.  Each script runs in its own child process (one Julia process at a time) in the environment of
# docs/techref/figures (TwoPhaseAMR as a path dependency; the scripts use only the package and the standard library), with the flags
# --heap-size-hint=3G -t 1, OPENBLAS_NUM_THREADS=1, JULIA_NUM_PRECOMPILE_TASKS=1.  Scripts of other folders are listed where they carry
# checks of this label (marked with their folder).

const SCRIPTS = @__DIR__
const ROOT = normpath(joinpath(SCRIPTS, "..", "..", ".."))
const PROJECT = joinpath(ROOT, "docs", "techref", "figures")
const LIST = ["code_01_stiffness.jl", "../../eq-viscosity-gradient/scripts/code_01_gradient_stiffness.jl"]

function run_script(script)
    path = normpath(joinpath(SCRIPTS, script))
    cmd = Cmd(`$(Base.julia_cmd()) --heap-size-hint=3G -t 1 --project=$PROJECT $path`; dir = ROOT)
    cmd = addenv(cmd, "OPENBLAS_NUM_THREADS" => "1", "JULIA_NUM_PRECOMPILE_TASKS" => "1")
    println("\n=== ", script, " ===")
    ok = success(pipeline(cmd; stdout = stdout, stderr = stderr))
    println("=== ", script, ": ", ok ? "PASS" : "FAIL", " ===")
    return ok
end

results = [run_script(script) for script in LIST]
println("\nrun_all (eq:viscosity:stiffness): ", all(results) ? "all checks passed" : "FAILURES")
exit(all(results) ? 0 : 1)

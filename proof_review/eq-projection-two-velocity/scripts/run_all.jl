# Runs every check of the audit of eq:projection:face-data (folder eq-projection-two-velocity) and exits non-zero if any fails.
#
#   julia --project=@claude-symbolics proof_review/eq-projection-two-velocity/scripts/run_all.jl
#
# The groups need different environments, so they run as sequential child processes (one Julia process busy at a time; this driver loads no packages):
#   code   -- the TwoPhaseAMR implementation   (--project=<repository root>, or docs/techref/figures in a working tree without a root Manifest.toml)
#   sym    -- symbolic checks                  (--project=@claude-symbolics)
# Flags for the shared machine: --heap-size-hint=3G -t 1, OPENBLAS_NUM_THREADS=1, JULIA_NUM_PRECOMPILE_TASKS=1.

const SCRIPTS = @__DIR__
const ROOT = normpath(joinpath(SCRIPTS, "..", "..", ".."))
const CODE_PROJECT = isfile(joinpath(ROOT, "Manifest.toml")) ? ROOT : joinpath(ROOT, "docs", "techref", "figures")

function run_script(project, script)
    cmd = Cmd(`$(Base.julia_cmd()) --heap-size-hint=3G -t 1 --project=$project $(joinpath(SCRIPTS, script))`; dir = ROOT)
    cmd = addenv(cmd, "OPENBLAS_NUM_THREADS" => "1", "JULIA_NUM_PRECOMPILE_TASKS" => "1")
    println("\n=== ", script, " ===")
    ok = success(pipeline(cmd; stdout = stdout, stderr = stderr))
    println("=== ", script, ": ", ok ? "PASS" : "FAIL", " ===")
    return ok
end

results = Bool[]
push!(results, run_script("@claude-symbolics", "sym_01_phase_algebra.jl"))
push!(results, run_script(CODE_PROJECT, "code_01_data_flux_gauge.jl"))
println("\nrun_all (eq-projection-two-velocity): ", all(results) ? "all checks passed" : "FAILURES")
exit(all(results) ? 0 : 1)

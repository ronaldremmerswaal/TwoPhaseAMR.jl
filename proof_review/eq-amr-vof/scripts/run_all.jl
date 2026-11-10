# Runs every check of the audit of eq:amr:vof, eq:amr:scalar (folder eq-amr-vof) and exits non-zero if any fails.
#
#   julia --project=@claude-symbolics proof_review/eq-amr-vof/scripts/run_all.jl
#
# The groups need different environments, so they run as sequential child processes (one Julia process busy at a time; this driver loads no packages):
#   code   -- the TwoPhaseAMR implementation   (--project=docs/techref/figures: the worktree has no root Manifest.toml; it resolves TwoPhaseAMR to the worktree's src/)
#   sym    -- symbolic checks                  (--project=@claude-symbolics)
# Flags for the shared machine: --heap-size-hint=3G -t 1, OPENBLAS_NUM_THREADS=1, JULIA_NUM_PRECOMPILE_TASKS=1.

const SCRIPTS = @__DIR__
const ROOT = normpath(joinpath(SCRIPTS, "..", "..", ".."))
const CODE_PROJECT = isfile(joinpath(ROOT, "Manifest.toml")) ? ROOT : joinpath(ROOT, "docs", "techref", "figures")

function run_script(project, path)
    cmd = Cmd(`$(Base.julia_cmd()) --heap-size-hint=3G -t 1 --project=$project $path`; dir = ROOT)
    cmd = addenv(cmd, "OPENBLAS_NUM_THREADS" => "1", "JULIA_NUM_PRECOMPILE_TASKS" => "1")
    println("\n=== ", relpath(path, ROOT), " ===")
    ok = success(pipeline(cmd; stdout = stdout, stderr = stderr))
    println("=== ", relpath(path, ROOT), ": ", ok ? "PASS" : "FAIL", " ===")
    return ok
end

results = Bool[]
push!(results, run_script(CODE_PROJECT, joinpath(SCRIPTS, "code_01_vof_scalar.jl")))
println("\nrun_all (eq-amr-vof): ", all(results) ? "all checks passed" : "FAILURES")
exit(all(results) ? 0 : 1)

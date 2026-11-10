# Runs every check of the audit of ch:solvers (Chapter 11, operators and linear solvers) and exits non-zero if any fails.
#
#   julia --project=docs/techref/figures proof_review/ch-solvers/scripts/run_all.jl
#
# No Symbolics.jl is needed: the claims are matrix/solver statements checked on the package (dense spectra, Lanczos, BigFloat
# evaluation) and in pure LinearAlgebra (step02a, step05). Each script runs in its own child process, one after the other (one Julia
# process busy at a time); this driver loads no packages. Flags for the shared machine: --heap-size-hint=3G -t 1,
# OPENBLAS_NUM_THREADS=1, JULIA_NUM_PRECOMPILE_TASKS=1. Wall time about 6 minutes (step02b dominates: dense V-cycle matrices up to 7552 cells).

const SCRIPTS = @__DIR__
const ROOT = normpath(joinpath(SCRIPTS, "..", "..", ".."))
const PROJECT = isfile(joinpath(ROOT, "Manifest.toml")) ? ROOT : joinpath(ROOT, "docs", "techref", "figures")

const GROUPS = [
    ("claim 1  null space (eq:solvers:null)", "step01_null_space.jl"),
    ("claim 2a V-cycle SPD without Galerkin: algebra", "step02a_vcycle_algebra.jl"),
    ("claim 2b V-cycle SPD: package, large / AMR / cut-cell cases", "step02b_vcycle_code.jl"),
    ("claim 2c V-cycle: pre/post sweeps", "step02c_sweep_options.jl"),
    ("claim 3  smoother bound, Gershgorin", "step03_smoother_bound.jl"),
    ("claim 4  GMRES stop, round-off floor", "step04_gmres_and_floor.jl"),
    ("claim 5  CMDGFM scaling", "step05_cmdgfm_scaling.jl"),
    ("claim 6  restricted transfer null vector", "step06_transfer_null_vector.jl"),
    ("claim 7  CG bound vs measured factors", "step07_cg_bound.jl"),
    ("side     one-phase viscous symmetry", "step08_viscous_symmetry.jl"),
]

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

results = [run_group(name, script) for (name, script) in GROUPS]
println("\nrun_all: ", all(results) ? "all checks passed" : "FAILURES")
exit(all(results) ? 0 : 1)

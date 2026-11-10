# Runs every check of the audit of eq:verif:energy, eq:verif:energy-two, eq:verif:energies, eq:verif:divergence-norm (Chapter 14 diagnostics) and exits non-zero if any fails.
#
#   julia --project=@claude-symbolics proof_review/eq-verif-diagnostics/scripts/run_all.jl
#
# Audit of the diagnostics of docs/techref/chapters/14_verification.tex:23-117 against src/simulation/diagnostics/interface.jl.
# The checks need the TwoPhaseAMR implementation, so they run as sequential child processes under the package environment
# (one Julia process busy at a time; this driver itself loads no packages):
#
#   1. code_01_kinetic_momentum.jl
#   2. code_02_potential_surface_volume.jl
#   3. code_03_divergence_norm.jl
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
    run_group("code_01_kinetic_momentum.jl", CODE_PROJECT, "code_01_kinetic_momentum.jl"),
    run_group("code_02_potential_surface_volume.jl", CODE_PROJECT, "code_02_potential_surface_volume.jl"),
    run_group("code_03_divergence_norm.jl", CODE_PROJECT, "code_03_divergence_norm.jl"),
]
println("\nrun_all: ", all(results) ? "all checks passed" : "FAILURES")
exit(all(results) ? 0 : 1)

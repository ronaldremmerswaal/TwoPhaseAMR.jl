using BenchmarkTools
using TwoPhaseAMR

function benchmark_case(mesh)
    scalar = CellField(mesh)
    initialize!(scalar, (x, y) -> 1 + 0.1sinpi(2x) * cospi(2y))
    velocity = FaceVelocity(mesh, (x, y, time, axis) -> axis == 1 ? 0.2 : -0.1)
    workspace = AdvectionWorkspace(scalar)
    dt = stable_timestep(velocity, workspace; cfl=0.5)
    face = FaceField{1}(mesh, (x, y) -> 1 + 0.1sinpi(2x) * cospi(2y))
    staggered_workspace = StaggeredAdvectionWorkspace(face)
    staggered_dt = stable_timestep(velocity, staggered_workspace; cfl=0.5)
    transported_velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ?
            1 + 0.1sinpi(2x) * cospi(2y) :
            -0.5 + 0.1cospi(2x) * sinpi(2y),
    )
    velocity_workspace = StaggeredAdvectionWorkspace(transported_velocity)
    pressure_gradient = copy(velocity)
    gradient_operator = GradientOperator(scalar)

    setup = @benchmark BlockTreeMesh(
        block_size=$(mesh.block_size), max_level=$(mesh.max_level),
        refine=(key, origin, extent) -> key.level < $(mesh.max_level),
    ) samples=20 evals=1
    staggered_setup = @benchmark StaggeredAdvectionWorkspace($face) samples=50 evals=1
    gradient_setup = @benchmark GradientOperator($scalar) samples=20 evals=1
    step = @benchmark advect!($scalar, $velocity, $dt, $workspace)
    staggered_step = @benchmark advect!(
        $face, $velocity, $staggered_dt, $staggered_workspace)
    velocity_step = @benchmark advect!(
        $transported_velocity, $velocity, $staggered_dt, $velocity_workspace)
    gradient_step = @benchmark gradient!(
        $pressure_gradient, $scalar, $gradient_operator)
    halo = @benchmark fill_halo!($scalar)
    return (; setup, staggered_setup, gradient_setup, step, staggered_step,
            velocity_step, gradient_step, halo)
end

println("Julia ", VERSION, ", threads=", Threads.nthreads())
println("CPU: ", Sys.cpu_info()[1].model)

function print_estimate(label, trial)
    estimate = median(trial)
    println("  ", rpad(label * ":", 23), estimate.time, " ns, ",
            estimate.memory, " bytes, ", estimate.allocs, " allocations")
end

for (name, mesh) in (
    "single block" => BlockTreeMesh(block_size=(32, 32)),
    "16 uniform blocks" => BlockTreeMesh(
        block_size=(16, 16), max_level=2,
        refine=(key, origin, extent) -> true,
    ),
)
    result = benchmark_case(mesh)
    println("\n", name, " (", ncells(mesh), " cells)")
    print_estimate("setup", result.setup)
    print_estimate("staggered setup", result.staggered_setup)
    print_estimate("gradient setup", result.gradient_setup)
    print_estimate("cell step", result.step)
    print_estimate("one face component", result.staggered_step)
    print_estimate("two face components", result.velocity_step)
    print_estimate("pressure gradient", result.gradient_step)
    print_estimate("halo", result.halo)
end

mixed_mesh = BlockTreeMesh(
    block_size=(16, 16), max_level=2,
    refine=(key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
)
mixed_pressure = CellField(mixed_mesh)
initialize!(mixed_pressure, (x, y) -> sinpi(2x) * cospi(2y))
mixed_gradient = FaceVelocity(mixed_mesh, (x, y, time, axis) -> 0.0)
mixed_operator = GradientOperator(mixed_pressure)
mixed_velocity = FaceVelocity(
    mixed_mesh, (x, y, time, axis) -> axis == 1 ? 0.2 : -0.1)
mixed_face = FaceField{1}(
    mixed_mesh, (x, y) -> 1 + 0.1sinpi(2x) * cospi(2y))
mixed_staggered_workspace = StaggeredAdvectionWorkspace(mixed_face)
mixed_staggered_dt = stable_timestep(
    mixed_velocity, mixed_staggered_workspace; cfl=0.5)
mixed_transported_velocity = FaceVelocity(
    mixed_mesh,
    (x, y, time, axis) -> axis == 1 ?
        1 + 0.1sinpi(2x) * cospi(2y) :
        -0.5 + 0.1cospi(2x) * sinpi(2y),
)
mixed_velocity_workspace = StaggeredAdvectionWorkspace(
    mixed_transported_velocity)
mixed_setup = @benchmark GradientOperator($mixed_pressure) samples=20 evals=1
mixed_step = @benchmark gradient!(
    $mixed_gradient, $mixed_pressure, $mixed_operator)
mixed_two_point_staggered_setup = @benchmark TwoPhaseAMR._staggered_advection_workspace(
    $mixed_face, false) samples=20 evals=1
mixed_three_point_staggered_setup = @benchmark StaggeredAdvectionWorkspace(
    $mixed_face) samples=20 evals=1
mixed_staggered_step = @benchmark advect!(
    $mixed_face, $mixed_velocity, $mixed_staggered_dt,
    $mixed_staggered_workspace; scheme=Upwind())
mixed_limited_staggered_step = @benchmark advect!(
    $mixed_face, $mixed_velocity, $mixed_staggered_dt,
    $mixed_staggered_workspace; scheme=LW(), amr_interface_scheme=MC())
mixed_velocity_step = @benchmark advect!(
    $mixed_transported_velocity, $mixed_velocity, $mixed_staggered_dt,
    $mixed_velocity_workspace; scheme=Upwind())
mixed_limited_velocity_step = @benchmark advect!(
    $mixed_transported_velocity, $mixed_velocity, $mixed_staggered_dt,
    $mixed_velocity_workspace; scheme=LW(), amr_interface_scheme=MC())
println("\nmixed-level operators (", ncells(mixed_mesh), " cells)")
print_estimate("staggered setup, 2pt", mixed_two_point_staggered_setup)
print_estimate("staggered setup, 3pt", mixed_three_point_staggered_setup)
print_estimate("one face, upwind", mixed_staggered_step)
print_estimate("one face, LW/MC", mixed_limited_staggered_step)
print_estimate("two faces, upwind", mixed_velocity_step)
print_estimate("two faces, LW/MC", mixed_limited_velocity_step)
print_estimate("gradient setup", mixed_setup)
print_estimate("pressure gradient", mixed_step)

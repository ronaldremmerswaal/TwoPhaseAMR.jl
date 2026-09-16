using TwoPhaseAMR
using LinearAlgebra: mul!
using Statistics: median

const BENCHMARK_SAMPLES = parse(Int, get(ENV, "TWOPHASE_KERNEL_SAMPLES", "5"))
const BENCHMARK_CASE = get(ENV, "TWOPHASE_KERNEL_CASE", "all")

function uniform_mesh()
    return BlockTreeMesh(
        block_size = (8, 8), max_level = 4, periodic = (false, false),
        refine = (key, origin, extent) -> key.level < 4,
    )
end

function mixed_mesh()
    return BlockTreeMesh(
        block_size = (8, 8), max_level = 5, periodic = (false, false),
        refine = (key, origin, extent) -> key.level < 2 || (
            key.level < 5 &&
            hypot(
                origin[1] + extent[1] / 2 - 0.5,
                origin[2] + extent[2] / 2 - 0.5,
            ) < 0.28 + maximum(extent)
        ),
    )
end

function field_sum(field::CellField)
    result = zero(eltype(field))
    for block in eachindex(field.blocks)
        result += sum(interior(field, block))
    end
    return result
end

face_sum(field::FaceVelocity) = sum(field.interior) + sum(field.boundary)
finite_field_sum(field::CellField) = sum(
    isfinite(value) ? value : zero(value)
    for block in eachindex(field.blocks) for value in interior(field, block)
)
finite_matrix_sum(matrices) = sum(
    isfinite(value) ? value : zero(value)
    for matrix in matrices for value in matrix
)

function measure!(
        rows, case_name, kernel, operation, prepare, checksum;
        samples = BENCHMARK_SAMPLES,
    )
    print(
        "[", Threads.nthreads(:default), " threads] ", case_name, "/",
        kernel, " ... ",
    )
    prepare()
    operation() # compilation and lazy workspace initialization
    times = Vector{Float64}(undef, samples)
    for sample in 1:samples
        prepare()
        GC.gc(false)
        start = time_ns()
        operation()
        times[sample] = time_ns() - start
    end
    prepare()
    allocations = @allocations operation()
    prepare()
    bytes = @allocated operation()
    value = Float64(checksum())
    elapsed = median(times)
    println(round(elapsed / 1.0e6; digits = 3), " ms")
    push!(
        rows,
        (
            case = case_name, kernel, threads = Threads.nthreads(:default),
            time_ns = elapsed, bytes, allocations, checksum = value,
        ),
    )
    return nothing
end

function transfer_benchmarks!(rows, case_name)
    halo = 1
    for n in (8, 128)
        parent = zeros(n + 2halo, n + 2halo)
        for j in 1:n, i in 1:n
            parent[i + halo, j + halo] =
                1 + 0.1sinpi(2i / n) * cospi(2j / n)
        end
        children = ntuple(_ -> similar(parent), 4)
        parent_output = similar(parent)
        measure!(
            rows, case_name, "prolong_block_$n",
            () -> prolong!(children, parent, halo),
            () -> foreach(child -> fill!(child, 0.0), children),
            () -> sum(sum, children),
        )
        prolong!(children, parent, halo)
        measure!(
            rows, case_name, "restrict_block_$n",
            () -> restrict!(parent_output, children, halo),
            () -> fill!(parent_output, 0.0),
            () -> sum(parent_output),
        )
    end
    return nothing
end

function adaptation_case()
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 4, periodic = (false, false),
        adaptation = MeshAdaptationOptions(coarsen_delay = 0),
        refine = (key, origin, extent) -> key.level < 3,
    )
    levelset = (x, y) -> hypot(x - 0.5, y - 0.52) - 0.21
    interface = InterfaceField(mesh, levelset; lipschitz = 1, atol = 1.0e-8)
    scalar = CellField(mesh, (x, y) -> 1 + 0.1sinpi(2x) * cospi(2y))
    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ?
            sinpi(2x)^2 * sinpi(2y) :
            -sinpi(2x) * sinpi(2y)^2,
    )
    workspace = AMRWorkspace(mesh, interface, scalar, velocity)
    labels = fill(:refine, nblocks(mesh))
    return workspace, labels, interface, scalar, velocity
end

function refined_adaptation_case()
    workspace, labels, interface, scalar, velocity = adaptation_case()
    adapt!(workspace, labels; step = 0, force = true)
    return (
        workspace, fill(:coarsen, nblocks(workspace.mesh)),
        interface, scalar, velocity,
    )
end

function adaptation_benchmarks!(rows, case_name)
    state = Ref{Any}()
    prepare_refine() = (state[] = adaptation_case())
    measure!(
        rows, case_name, "mesh_adapt_refine_with_fields",
        () -> adapt!(state[][1], state[][2]; step = 0, force = true),
        prepare_refine,
        () -> nblocks(state[][1].mesh) + field_sum(state[][4]),
    )
    prepare_coarsen() = (state[] = refined_adaptation_case())
    measure!(
        rows, case_name, "mesh_adapt_coarsen_with_fields",
        () -> adapt!(state[][1], state[][2]; step = 1, force = true),
        prepare_coarsen,
        () -> nblocks(state[][1].mesh) + field_sum(state[][4]),
    )
    return nothing
end

function benchmark_case!(rows, case_name, mesh_factory)
    mesh_ref = Ref{Any}()
    measure!(
        rows, case_name, "mesh_construction",
        () -> (mesh_ref[] = mesh_factory()), () -> nothing,
        () -> nblocks(mesh_ref[]) + length(mesh_ref[].faces),
    )
    mesh = mesh_factory()
    println(
        case_name, ": ", nblocks(mesh), " blocks, ", ncells(mesh),
        " cells, ", length(mesh.faces), " interior faces",
    )

    levelset = (x, y) -> hypot(x - 0.5, y - 0.52) - 0.21
    scalar = CellField(mesh)
    measure!(
        rows, case_name, "cell_initialization",
        () -> initialize!(scalar, (x, y) -> 1 + 0.1sinpi(2x) * cospi(2y)),
        () -> nothing, () -> field_sum(scalar),
    )

    interface = InterfaceField(mesh; init = 0.0)
    measure!(
        rows, case_name, "interface_levelset_initialization",
        () -> initialize!(
            interface, levelset; lipschitz = 1, atol = 1.0e-8,
        ),
        () -> nothing, () -> field_sum(interface.fraction),
    )
    interface_template = InterfaceField(
        mesh, levelset; lipschitz = 1, atol = 1.0e-8,
    )
    measure!(
        rows, case_name, "interface_reconstruction",
        () -> reconstruct!(interface),
        () -> begin
            for block in eachindex(interface.fraction.blocks)
                copyto!(
                    interface.fraction.blocks[block],
                    interface_template.fraction.blocks[block],
                )
            end
        end,
        () -> finite_field_sum(interface.curvature),
    )
    measure!(
        rows, case_name, "height_function_sampling",
        () -> TwoPhaseAMR._fill_height_values!(interface),
        () -> nothing,
        () -> finite_matrix_sum(interface.height_cache.values),
    )

    scalar_template = CellField(
        mesh, (x, y) -> 1 + 0.1sinpi(2x) * cospi(2y),
    )
    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> 0.15 * (
            axis == 1 ? sinpi(2x)^2 * sinpi(2y) :
            -sinpi(2x) * sinpi(2y)^2
        ),
    )
    advection_workspace = AdvectionWorkspace(scalar)
    cell_dt = stable_timestep(velocity, advection_workspace; cfl = 0.2)
    measure!(
        rows, case_name, "cell_advection",
        () -> advect!(scalar, velocity, cell_dt, advection_workspace),
        () -> begin
            for block in eachindex(scalar.blocks)
                copyto!(scalar.blocks[block], scalar_template.blocks[block])
            end
        end,
        () -> field_sum(scalar),
    )
    measure!(
        rows, case_name, "cell_halo_exchange",
        () -> fill_halo!(scalar), () -> nothing,
        () -> sum(sum, scalar.blocks),
    )

    staggered_velocity = copy(velocity)
    staggered_workspace = StaggeredVelocityAdvectionWorkspace(
        staggered_velocity,
    )
    staggered_dt = stable_timestep(
        velocity, staggered_workspace; cfl = 0.2,
    )
    measure!(
        rows, case_name, "staggered_velocity_advection",
        () -> advect!(
            staggered_velocity, velocity, staggered_dt,
            staggered_workspace; scheme = MC(),
        ),
        () -> begin
            copyto!(staggered_velocity.interior, velocity.interior)
            copyto!(staggered_velocity.boundary, velocity.boundary)
        end,
        () -> face_sum(staggered_velocity),
    )

    function reset_interface!()
        for block in eachindex(interface.fraction.blocks)
            copyto!(
                interface.fraction.blocks[block],
                interface_template.fraction.blocks[block],
            )
        end
        reconstruct!(interface)
        return nothing
    end
    wy_workspace = WYAdvectionWorkspace(interface)
    wy_dt = stable_timestep(velocity, wy_workspace; cfl = 0.2)
    measure!(
        rows, case_name, "wy_interface_advection",
        () -> advect!(interface, velocity, wy_dt, wy_workspace),
        reset_interface!, () -> field_sum(interface.fraction),
    )

    phase_scalar = PhaseScalar(
        interface;
        liquid = (x, y) -> 1 + 0.1x - 0.05y,
        gas = (x, y) -> 0.8 - 0.04x + 0.07y,
    )
    liquid_template = copy(phase_scalar.liquid)
    gas_template = copy(phase_scalar.gas)
    function reset_phase_scalar!()
        reset_interface!()
        for block in eachindex(phase_scalar.liquid.blocks)
            copyto!(phase_scalar.liquid.blocks[block], liquid_template.blocks[block])
            copyto!(phase_scalar.gas.blocks[block], gas_template.blocks[block])
        end
        return nothing
    end
    measure!(
        rows, case_name, "wy_phase_scalar_advection",
        () -> advect!(
            interface, velocity, wy_dt, wy_workspace;
            scalars = (phase_scalar,),
        ),
        reset_phase_scalar!,
        () -> field_sum(phase_scalar.liquid) + field_sum(phase_scalar.gas),
    )
    momentum_destination = FaceVelocity(mesh)
    momentum_workspace = MomentumAdvectionWorkspace(interface)
    measure!(
        rows, case_name, "wy_momentum_advection",
        () -> advect!(
            interface, velocity, wy_dt, wy_workspace;
            momentum = momentum_destination, density = (1000.0, 1.0),
            momentum_workspace, scheme = MC(),
        ),
        reset_interface!,
        () -> field_sum(interface.fraction) + face_sum(momentum_destination),
    )

    pressure = CellField(mesh, (x, y) -> sinpi(2x) * cospi(2y))
    gradient_destination = FaceVelocity(mesh)
    gradient_ref = Ref{Any}()
    measure!(
        rows, case_name, "pressure_gradient_operator_construction",
        () -> (gradient_ref[] = GradientOperator(pressure)), () -> nothing,
        () -> length(gradient_ref[].pressure_jump),
    )
    gradient_operator = GradientOperator(pressure)
    measure!(
        rows, case_name, "pressure_gradient",
        () -> gradient!(gradient_destination, pressure, gradient_operator),
        () -> nothing, () -> face_sum(gradient_destination),
    )

    pressure_operator_ref = Ref{Any}()
    measure!(
        rows, case_name, "pressure_poisson_operator_construction",
        () -> (
            pressure_operator_ref[] = PoissonOperator(
                interface_template; density = (1000.0, 1.0),
            )
        ),
        () -> nothing,
        () -> length(pressure_operator_ref[].volumes),
    )
    pressure_operator = PoissonOperator(
        interface_template; density = (1000.0, 1.0),
    )
    pressure_action = TwoPhaseAMR.FactorizedEllipticOperator(pressure_operator)
    pressure_source = [sin(0.3index) for index in eachindex(pressure_operator.volumes)]
    pressure_product = similar(pressure_source)
    measure!(
        rows, case_name, "pressure_poisson_operator_apply",
        () -> mul!(pressure_product, pressure_action, pressure_source),
        () -> nothing, () -> sum(pressure_product),
    )
    pressure_solution = CellField(mesh)
    pressure_rhs = CellField(mesh, (x, y) -> sinpi(2x) * cospi(2y))
    pressure_solver = EllipticSolver(
        pressure_operator; preconditioner = :tree_geometric,
    )
    pressure_stats = Ref{Any}()
    measure!(
        rows, case_name, "pressure_poisson_solve",
        () -> (
            pressure_stats[] = solve!(
                pressure_solution, pressure_solver, pressure_rhs;
                rtol = 1.0e-8, max_cycles = 100,
            )
        ),
        () -> begin
            for block in pressure_solution.blocks
                fill!(block, 0.0)
            end
        end,
        () -> field_sum(pressure_solution) + pressure_stats[].cycles,
        samples = max(3, BENCHMARK_SAMPLES ÷ 2),
    )

    diffusion_operator_ref = Ref{Any}()
    measure!(
        rows, case_name, "diffusion_operator_construction",
        () -> (
            diffusion_operator_ref[] = StaggeredDiffusionOperator(
                interface_template; viscosity = (0.01, 1.0e-4),
                boundary = FreeSlip(),
            )
        ),
        () -> nothing,
        () -> length(diffusion_operator_ref[].volumes),
        samples = max(3, BENCHMARK_SAMPLES ÷ 2),
    )
    diffusion_operator = StaggeredDiffusionOperator(
        interface_template; viscosity = (0.01, 1.0e-4),
        boundary = FreeSlip(),
    )
    diffusion_destination = FaceVelocity(mesh)
    measure!(
        rows, case_name, "diffusion_coefficient_refresh",
        () -> TwoPhaseAMR._update_dynamic_viscosity!(
            diffusion_operator, interface_template, (0.01, 1.0e-4),
        ),
        () -> nothing,
        () -> sum(diffusion_operator.stress_weights),
    )
    measure!(
        rows, case_name, "diffusion_operator_apply",
        () -> diffusion!(diffusion_destination, velocity, diffusion_operator),
        () -> nothing, () -> face_sum(diffusion_destination),
    )
    density = mixture_density(interface_template, (1000.0, 1.0))
    diffusion_dt = min(wy_dt, 0.01)
    diffusion_solver = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        diffusion_operator, density, diffusion_dt;
        preconditioner = :tree_geometric,
    )
    diffusion_solution = copy(velocity)
    measure!(
        rows, case_name, "diffusion_solve",
        () -> (
            implicit_midpoint_diffusion!(
                diffusion_solution, velocity, diffusion_operator,
                diffusion_dt, diffusion_solver;
                old_velocity = velocity, density,
                rtol = 1.0e-8, max_cycles = 100,
            )
        ),
        () -> begin
            copyto!(diffusion_solution.interior, velocity.interior)
            copyto!(diffusion_solution.boundary, velocity.boundary)
        end,
        () -> face_sum(diffusion_solution) + diffusion_solver.stats.cycles,
        samples = max(3, BENCHMARK_SAMPLES ÷ 2),
    )

    force_destination = FaceVelocity(mesh)
    force_workspace = EllipticWorkspace(pressure_operator)
    measure!(
        rows, case_name, "surface_tension_csf",
        () -> surface_tension!(
            force_destination, interface_template, pressure_operator,
            force_workspace; sigma = 1.0, formulation = CSF(),
        ),
        () -> nothing, () -> face_sum(force_destination),
    )
    measure!(
        rows, case_name, "gravity",
        () -> gravity!(
            force_destination, interface_template, pressure_operator,
            force_workspace; density = (1000.0, 1.0),
            gravity = (0.0, -1.0),
        ),
        () -> nothing, () -> face_sum(force_destination),
    )
    return nothing
end

function write_results(path, rows)
    open(path, "w") do io
        println(io, "case,kernel,threads,time_ns,bytes,allocations,checksum")
        for row in rows
            println(
                io, row.case, ',', row.kernel, ',', row.threads, ',',
                row.time_ns, ',', row.bytes, ',', row.allocations, ',',
                row.checksum,
            )
        end
    end
    return path
end

rows = NamedTuple[]
for (case_name, factory) in (
        "uniform" => uniform_mesh,
        "mixed" => mixed_mesh,
    )
    BENCHMARK_CASE in ("all", case_name) || continue
    benchmark_case!(rows, case_name, factory)
end
if BENCHMARK_CASE in ("all", "uniform")
    transfer_benchmarks!(rows, "uniform")
    adaptation_benchmarks!(rows, "uniform")
end
output_path = isempty(ARGS) ?
    joinpath(@__DIR__, "threaded-kernels-$(Threads.nthreads(:default)).csv") :
    abspath(ARGS[1])
write_results(output_path, rows)
println("Wrote ", output_path)

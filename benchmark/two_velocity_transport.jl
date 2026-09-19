using BenchmarkTools
using TwoPhaseAMR
using Statistics: median

const SAMPLES = parse(Int, get(ENV, "TWO_VELOCITY_BENCHMARK_SAMPLES", "10"))

function _transport_case(mesh)
    periodic_distance(value, centre) = mod(value - centre + 0.5, 1) - 0.5
    levelset = (x, y) ->
    (abs(periodic_distance(x + y, 0.7)) - 0.18) / sqrt(2)
    interface = InterfaceField(mesh, levelset; lipschitz = 1)
    inverse_root_two = inv(sqrt(2.0))
    normal = (inverse_root_two, inverse_root_two)
    tangent = (-inverse_root_two, inverse_root_two)
    velocity = ntuple(2) do phase
        tangent_speed = phase == 1 ? 0.08 : -0.1
        vector = (
            0.12normal[1] + tangent_speed * tangent[1],
            0.12normal[2] + tangent_speed * tangent[2],
        )
        FaceVelocity(
            mesh,
            (x, y, time, axis) -> vector[axis],
        )
    end
    scalar_functions = (
        (x, y) -> 1 + 0.2sinpi(2x) + 0.1cospi(2y),
        (x, y) -> 2 + 0.15cospi(2x) - 0.1sinpi(2y),
    )
    return (; interface, velocity, scalar_functions)
end

function _phase_scalars(interface, functions)
    return (
        PhaseScalar(interface; liquid = functions[1], gas = functions[2]),
        PhaseScalar(interface),
    )
end

function _phase_staggered_scalars(interface, functions)
    return (
        PhaseFaceScalar{1}(
            interface; liquid = functions[1], gas = functions[2],
        ),
        PhaseFaceScalar{2}(
            interface; liquid = functions[1], gas = functions[2],
        ),
    )
end

function _phase_dual_flux_checksum(quantity, work)
    schedule = quantity.workspace.schedule
    result = zero(eltype(quantity.liquid))
    for face in schedule.faces
        result += TwoPhaseAMR._phase_dual_flux(
            face, schedule, work.liquid_flux, work.mesh,
        )
        result += TwoPhaseAMR._phase_dual_flux(
            face, schedule, work.gas_flux, work.mesh,
        )
    end
    return result
end

function _trial_summary(trial)
    estimate = median(trial)
    return (;
        milliseconds = estimate.time / 1.0e6,
        bytes = estimate.memory, allocations = estimate.allocs,
    )
end

function _print_trial(label, trial)
    estimate = _trial_summary(trial)
    println(
        rpad(label, 32),
        lpad(round(estimate.milliseconds; digits = 3), 11), " ms  ",
        lpad(estimate.bytes, 13), " B  ",
        lpad(estimate.allocations, 9), " allocs",
    )
    return estimate
end

function _benchmark_transport(mesh)
    template = _transport_case(mesh)
    projected_method = ProjectedLiquidVelocity(
        tolerance = 1.0e-12, max_cycles = 500,
    )
    frozen_method = FrozenInterfaceFluxes()
    projected_limit = stable_timestep(
        template.velocity,
        WYAdvectionWorkspace(
            template.interface, template.velocity;
            interface_method = projected_method,
        );
        cfl = 0.35,
    )
    frozen_limit = stable_timestep(
        template.velocity,
        WYAdvectionWorkspace(
            template.interface, template.velocity;
            interface_method = frozen_method,
        );
        cfl = 0.35,
    )
    dt = min(projected_limit, frozen_limit)

    one_workspace = @benchmark WYAdvectionWorkspace(interface) setup = (
        interface = copy($(template.interface))
    ) evals = 1 samples = SAMPLES
    projected_workspace = @benchmark WYAdvectionWorkspace(
        interface, $(template.velocity); interface_method = $projected_method,
    ) setup = (
        interface = copy($(template.interface))
    ) evals = 1 samples = SAMPLES
    frozen_workspace = @benchmark WYAdvectionWorkspace(
        interface, $(template.velocity); interface_method = $frozen_method,
    ) setup = (
        interface = copy($(template.interface))
    ) evals = 1 samples = SAMPLES
    momentum_workspace = @benchmark MomentumAdvectionWorkspace(
        interface, $(template.velocity),
    ) setup = (
        interface = copy($(template.interface))
    ) evals = 1 samples = SAMPLES

    projection = @benchmark TwoPhaseAMR._project_liquid_velocity!(
        workspace, $(template.velocity),
    ) setup = (
        interface = copy($(template.interface));
        workspace = WYAdvectionWorkspace(
            interface, $(template.velocity);
            interface_method = $projected_method,
        )
    ) evals = 1 samples = SAMPLES

    one_step = @benchmark advect!(
        interface, $(template.velocity[1]), $dt, workspace;
        scalars = scalars, scheme = Fromm(),
    ) setup = (
        interface = copy($(template.interface));
        scalars = _phase_scalars(interface, $(template.scalar_functions));
        workspace = WYAdvectionWorkspace(interface)
    ) evals = 1 samples = SAMPLES
    projected_step = @benchmark advect!(
        interface, $(template.velocity), $dt, workspace;
        scalars = scalars, scheme = Fromm(),
    ) setup = (
        interface = copy($(template.interface));
        scalars = _phase_scalars(interface, $(template.scalar_functions));
        workspace = WYAdvectionWorkspace(
            interface, $(template.velocity);
            interface_method = $projected_method,
        )
    ) evals = 1 samples = SAMPLES
    frozen_step = @benchmark advect!(
        interface, $(template.velocity), $dt, workspace;
        scalars = scalars, scheme = Fromm(),
    ) setup = (
        interface = copy($(template.interface));
        scalars = _phase_scalars(interface, $(template.scalar_functions));
        workspace = WYAdvectionWorkspace(
            interface, $(template.velocity);
            interface_method = $frozen_method,
        )
    ) evals = 1 samples = SAMPLES

    dual_flux_interpolation = @benchmark _phase_dual_flux_checksum(
        quantity, workspace,
    ) setup = begin
        interface = copy($(template.interface))
        quantity = _phase_staggered_scalars(
            interface, $(template.scalar_functions),
        )[1]
        workspace = WYAdvectionWorkspace(
            interface, $(template.velocity);
            interface_method = $frozen_method,
        )
        reconstruct!(interface)
        TwoPhaseAMR._freeze_phase_apertures!(interface, workspace)
        TwoPhaseAMR._compute_directional_wy_fluxes!(
            interface, $(template.velocity), $dt, workspace, 1,
        )
    end evals = 1 samples = SAMPLES

    one_complete_step = @benchmark advect!(
        interface, $(template.velocity[1]), $dt, workspace;
        scalars, staggered_scalars = staggered,
        momentum = destination, density = (1000.0, 1.0),
        momentum_workspace = momentum, scheme = Fromm(),
    ) setup = begin
        interface = copy($(template.interface))
        scalars = _phase_scalars(interface, $(template.scalar_functions))
        staggered = _phase_staggered_scalars(
            interface, $(template.scalar_functions),
        )
        destination = FaceVelocity($(mesh))
        workspace = WYAdvectionWorkspace(interface)
        momentum = MomentumAdvectionWorkspace(interface)
    end evals = 1 samples = SAMPLES
    projected_complete_step = @benchmark advect!(
        interface, $(template.velocity), $dt, workspace;
        scalars, staggered_scalars = staggered,
        momentum = destination, density = (1000.0, 1.0),
        momentum_workspace = momentum, scheme = Fromm(),
    ) setup = begin
        interface = copy($(template.interface))
        scalars = _phase_scalars(interface, $(template.scalar_functions))
        staggered = _phase_staggered_scalars(
            interface, $(template.scalar_functions),
        )
        destination = (FaceVelocity($(mesh)), FaceVelocity($(mesh)))
        workspace = WYAdvectionWorkspace(
            interface, $(template.velocity);
            interface_method = $projected_method,
        )
        momentum = MomentumAdvectionWorkspace(
            interface, $(template.velocity),
        )
    end evals = 1 samples = SAMPLES
    frozen_complete_step = @benchmark advect!(
        interface, $(template.velocity), $dt, workspace;
        scalars, staggered_scalars = staggered,
        momentum = destination, density = (1000.0, 1.0),
        momentum_workspace = momentum, scheme = Fromm(),
    ) setup = begin
        interface = copy($(template.interface))
        scalars = _phase_scalars(interface, $(template.scalar_functions))
        staggered = _phase_staggered_scalars(
            interface, $(template.scalar_functions),
        )
        destination = (FaceVelocity($(mesh)), FaceVelocity($(mesh)))
        workspace = WYAdvectionWorkspace(
            interface, $(template.velocity);
            interface_method = $frozen_method,
        )
        momentum = MomentumAdvectionWorkspace(
            interface, $(template.velocity),
        )
    end evals = 1 samples = SAMPLES

    return (;
        one_workspace, projected_workspace, frozen_workspace,
        momentum_workspace, projection, dual_flux_interpolation,
        one_step, projected_step, frozen_step,
        one_complete_step, projected_complete_step,
        frozen_complete_step,
    )
end

println("Julia ", VERSION, ", threads=", Threads.nthreads(:default))
println("CPU: ", Sys.cpu_info()[1].model)
println("samples=", SAMPLES)

for (name, mesh) in (
        "uniform" => BlockTreeMesh(
            block_size = (32, 32), periodic = (true, true),
        ),
        "mixed" => BlockTreeMesh(
            block_size = (12, 12), max_level = 2,
            periodic = (true, true),
            refine = (key, origin, extent) -> key.level == 0 ||
                (key.level == 1 && key.i == 0 && key.j == 0),
        ),
    )
    println("\n$name: ", ncells(mesh), " cells, ", length(mesh.faces), " faces")
    trials = _benchmark_transport(mesh)
    summaries = Dict{Symbol, NamedTuple}()
    for key in keys(trials)
        summaries[key] = _print_trial(String(key), getproperty(trials, key))
    end
    one = summaries[:one_step]
    for method in (:projected_step, :frozen_step)
        value = summaries[method]
        println(
            "  ", method, "/one_step: ",
            round(value.milliseconds / one.milliseconds; digits = 2),
            "x time, ", round(value.bytes / max(one.bytes, 1); digits = 2),
            "x bytes",
        )
    end
    one_complete = summaries[:one_complete_step]
    for method in (:projected_complete_step, :frozen_complete_step)
        value = summaries[method]
        println(
            "  ", method, "/one_complete_step: ",
            round(value.milliseconds / one_complete.milliseconds; digits = 2),
            "x time, ",
            round(value.bytes / max(one_complete.bytes, 1); digits = 2),
            "x bytes",
        )
    end
end

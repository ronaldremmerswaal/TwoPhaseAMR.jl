# Development-only, non-overlapping pressure measurements. No solver methods
# are replaced. Run `julia --project=. .../pressure_probe.jl cmdgfm 0.5 3`.
using TwoPhaseAMR
using LinearAlgebra
using SparseArrays
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "..", "examples", "stokes3_wave.jl"))
const M = TwoPhaseAMR

function timed(label, f, samples)
    result = @timed f()
    push!(samples, (label = label, seconds = result.time,
        bytes = result.bytes, gc = result.gctime,
        compile = result.compile_time))
    return result.value
end

function decompose_cmdgfm(flow; dt = 1.0)
    state = M._two_velocity_flow_workspace!(
        flow.workspace.two_phase, flow.solver.two_phase,
    )
    solver = state.projection_solver
    work = state.projection_workspace
    mesh = solver.interface.mesh
    velocity = flow.velocity
    T = eltype(solver.interface)
    samples = NamedTuple[]
    data = timed(:face_data, () -> M._mimetic_face_data(solver, work), samples)
    operator = timed(:pressure_operator,
        () -> M._mimetic_pressure_operator(solver), samples)
    incidence = timed(:incidence,
        () -> M._mimetic_incidence(mesh, T), samples)
    active = timed(:active_cells,
        () -> M._mimetic_active_cells(solver.interface), samples)
    weights = isnothing(solver.interface.geometry) ?
        ones(T, length(mesh.faces)) : copy(solver.interface.geometry.aperture)
    indices, anchors = timed(:gauge_partition,
        () -> M._gauge_free_component_partition(mesh, active, weights), samples)
    stencils_all = timed(:jump_stencils,
        () -> M._cmdgfm_jump_stencils(solver, data), samples)
    stencils = filter(s -> s.resolved, stencils_all)
    mixed = [s.face for s in stencils]
    pressure_positive = timed(:pressure_slice,
        () -> operator.stiffness[indices, indices], samples)
    incidence_pressure = timed(:incidence_slice,
        () -> incidence[:, indices], samples)
    pressure_jump = timed(:pressure_jump,
        () -> transpose(incidence_pressure[mixed, :]) *
            spdiagm(0 => data.coupling[mixed]), samples)
    gradient = timed(:common_gradient,
        () -> M._cmdgfm_common_gradient_matrix(
            operator, incidence, solver.interface.geometry), samples)
    jump_pressure, jump_block, jump_rhs, unresolved = timed(:jump_blocks,
        () -> M._cmdgfm_jump_blocks(stencils, mixed, data, gradient,
            indices, velocity, dt, T), samples)
    unresolved == 0 || error("unexpected retained unresolved row")
    matrix = timed(:block_concat,
        () -> [pressure_positive pressure_jump; jump_pressure jump_block],
        samples)
    mixture = data.liquid_aperture .* velocity[1].interior .+
        data.gas_aperture .* velocity[2].interior
    boundary, = timed(:boundary_constraints,
        () -> M._mimetic_boundary_constraints(solver.interface, velocity),
        samples)
    rhs = timed(:rhs,
        () -> vcat((transpose(incidence_pressure) * mixture .-
            boundary[indices]) ./ T(dt), jump_rhs), samples)
    assembled = timed(:whole_assembly_check,
        () -> M._assemble_cmdgfm_system!(solver, work, dt, velocity), samples)
    matrix == assembled.matrix || error("staged matrix differs from production")
    rhs == assembled.right_hand_side || error("staged rhs differs from production")
    indices == assembled.pressure_indices || error("pressure gauge differs")
    mixed == assembled.mixed_faces || error("retained faces differ")
    # Drill-down repeats work inside `pressure_operator` and must NOT be added
    # to the non-overlapping assembly rows above.
    geometry = solver.interface.geometry
    if isnothing(geometry)
        # Stokes3's force path refreshes this operator before each physical
        # projection. Fresh assembly and coefficient refresh can differ by
        # roundoff even when their coefficients and metrics agree. Quantify
        # that difference instead of assuming bitwise identity.
        scalar = flow.solver.pressure_solver.operator
        metric = M._cmdgfm_metric_matrix(
            operator.gradient_operator.inverse_metric)
        scalar_metric = M._cmdgfm_metric_matrix(
            scalar.gradient_operator.inverse_metric)
        coefficient_difference = maximum(abs,
            operator.face_coefficient .- scalar.face_coefficient)
        stiffness_difference = maximum(abs,
            operator.stiffness - scalar.stiffness)
        metric_difference = maximum(abs, metric - scalar_metric)
        @printf("SCALAR_COMPARISON faces=%d coefficient_linf=%.8g stiffness_linf=%.8g metric_linf=%.8g\n",
            length(mesh.faces), coefficient_difference,
            stiffness_difference, metric_difference)
        density = timed(:drill_mixture_density,
            () -> M.mixture_density(solver.interface, solver.density),
            samples)
        coefficient = inv.(density)
        metric = timed(:drill_gradient_metric,
            () -> M.GradientOperator(mesh, T), samples)
        core = timed(:drill_elliptic_from_metric,
            () -> M.EllipticOperator(mesh; reaction = 0, diffusion = 1,
                face_coefficient = coefficient, gradient_operator = metric,
                T), samples)
        core.stiffness == operator.stiffness ||
            error("pressure operator drill-down disagrees")
    else
        density = timed(:drill_mixture_density,
            () -> M.mixture_density(solver.interface, solver.density,
                geometry), samples)
        coefficient = geometry.aperture ./ density
        metric = timed(:drill_gradient_metric,
            () -> M.GradientOperator(geometry), samples)
        volumes = M._geometry_cell_volumes(geometry)
        core = timed(:drill_elliptic_from_metric,
            () -> M.EllipticOperator(mesh; reaction = 0, diffusion = 1,
                face_coefficient = coefficient, gradient_operator = metric,
                volumes, active_cells = geometry.active_cells, T), samples)
        core.stiffness == operator.stiffness ||
            error("pressure operator drill-down disagrees")
    end
    timed(:asymmetry, () -> maximum(abs, matrix - transpose(matrix)), samples)
    # Current flow has just undergone a projection; gradient recovery may be
    # timed without evolving velocity or pressure.
    timed(:gradient_recovery,
        () -> M.cmdgfm_gradient!(
            (work.liquid_gradient, work.gas_gradient), flow.pressure,
            solver, work, operator, data), samples)
    return (; samples, assembled, solver, work, faces = length(mesh.faces),
        cells = M.ncells(mesh), mixed = length(mixed),
        matrix_nnz = nnz(matrix), jump_nnz = nnz(jump_block),
        stencils_all = length(stencils_all))
end

function preconditioner_probe!(probe)
    (; solver, work, assembled) = probe
    samples = NamedTuple[]
    # Compare construction with the exact same frozen matrix. Restore the
    # original object at the end; this script runs after the trajectory.
    previous = work.block_preconditioner
    work.block_preconditioner = nothing
    cold = timed(:block_preconditioner_cold,
        () -> M._cmdgfm_preconditioner(solver, assembled, work), samples)
    hot = timed(:block_preconditioner_same_matrix,
        () -> M._cmdgfm_preconditioner(solver, assembled, work), samples)
    timed(:pressure_gmg_numeric_refresh,
        () -> M._refresh_geometric_multigrid!(
            hot.pressure_preconditioner.multigrid,
            assembled.pressure_operator), samples)
    rhs = ones(eltype(assembled.right_hand_side), size(assembled.matrix, 1))
    out = similar(rhs)
    timed(:block_preconditioner_apply,
        () -> ldiv!(out, hot, rhs), samples)
    if !isnothing(hot.jump_preconditioner)
        jump_rhs = ones(eltype(rhs), size(assembled.jump_block, 1))
        jump_out = similar(jump_rhs)
        timed(:jump_ilu_apply,
            () -> ldiv!(jump_out, hot.jump_preconditioner, jump_rhs), samples)
        timed(:jump_ilu_fresh,
            () -> M.incomplete_lu_preconditioner(
                assembled.jump_block; drop_tolerance = 1e-3), samples)
    end
    pressure_rhs = ones(eltype(rhs), length(assembled.pressure_indices))
    pressure_out = similar(pressure_rhs)
    timed(:pressure_gmg_apply,
        () -> ldiv!(pressure_out, hot.pressure_preconditioner, pressure_rhs),
        samples)
    krylov = M.Krylov.BicgstabWorkspace(
        assembled.matrix, assembled.right_hand_side)
    solve() = M.Krylov.bicgstab!(
        krylov, assembled.matrix, assembled.right_hand_side;
        N = hot, ldiv = true, atol = 0.0, rtol = 1e-8, itmax = 500)
    solve() # compile the fixed-matrix solve before the timed sample
    timed(:bicgstab_frozen_rhs, solve, samples)
    factor = hot.jump_preconditioner.value
    factor_nnz = nnz(factor.L) + nnz(factor.U)
    work.block_preconditioner = previous
    return (; samples, cold_reused = cold.pressure_hierarchy_reused,
        hot_pressure_reused = hot.pressure_hierarchy_reused,
        hot_jump_reused = hot.jump_factor_reused,
        factor_nnz, factor_fill = factor_nnz / nnz(assembled.jump_block),
        krylov_iterations = M.Krylov.iteration_count(krylov))
end

function decompose_one(flow)
    model = flow.solver.two_phase
    pressure_solver = flow.solver.pressure_solver
    interface = model.interface
    geometry = flow.solver.geometry
    samples = NamedTuple[]
    operator = timed(:scalar_operator_coefficients, () ->
        isnothing(geometry) ? M.PoissonOperator(interface;
            density = model.density, formulation = model.formulation,
            template = pressure_solver.operator) :
            M.PoissonOperator(interface, geometry;
                density = model.density, formulation = model.formulation,
                template = pressure_solver.operator), samples)
    timed(:scalar_gmg_refresh,
        () -> M._refresh_geometric_multigrid!(
            pressure_solver.preconditioner, operator), samples)
    timed(:whole_scalar_refresh,
        () -> M._refresh_pressure_substep!(
            flow.solver, flow.workspace, model), samples)
    pressure_rhs = ones(eltype(operator.volumes), length(operator.volumes))
    pressure_out = similar(pressure_rhs)
    timed(:scalar_gmg_apply,
        () -> ldiv!(pressure_out,
            pressure_solver.preconditioner, pressure_rhs), samples)
    return (; samples, faces = length(flow.solver.mesh.faces),
        cells = M.ncells(flow.solver.mesh), matrix_nnz = nnz(operator.matrix))
end

function structural_record(state)
    workspace = state.workspace.two_phase.two_velocity[]
    isnothing(workspace) && return nothing
    pressure = workspace.projection_workspace
    isempty(pressure.matrix) && return nothing
    pre = pressure.block_preconditioner
    krylov = pressure.krylov
    return (;
        step = state.step, faces = length(state.solver.mesh.faces),
        cells = M.ncells(state.solver.mesh),
        pressure_unknowns = length(pressure.pressure_indices),
        jump_unknowns = length(pressure.mixed_faces),
        matrix_nnz = nnz(pressure.matrix), jump_nnz = nnz(pressure.jump_block),
        gauge_hash = hash(pressure.pressure_indices),
        faces_hash = hash(pressure.mixed_faces),
        pattern_hash = hash((pressure.matrix.colptr,
            rowvals(pressure.matrix))),
        values_hash = hash(nonzeros(pressure.jump_block)),
        pressure_reused = pre isa M.CMDGFMBlockPreconditioner &&
            pre.pressure_hierarchy_reused,
        jump_reused = pre isa M.CMDGFMBlockPreconditioner &&
            pre.jump_factor_reused,
        last_krylov_iterations = isnothing(krylov) ? 0 :
            M.Krylov.iteration_count(krylov),
    )
end

function print_samples(prefix, samples)
    for sample in samples
        @printf("%s stage=%s seconds=%.6f bytes=%d gc=%.6f compile=%.6f\n",
            prefix, sample.label, sample.seconds, sample.bytes,
            sample.gc, sample.compile)
    end
end

function run_probe(mode::Symbol; final_time = 0.5, repetitions = 3)
    model = mode == :one ? OneVelocityModel() :
        TwoVelocityModel(pressure_projection = CMDGFMProjection())
    @printf("ENV mode=%s julia=%s threads=%d blas=%d target=%.3f reps=%d\n",
        mode, VERSION, Threads.nthreads(), BLAS.get_num_threads(),
        final_time, repetitions)
    warm = run_example(velocity_model = model, final_time = 0.03)
    @printf("WARM mode=%s steps=%d faces=%d\n", mode,
        warm.result.steps, length(warm.flow.solver.mesh.faces))
    flush(stdout)
    for run in 1:repetitions
        flow = run_example(velocity_model = model, final_time = 0.0).flow
        records = NamedTuple[]
        cycles = Int[]
        if mode == :cmdgfm
            add_diagnostic!(flow, :pressure_structure,
                state -> begin
                    record = structural_record(state)
                    isnothing(record) || push!(records, record)
                    return isnothing(record) ? 0 : record.jump_unknowns
                end)
        else
            add_diagnostic!(flow, :pressure_cycles,
                state -> begin
                    value = state.solver.pressure_solver.stats.cycles
                    push!(cycles, value)
                    return value
                end)
        end
        if run == 1 && mode == :cmdgfm
            decompose_cmdgfm(flow)
            initial = decompose_cmdgfm(flow)
            @printf("FROZEN_INITIAL faces=%d cells=%d mixed=%d matrix_nnz=%d\n",
                initial.faces, initial.cells, initial.mixed,
                initial.matrix_nnz)
            print_samples("ASSEMBLY_INITIAL", initial.samples)
        end
        initial_faces = length(flow.solver.mesh.faces)
        advance = @timed simulate!(flow, final_time)
        stats = advance.value.pressure
        @printf("RUN mode=%s rep=%d seconds=%.6f bytes=%d gc=%.6f compile=%.6f steps=%d amr=%d initial_faces=%d final_faces=%d final_cells=%d iterations=%d converged=%s\n",
            mode, run, advance.time, advance.bytes, advance.gctime,
            advance.compile_time, advance.value.steps,
            advance.value.mesh_updates, initial_faces,
            length(flow.solver.mesh.faces), M.ncells(flow.solver.mesh),
            hasproperty(stats, :iterations) ? stats.iterations : stats.cycles,
            stats.converged)
        if mode == :cmdgfm
            changes = sum(records[i].pattern_hash != records[i-1].pattern_hash
                for i in 2:length(records))
            face_changes = sum(records[i].faces_hash != records[i-1].faces_hash
                for i in 2:length(records))
            jump_reuse = count(r -> r.jump_reused, records)
            pressure_reuse = count(r -> r.pressure_reused, records)
            @printf("STRUCT rep=%d records=%d pattern_changes=%d jump_face_changes=%d pressure_reuses=%d jump_reuses=%d\n",
                run, length(records), changes, face_changes,
                pressure_reuse, jump_reuse)
            post_initial = filter(r -> r.step > 0, records)
            if !isempty(post_initial)
                iterations = [r.last_krylov_iterations for r in post_initial]
                @printf("KRYLOV_LAST_CALL rep=%d median=%.1f p95=%.1f max=%d (last correction only)\n",
                    run, median(iterations), quantile(iterations, 0.95),
                    maximum(iterations))
            end
            for i in eachindex(records)
                record = records[i]
                notable = i <= 3 || i > length(records) - 3 ||
                    (i > 1 && record.faces != records[i-1].faces)
                notable || continue
                @printf("STEP rep=%d step=%d faces=%d cells=%d pressure=%d jump=%d matrix_nnz=%d jump_nnz=%d last_krylov=%d pressure_reused=%s jump_reused=%s\n",
                    run, record.step, record.faces, record.cells,
                    record.pressure_unknowns, record.jump_unknowns,
                    record.matrix_nnz, record.jump_nnz,
                    record.last_krylov_iterations,
                    record.pressure_reused, record.jump_reused)
            end
            if run == 1
                # First call also compiles the decomposition. The second is
                # timed after that warm-up, always on exactly the same state.
                decompose_cmdgfm(flow)
                probe = decompose_cmdgfm(flow)
                @printf("FROZEN faces=%d cells=%d mixed=%d all_stencils=%d matrix_nnz=%d jump_nnz=%d\n",
                    probe.faces, probe.cells, probe.mixed,
                    probe.stencils_all, probe.matrix_nnz, probe.jump_nnz)
                print_samples("ASSEMBLY", probe.samples)
                preconditioner_probe!(probe)
                pre = preconditioner_probe!(probe)
                print_samples("PRECONDITIONER", pre.samples)
                @printf("FACTOR jump_nnz=%d factor_nnz=%d fill=%.3f frozen_krylov_iterations=%d\n",
                    probe.jump_nnz, pre.factor_nnz, pre.factor_fill,
                    pre.krylov_iterations)
                decompose_one(flow)
                scalar = decompose_one(flow)
                @printf("MATCHED_SCALAR faces=%d cells=%d matrix_nnz=%d\n",
                    scalar.faces, scalar.cells, scalar.matrix_nnz)
                print_samples("MATCHED_SCALAR", scalar.samples)
            end
        else
            post_initial = length(cycles) > 1 ? cycles[2:end] : cycles
            if !isempty(post_initial)
                @printf("CG_CYCLES rep=%d median=%.1f p95=%.1f max=%d\n",
                    run, median(post_initial), quantile(post_initial, 0.95),
                    maximum(post_initial))
            end
            if run == 1
                decompose_one(flow)
                probe = decompose_one(flow)
                @printf("FROZEN_ONE faces=%d cells=%d matrix_nnz=%d\n",
                    probe.faces, probe.cells, probe.matrix_nnz)
                print_samples("SCALAR", probe.samples)
            end
        end
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    mode = Symbol(get(ARGS, 1, "cmdgfm"))
    mode in (:one, :cmdgfm) || error("mode must be one or cmdgfm")
    run_probe(mode;
        final_time = parse(Float64, get(ARGS, 2, "0.5")),
        repetitions = parse(Int, get(ARGS, 3, "3")))
end

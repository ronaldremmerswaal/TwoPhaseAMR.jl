# Read-only production profiling and frozen-state AMR experiments.
# No production methods are replaced. The uncached reference and all cached
# alternatives change only scratch: physical inactive velocities remain NaN.
#
# julia --project=. development/stokes3_performance/amr_analysis.jl one 0.5
# julia --project=. development/stokes3_performance/amr_analysis.jl cmdgfm 0.5
# Add "allocs" as the third argument for a sampled allocation profile instead.
include(joinpath(@__DIR__, "benchmark.jl"))
const AMRAnalysis = TwoPhaseAMR

function amr_category(names)
    if contains_name(names, "_rebuild_two_phase_topology_dependent_state!")
        contains_name(names, "_finish_phase_face_transfer!") && return :rebuild_intrinsic_transport
        contains_name(names, "_reconfigure_momentum_component!") && return :rebuild_shared_transport
        contains_name(names, "_staggered_diffusion_operator") && return :rebuild_diffusion_operator
        contains_name(names, "VariableStaggeredDiffusionSolver") && return :rebuild_diffusion_solver
        contains_name(names, "TreeGeometricMultigridPreconditioner") && return :rebuild_pressure_multigrid
        contains_name(names, "PoissonOperator") && return :rebuild_pressure_operator
        return :rebuild_other
    elseif contains_name(names, "adapt!")
        if contains_name(names, "_prepare_adaptive_phase_velocities!")
            # Schedule construction includes dual-face adjacency, sort/Dict
            # work and advection-only scratch that transfer does not need.
            any(fragment -> contains_name(names, fragment),
                ("StaggeredAdvectionWorkspace", "_bounded_phase_schedule",
                 "_staggered_advection_workspace")) && return :adapt_phase_transfer_schedules
            contains_name(names, "_extend_phase_transfer_values!") && return :adapt_phase_transfer_extension
            return :adapt_phase_transfer_other
        end
        contains_name(names, "_extrapolate_curl_stencil!") && return :adapt_vorticity_extension
        contains_name(names, "_cell_centered_velocity") && return :adapt_vorticity_interpolation
        (contains_name(names, "visualization_values") ||
            contains_name(names, "_vorticity_values!")) && return :adapt_vorticity_other
        contains_name(names, "_transfer_face_field!") && return :adapt_face_transfer
        contains_name(names, "_transfer_cell_field!") && return :adapt_cell_transfer
        contains_name(names, "_stage_field!") && return :adapt_stage_fields
        contains_name(names, "_finish_prolongation!") && return :adapt_finish_interface
        contains_name(names, "_rebuild_mesh_schedules!") && return :adapt_mesh_schedules
        contains_name(names, "_build_face_transfer_geometry!") && return :adapt_transfer_geometry
        contains_name(names, "refinement_labels!") && return :adapt_other_indicators
        return :adapt_other
    end
    return stage(names)
end

function focused_sample_summary(wall)
    data, dictionary = Profile.retrieve(; include_meta = false)
    counts = Dict{Symbol, Int}()
    functions = Dict{String, Int}()
    current = UInt[]
    for ip in data
        if iszero(ip)
            isempty(current) && continue
            names = stack_names(current, dictionary)
            category = amr_category(names)
            counts[category] = get(counts, category, 0) + 1
            if startswith(string(category), "adapt_") || startswith(string(category), "rebuild_")
                for name in unique(names)
                    startswith(name, "_") || continue
                    functions[name] = get(functions, name, 0) + 1
                end
            end
            empty!(current)
        else
            push!(current, ip)
        end
    end
    total = sum(values(counts))
    active = total - get(counts, :unclassified, 0)
    println("FOCUSED_SAMPLES total=", total, " active=", active)
    for (name, count) in sort!(collect(counts); by = last, rev = true)
        name == :unclassified && continue
        @printf("AMR_STAGE %-38s %7d %6.2f%% %9.4fs\n", name, count,
            100 * count / active, wall * count / active)
    end
    # These inclusive function rows overlap and must not be added.
    for (name, count) in first(sort!(collect(functions); by = last, rev = true),
            min(30, length(functions)))
        @printf("AMR_INCLUSIVE %-48s %7d %9.4fs\n", name, count, wall * count / active)
    end
end

function focused_allocation_summary(rate)
    counts = Dict{Symbol, Int}()
    bytes = Dict{Symbol, Int}()
    allocations = Profile.Allocs.fetch().allocs
    for allocation in allocations
        category = amr_category(string.(getproperty.(allocation.stacktrace, :func)))
        counts[category] = get(counts, category, 0) + 1
        bytes[category] = get(bytes, category, 0) + allocation.size
    end
    println("ALLOC_SAMPLES count=", length(allocations), " rate=", rate,
        " note=uniform_per_allocation_sampling_not_exact_bytes")
    for (name, size) in sort!(collect(bytes); by = last, rev = true)
        @printf("AMR_ALLOC %-38s count=%7d sampled_bytes=%12d estimated_GB=%9.4f\n",
            name, counts[name], size, size / rate / 1e9)
    end
end

# Keep the original allocating path as an explicit reference even after the
# production cache is implemented. Its discarded general schedule is part of
# the measured cost, not a different extension or a production method override.
function uncached_phase_transfer!(velocity, interface)
    mesh = velocity.mesh
    T = eltype(velocity)
    for axis in 1:2
        field = axis == 1 ? FaceField{1}(mesh, T) : FaceField{2}(mesh, T)
        copyto!(field, velocity)
        work = StaggeredAdvectionWorkspace(field)
        if !all(mesh.periodic)
            work = StaggeredAdvectionWorkspace(work.tendency, work.rates,
                AMRAnalysis._bounded_phase_schedule(mesh, Val(axis), T), Ref(true))
        end
        values = vcat(field.interior, field.boundary)
        AMRAnalysis._extend_phase_transfer_values!(values, work, mesh, axis)
        AMRAnalysis._unpack_phase_faces!(field, values)
        copyto!(velocity, field)
    end
    return velocity
end

# Original small development alternative: borrow existing same-mesh
# schedules, still allocating numerical scratch. The production path now also
# retains scratch and invalidates central mesh connectivity by topology epoch.
function cached_schedule_transfer!(velocity, xwork, ywork)
    T = eltype(velocity)
    for (axis, work) in ((1, xwork), (2, ywork))
        field = axis == 1 ? FaceField{1}(velocity.mesh, T) : FaceField{2}(velocity.mesh, T)
        copyto!(field, velocity)
        values = vcat(field.interior, field.boundary)
        AMRAnalysis._extend_phase_transfer_values!(values, work, velocity.mesh, axis)
        AMRAnalysis._unpack_phase_faces!(field, values)
        copyto!(velocity, field)
    end
    return velocity
end

function frozen_transfer_comparison(flow, state)
    mesh = flow.solver.mesh
    xwork, ywork = state.momentum.x.workspace, state.momentum.y.workspace
    reference = map(v -> FaceVelocity(mesh, copy(v.interior), copy(v.boundary)), flow.velocity)
    candidate = map(v -> FaceVelocity(mesh, copy(v.interior), copy(v.boundary)), flow.velocity)
    production = map(copy, flow.velocity)
    scratch = AMRAnalysis.PhaseTransferScratch(eltype(first(flow.velocity)))
    for velocity in reference
        uncached_phase_transfer!(velocity, flow.interface)
    end
    for velocity in candidate
        cached_schedule_transfer!(velocity, xwork, ywork)
    end
    for velocity in production
        AMRAnalysis._extend_phase_velocity_for_transfer!(velocity, flow.interface, scratch)
    end
    error = maximum(maximum(abs.(r.interior .- c.interior)) for (r, c) in zip(reference, candidate))
    boundary_error = maximum(maximum(abs.(r.boundary .- c.boundary); init = 0.0)
        for (r, c) in zip(reference, candidate))
    @printf("FROZEN_EQUIVALENCE time=%.3f faces=%d interior_Linf=%.4e boundary_Linf=%.4e\n",
        flow.time, length(mesh.faces), error, boundary_error)
    @assert iszero(error) && iszero(boundary_error)
    @assert all(isequal(r.interior, p.interior) && isequal(r.boundary, p.boundary)
        for (r, p) in zip(reference, production))
    # Reset outside the timed region; no AMR or physical state is changed.
    for repetition in 1:5
        for (dest, source) in zip(reference, flow.velocity)
            copyto!(dest.interior, source.interior); copyto!(dest.boundary, source.boundary)
        end
        measured = @timed for velocity in reference
            uncached_phase_transfer!(velocity, flow.interface)
        end
        @printf("FROZEN_TRANSFER variant=uncached_reference repeat=%d time=%.6f bytes=%d gc=%.6f\n",
            repetition, measured.time, measured.bytes, measured.gctime)
        for (dest, source) in zip(candidate, flow.velocity)
            copyto!(dest.interior, source.interior); copyto!(dest.boundary, source.boundary)
        end
        measured = @timed for velocity in candidate
            cached_schedule_transfer!(velocity, xwork, ywork)
        end
        @printf("FROZEN_TRANSFER variant=cached_schedule repeat=%d time=%.6f bytes=%d gc=%.6f\n",
            repetition, measured.time, measured.bytes, measured.gctime)
        for (dest, source) in zip(production, flow.velocity)
            copyto!(dest.interior, source.interior); copyto!(dest.boundary, source.boundary)
        end
        measured = @timed for velocity in production
            AMRAnalysis._extend_phase_velocity_for_transfer!(velocity, flow.interface, scratch)
        end
        @printf("FROZEN_TRANSFER variant=production_cached_scratch repeat=%d time=%.6f bytes=%d gc=%.6f\n",
            repetition, measured.time, measured.bytes, measured.gctime)
    end
end

function phase_activity_scan(interface, phase)
    mesh = interface.mesh
    active_faces = active_cells = 0
    for (index, face) in pairs(mesh.faces)
        active_faces += AMRAnalysis._phase_face_is_active(interface, index, face, phase)
    end
    nx, ny = mesh.block_size
    for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        active_cells += AMRAnalysis._phase_cell_is_active(interface,
            AMRAnalysis.CellRef(block, i, j), phase)
    end
    return active_faces, active_cells
end

function specialized_phase_activity_scan(interface, geometry::G, phase) where {G}
    mesh = interface.mesh
    active_faces = active_cells = 0
    for (index, face) in pairs(mesh.faces)
        active_faces += AMRAnalysis._phase_face_is_active(interface, geometry,
            index, face, phase)
    end
    nx, ny = mesh.block_size
    for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        active_cells += AMRAnalysis._phase_cell_is_active(interface, geometry,
            AMRAnalysis.CellRef(block, i, j), phase)
    end
    return active_faces, active_cells
end

function frozen_activity_comparison(flow)
    interface = flow.interface
    reference = phase_activity_scan(interface, :liquid)
    candidate = specialized_phase_activity_scan(interface, interface.geometry, :liquid)
    @assert reference == candidate
    println("FROZEN_ACTIVITY_EQUIVALENCE time=", flow.time,
        " faces=", length(interface.mesh.faces), " counts=", reference)
    for repetition in 1:5
        measured = @timed phase_activity_scan(interface, :liquid)
        @printf("FROZEN_ACTIVITY variant=production repeat=%d time=%.6f bytes=%d\n",
            repetition, measured.time, measured.bytes)
        measured = @timed specialized_phase_activity_scan(interface, interface.geometry, :liquid)
        @printf("FROZEN_ACTIVITY variant=typed_geometry repeat=%d time=%.6f bytes=%d\n",
            repetition, measured.time, measured.bytes)
    end
end

function amr_analysis(mode; final_time = 0.5, allocations = false)
    model = mode == :one ? OneVelocityModel() :
        TwoVelocityModel(pressure_projection = CMDGFMProjection())
    mode in (:one, :cmdgfm) || throw(ArgumentError("expected one or cmdgfm"))
    println("SOURCE ", pathof(TwoPhaseAMR), " runtime=", VERSION,
        " threads=", Threads.nthreads(), " blas=", BLAS.get_num_threads())
    warm = run_example(velocity_model = model, final_time = min(0.05, final_time))
    println("WARMUP steps=", warm.result.steps, " updates=", warm.result.mesh_updates)
    if mode == :cmdgfm
        frozen_transfer_comparison(warm.flow, warm.flow.workspace.two_phase.two_velocity[])
        frozen_activity_comparison(warm.flow)
    end
    flow = run_example(velocity_model = model, final_time = 0.0).flow
    measured = nothing
    flush(stdout)
    if allocations
        rate = 0.0002
        Profile.Allocs.clear()
        Profile.Allocs.@profile sample_rate=rate measured = @timed simulate!(flow, final_time)
        focused_allocation_summary(rate)
    else
        Profile.init(; n = 6_000_000, delay = 0.002)
        Profile.clear()
        Profile.@profile measured = @timed simulate!(flow, final_time)
        focused_sample_summary(measured.time)
    end
    @printf("FOCUSED_ADVANCE mode=%s final_time=%.3f wall=%.6f bytes=%d gc=%.6f compile=%.6f steps=%d updates=%d faces=%d blocks=%d\n",
        mode, final_time, measured.time, measured.bytes, measured.gctime,
        measured.compile_time, measured.value.steps, measured.value.mesh_updates,
        length(flow.solver.mesh.faces), nblocks(flow.solver.mesh))
    flush(stdout)
    if mode == :cmdgfm
        frozen_transfer_comparison(flow, flow.workspace.two_phase.two_velocity[])
        frozen_activity_comparison(flow)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    amr_analysis(Symbol(get(ARGS, 1, "cmdgfm"));
        final_time = parse(Float64, get(ARGS, 2, "0.5")),
        allocations = get(ARGS, 3, "") == "allocs")
end

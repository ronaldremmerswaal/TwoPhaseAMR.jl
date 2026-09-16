using TwoPhaseAMR
using LinearAlgebra: BLAS
using Statistics: median

const TP = TwoPhaseAMR
const SAMPLES = parse(Int, get(ENV, "TWOPHASE_LOCALITY_SAMPLES", "15"))

Threads.nthreads(:default) == 1 || error(
    "the locality experiment isolates serial traversal; run with --threads=1",
)
BLAS.set_num_threads(1)

"""A compact permutation grouped by owning leaf-block index."""
struct BlockIndexGroups
    offsets::Vector{Int}
    indices::Vector{Int}
end

function BlockIndexGroups(owners::Vector{Int}, number_of_blocks::Int)
    counts = zeros(Int, number_of_blocks)
    for owner in owners
        @inbounds counts[owner] += 1
    end
    offsets = Vector{Int}(undef, number_of_blocks + 1)
    offsets[1] = 1
    for block in 1:number_of_blocks
        offsets[block + 1] = offsets[block] + counts[block]
    end
    cursor = copy(offsets)
    indices = Vector{Int}(undef, length(owners))
    for index in eachindex(owners)
        owner = @inbounds owners[index]
        @inbounds indices[cursor[owner]] = index
        @inbounds cursor[owner] += 1
    end
    return BlockIndexGroups(offsets, indices)
end

"""Axis-specific global-face maps plus an owner-block traversal permutation."""
struct ComponentFacePlan
    interior_global::Vector{Int}
    boundary_global::Vector{Int}
    groups::BlockIndexGroups
end

function ComponentFacePlan(mesh::BlockTreeMesh, axis::Int)
    interior_global = Int[]
    boundary_global = Int[]
    owners = Int[]
    sizehint!(interior_global, count(face -> face.axis == axis, mesh.faces))
    sizehint!(
        boundary_global,
        count(face -> face.axis == axis, mesh.boundary_faces),
    )
    for (global_index, face) in pairs(mesh.faces)
        face.axis == axis || continue
        push!(interior_global, global_index)
        # The minus cell is the deterministic owner. The other input can be in
        # a neighbouring block, as it must be for every face-based stencil.
        push!(owners, face.minus.block)
    end
    for (global_index, face) in pairs(mesh.boundary_faces)
        face.axis == axis || continue
        push!(boundary_global, global_index)
        push!(owners, face.cell.block)
    end
    return ComponentFacePlan(
        interior_global, boundary_global,
        BlockIndexGroups(owners, nblocks(mesh)),
    )
end

@inline function _component_face_average!(
        destination::FaceField, source::CellField,
        plan::ComponentFacePlan, output::Int,
    )
    mesh = source.mesh
    number_of_interior = length(plan.interior_global)
    if output <= number_of_interior
        global_index = @inbounds plan.interior_global[output]
        face = @inbounds mesh.faces[global_index]
        hm = @inbounds mesh.blocks[face.minus.block].spacing[face.axis]
        hp = @inbounds mesh.blocks[face.plus.block].spacing[face.axis]
        @inbounds destination.interior[output] = (
            hm * TP._get(source, face.minus) +
                hp * TP._get(source, face.plus)
        ) / (hm + hp)
    else
        boundary_output = output - number_of_interior
        global_index = @inbounds plan.boundary_global[boundary_output]
        face = @inbounds mesh.boundary_faces[global_index]
        @inbounds destination.boundary[boundary_output] =
            TP._get(source, face.cell)
    end
    return nothing
end

function global_face_average!(
        destination::FaceField, source::CellField,
        plan::ComponentFacePlan,
    )
    number_of_outputs = length(plan.interior_global) +
        length(plan.boundary_global)
    for output in 1:number_of_outputs
        _component_face_average!(destination, source, plan, output)
    end
    return destination
end

function block_face_average!(
        destination::FaceField, source::CellField,
        plan::ComponentFacePlan,
    )
    groups = plan.groups
    for block in 1:nblocks(source.mesh)
        for pointer in groups.offsets[block]:(groups.offsets[block + 1] - 1)
            output = @inbounds groups.indices[pointer]
            _component_face_average!(destination, source, plan, output)
        end
    end
    return destination
end

function global_predicted_fraction!(incoming, fraction, increment, cells)
    for cell in eachindex(cells)
        ref = @inbounds cells[cell]
        TP._set!(
            incoming,
            clamp(
                TP._get(fraction, ref) + TP._get(increment, ref),
                zero(eltype(incoming)), one(eltype(incoming)),
            ),
            ref,
        )
    end
    return incoming
end

function block_predicted_fraction!(incoming, fraction, increment)
    mesh = incoming.mesh
    nx, ny = mesh.block_size
    halo = mesh.halo
    for block in eachindex(mesh.blocks)
        destination = @inbounds incoming.blocks[block]
        old = @inbounds fraction.blocks[block]
        change = @inbounds increment.blocks[block]
        @inbounds for j in 1:ny, i in 1:nx
            destination[i + halo, j + halo] = clamp(
                old[i + halo, j + halo] + change[i + halo, j + halo],
                zero(eltype(incoming)), one(eltype(incoming)),
            )
        end
    end
    return incoming
end

function global_dilation_product!(incoming, indicator, outgoing, density, cells)
    for cell in eachindex(cells)
        ref = @inbounds cells[cell]
        chi = TP._get(indicator, ref)
        rhochi = density[1] * chi + density[2] * (1 - chi)
        TP._set!(incoming, rhochi * TP._get(outgoing, ref), ref)
    end
    return incoming
end

function block_dilation_product!(incoming, indicator, outgoing, density)
    mesh = incoming.mesh
    nx, ny = mesh.block_size
    halo = mesh.halo
    for block in eachindex(mesh.blocks)
        destination = @inbounds incoming.blocks[block]
        chi_block = @inbounds indicator.blocks[block]
        outgoing_block = @inbounds outgoing.blocks[block]
        @inbounds for j in 1:ny, i in 1:nx
            chi = chi_block[i + halo, j + halo]
            rhochi = density[1] * chi + density[2] * (1 - chi)
            destination[i + halo, j + halo] =
                rhochi * outgoing_block[i + halo, j + halo]
        end
    end
    return incoming
end

@inline function _momentum_delta(
        output, component, work, axis, density, ctu_threshold,
        scheme, amr_interface_scheme, oldc, newc, values,
    )
    schedule = component.advection.schedule
    value = zero(eltype(values))
    for (rho, flux, liquid) in (
            (density[1], work.liquid_flux, true),
            (density[2], work.gas_flux, false),
        )
        for pointer in component.advection.dual_offsets[output]:(
                component.advection.dual_offsets[output + 1] - 1
            )
            face_index = @inbounds component.advection.dual_faces[pointer]
            face = @inbounds schedule.faces[face_index]
            face.axis == axis || continue
            m, p = face.minus, face.plus
            cm, cp = liquid ? (oldc[m], oldc[p]) :
                (1 - oldc[m], 1 - oldc[p])
            nm, np = liquid ? (newc[m], newc[p]) :
                (1 - newc[m], 1 - newc[p])
            _, minus_difference, plus_difference = TP._phase_transport(
                face, schedule, flux, work.mesh, values,
                cm, cp, nm, np, ctu_threshold,
                scheme, amr_interface_scheme,
            )
            sign = @inbounds component.advection.dual_signs[pointer]
            value += rho * sign * (
                sign > 0 ? minus_difference : plus_difference
            ) / TP._phase_dual_volume(schedule, output)
        end
    end
    return value
end

function global_dual_transport!(
        delta, component, work, axis, density, ctu_threshold,
        scheme, amr_interface_scheme, oldc, newc, values,
    )
    for output in eachindex(values)
        @inbounds delta[output] = _momentum_delta(
            output, component, work, axis, density, ctu_threshold,
            scheme, amr_interface_scheme, oldc, newc, values,
        )
    end
    return delta
end

function block_dual_transport!(
        delta, component, work, axis, density, ctu_threshold,
        scheme, amr_interface_scheme, oldc, newc, values,
        groups::BlockIndexGroups,
    )
    for block in 1:nblocks(work.mesh)
        for pointer in groups.offsets[block]:(groups.offsets[block + 1] - 1)
            output = @inbounds groups.indices[pointer]
            @inbounds delta[output] = _momentum_delta(
                output, component, work, axis, density, ctu_threshold,
                scheme, amr_interface_scheme, oldc, newc, values,
            )
        end
    end
    return delta
end

function prepare_dual_inputs!(component, work, interface, face_plan, cells)
    global_face_average!(component.fraction, interface.fraction, face_plan)
    global_predicted_fraction!(
        work.incoming, interface.fraction, work.increment, cells,
    )
    global_face_average!(component.predicted_fraction, work.incoming, face_plan)
    oldc = TP._pack_phase_faces!(component.old_fraction, component.fraction)
    newc = TP._pack_phase_faces!(
        component.new_fraction, component.predicted_fraction,
    )
    values = TP._pack_phase_faces!(component.values, component.value)
    return oldc, newc, values
end

function global_scheduled_component!(
        component, work, interface, axis, density,
        scheme, amr_interface_scheme, face_plan, cells,
    )
    global_face_average!(component.fraction, interface.fraction, face_plan)
    global_predicted_fraction!(
        work.incoming, interface.fraction, work.increment, cells,
    )
    global_face_average!(component.predicted_fraction, work.incoming, face_plan)
    oldc = TP._pack_phase_faces!(component.old_fraction, component.fraction)
    newc = TP._pack_phase_faces!(
        component.new_fraction, component.predicted_fraction,
    )
    values = TP._pack_phase_faces!(component.values, component.value)
    sources = TP._pack_phase_faces!(component.sources, component.source)
    global_dual_transport!(
        component.delta, component, work, axis, density,
        work.ctu_threshold, scheme, amr_interface_scheme,
        oldc, newc, values,
    )
    global_dilation_product!(
        work.incoming, work.indicator, work.outgoing, density, cells,
    )
    global_face_average!(component.advection.rates, work.incoming, face_plan)
    rates = TP._pack_phase_faces!(component.rates, component.advection.rates)
    for output in eachindex(values)
        @inbounds begin
            rhonew = density[1] * newc[output] + density[2] * (1 - newc[output])
            values[output] += (
                component.delta[output] +
                    rates[output] * (sources[output] - values[output])
            ) / rhonew
        end
    end
    TP._unpack_phase_faces!(component.value, values)
    return component
end

function block_component!(
        component, work, interface, axis, density,
        scheme, amr_interface_scheme, face_plan,
    )
    block_face_average!(component.fraction, interface.fraction, face_plan)
    block_predicted_fraction!(work.incoming, interface.fraction, work.increment)
    block_face_average!(component.predicted_fraction, work.incoming, face_plan)
    oldc = TP._pack_phase_faces!(component.old_fraction, component.fraction)
    newc = TP._pack_phase_faces!(
        component.new_fraction, component.predicted_fraction,
    )
    values = TP._pack_phase_faces!(component.values, component.value)
    sources = TP._pack_phase_faces!(component.sources, component.source)
    block_dual_transport!(
        component.delta, component, work, axis, density,
        work.ctu_threshold, scheme, amr_interface_scheme,
        oldc, newc, values, face_plan.groups,
    )
    block_dilation_product!(work.incoming, work.indicator, work.outgoing, density)
    block_face_average!(component.advection.rates, work.incoming, face_plan)
    rates = TP._pack_phase_faces!(component.rates, component.advection.rates)
    # This loop only touches dense face vectors. Contiguous global order is the
    # locality-preserving traversal for this part of the block-major variant.
    for output in eachindex(values)
        @inbounds begin
            rhonew = density[1] * newc[output] + density[2] * (1 - newc[output])
            values[output] += (
                component.delta[output] +
                    rates[output] * (sources[output] - values[output])
            ) / rhonew
        end
    end
    TP._unpack_phase_faces!(component.value, values)
    return component
end

function block_cell_component!(
        component, work, interface, axis, density,
        scheme, amr_interface_scheme,
    )
    # Use block-local loops only where the underlying data are block-backed
    # matrices. Face fields and dual schedules are flat vectors, for which a
    # global contiguous traversal is the natural local order.
    TP._phase_face_average!(component.fraction, interface.fraction)
    block_predicted_fraction!(work.incoming, interface.fraction, work.increment)
    TP._phase_face_average!(component.predicted_fraction, work.incoming)
    oldc = TP._pack_phase_faces!(component.old_fraction, component.fraction)
    newc = TP._pack_phase_faces!(
        component.new_fraction, component.predicted_fraction,
    )
    values = TP._pack_phase_faces!(component.values, component.value)
    sources = TP._pack_phase_faces!(component.sources, component.source)
    delta = component.delta
    schedule = component.advection.schedule
    # Keep this loop text identical to the production implementation so this
    # variant changes only the two CellField traversals.
    for output in eachindex(values)
        value = zero(eltype(delta))
        for (rho, flux, liquid) in (
                (density[1], work.liquid_flux, true),
                (density[2], work.gas_flux, false),
            )
            for pointer in component.advection.dual_offsets[output]:(
                    component.advection.dual_offsets[output + 1] - 1
                )
                face_index = @inbounds component.advection.dual_faces[pointer]
                face = schedule.faces[face_index]
                face.axis == axis || continue
                m, p = face.minus, face.plus
                cm, cp = liquid ? (oldc[m], oldc[p]) :
                    (1 - oldc[m], 1 - oldc[p])
                nm, np = liquid ? (newc[m], newc[p]) :
                    (1 - newc[m], 1 - newc[p])
                _, minus_difference, plus_difference = TP._phase_transport(
                    face, schedule, flux, work.mesh, values,
                    cm, cp, nm, np, work.ctu_threshold,
                    scheme, amr_interface_scheme,
                )
                sign = @inbounds component.advection.dual_signs[pointer]
                value += rho * sign * (
                    sign > 0 ? minus_difference : plus_difference
                ) / TP._phase_dual_volume(schedule, output)
            end
        end
        @inbounds delta[output] = value
    end
    block_dilation_product!(work.incoming, work.indicator, work.outgoing, density)
    TP._phase_face_average!(component.advection.rates, work.incoming)
    rates = TP._pack_phase_faces!(component.rates, component.advection.rates)
    for output in eachindex(values)
        @inbounds begin
            rhonew = density[1] * newc[output] + density[2] * (1 - newc[output])
            values[output] += (
                component.delta[output] +
                    rates[output] * (sources[output] - values[output])
            ) / rhonew
        end
    end
    TP._unpack_phase_faces!(component.value, values)
    return component
end

function uniform_mesh()
    return BlockTreeMesh(
        block_size = (8, 8), max_level = 5, periodic = (true, false),
        refine = (key, origin, extent) -> key.level < 5,
    )
end

function mixed_mesh()
    return BlockTreeMesh(
        block_size = (8, 8), max_level = 6, periodic = (true, false),
        refine = (key, origin, extent) -> key.level < 3 || (
            key.level < 6 &&
                hypot(
                origin[1] + extent[1] / 2 - 0.5,
                origin[2] + extent[2] / 2 - 0.52,
            ) < 0.2 + maximum(extent)
        ),
    )
end

function measure!(
        rows, case_name, kernel, traversal, operation, checksum;
        prepare = () -> nothing,
    )
    prepare()
    operation() # Compile before timing.
    times = Vector{Float64}(undef, SAMPLES)
    for sample in eachindex(times)
        prepare()
        GC.gc(false)
        start = time_ns()
        operation()
        times[sample] = time_ns() - start
    end
    prepare()
    GC.gc(false)
    bytes = @allocated operation()
    prepare()
    allocations = @allocations operation()
    value = Float64(checksum())
    elapsed = median(times)
    push!(
        rows,
        (;
            case = case_name, kernel, traversal,
            time_ns = elapsed, minimum_ns = minimum(times),
            bytes, allocations, checksum = value,
        ),
    )
    return nothing
end

function assert_same(label, first, second)
    first == second || error("$label traversals produced different results")
    return nothing
end

function benchmark_case!(rows, case_name, mesh)
    println(
        case_name, ": ", nblocks(mesh), " blocks, ", ncells(mesh),
        " cells, ", length(mesh.faces), " interior faces",
    )
    levelset = (x, y) -> y - 0.52 - 0.035cospi(4x)
    interface = InterfaceField(mesh, levelset; lipschitz = 1.2, atol = 1.0e-9)
    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> 0.12 * (
            axis == 1 ? sinpi(2x)^2 * sinpi(2y) :
                -sinpi(2x) * sinpi(2y)^2
        ),
    )
    work = WYAdvectionWorkspace(interface)
    cells = TP._cell_references(mesh)
    dt = stable_timestep(velocity, work; cfl = 0.2)
    # Populate genuine geometric phase fluxes and WY dilation data. This setup
    # is deliberately outside every timed region.
    advect!(interface, velocity, dt, work)

    density = (1000.0, 1.0)
    axis = 2 # The retained WY work arrays belong to the final y sweep.
    scheme = MC()
    amr_interface_scheme = default_amr_interface_scheme(scheme)
    global_workspace = MomentumAdvectionWorkspace(interface)
    block_workspace = MomentumAdvectionWorkspace(interface)
    block_cell_workspace = MomentumAdvectionWorkspace(interface)
    production_workspace = MomentumAdvectionWorkspace(interface)
    for workspace in (
            global_workspace, block_workspace, block_cell_workspace,
            production_workspace,
        )
        TP._prepare_momentum_advection!(workspace, velocity)
    end
    component_global = global_workspace.x
    component_block = block_workspace.x
    component_block_cell = block_cell_workspace.x
    component_production = production_workspace.x
    face_plan = ComponentFacePlan(mesh, 1)

    # 1. CellField update: flattened CellRef traversal versus explicit blocks.
    measure!(
        rows, case_name, "predicted_cell_fraction", "global_cellrefs",
        () -> global_predicted_fraction!(
            work.incoming, interface.fraction, work.increment, cells,
        ),
        () -> sum(sum, work.incoming.blocks),
    )
    reference_cell = copy(work.incoming)
    measure!(
        rows, case_name, "predicted_cell_fraction", "block_nested",
        () -> block_predicted_fraction!(
            work.incoming, interface.fraction, work.increment,
        ),
        () -> sum(sum, work.incoming.blocks),
    )
    assert_same("predicted-cell", reference_cell.blocks, work.incoming.blocks)

    # 2. Centred-to-staggered interpolation. The indexed global traversal and
    # block traversal use the same precomputed axis-specific face plan.
    global_faces = FaceField{1}(mesh)
    block_faces = FaceField{1}(mesh)
    measure!(
        rows, case_name, "phase_face_average", "production_global_faces",
        () -> TP._phase_face_average!(
            component_production.fraction, interface.fraction,
        ),
        () -> sum(component_production.fraction.interior) +
            sum(component_production.fraction.boundary),
    )
    measure!(
        rows, case_name, "phase_face_average", "indexed_global_faces",
        () -> global_face_average!(global_faces, interface.fraction, face_plan),
        () -> sum(global_faces.interior) + sum(global_faces.boundary),
    )
    measure!(
        rows, case_name, "phase_face_average", "block_faces",
        () -> block_face_average!(block_faces, interface.fraction, face_plan),
        () -> sum(block_faces.interior) + sum(block_faces.boundary),
    )
    assert_same("face-average interior", global_faces.interior, block_faces.interior)
    assert_same("face-average boundary", global_faces.boundary, block_faces.boundary)
    assert_same(
        "production face-average interior",
        component_production.fraction.interior, block_faces.interior,
    )
    assert_same(
        "production face-average boundary",
        component_production.fraction.boundary, block_faces.boundary,
    )

    # 3. The expensive compatible dual-volume transport gather.
    oldc, newc, values = prepare_dual_inputs!(
        component_global, work, interface, face_plan, cells,
    )
    copyto!(component_block.old_fraction, oldc)
    copyto!(component_block.new_fraction, newc)
    copyto!(component_block.values, values)
    measure!(
        rows, case_name, "dual_phase_transport", "global_outputs",
        () -> global_dual_transport!(
            component_global.delta, component_global, work, axis, density,
            work.ctu_threshold, scheme, amr_interface_scheme,
            component_global.old_fraction, component_global.new_fraction,
            component_global.values,
        ),
        () -> sum(component_global.delta),
    )
    measure!(
        rows, case_name, "dual_phase_transport", "block_outputs",
        () -> block_dual_transport!(
            component_block.delta, component_block, work, axis, density,
            work.ctu_threshold, scheme, amr_interface_scheme,
            component_block.old_fraction, component_block.new_fraction,
            component_block.values, face_plan.groups,
        ),
        () -> sum(component_block.delta),
    )
    assert_same(
        "dual phase transport", component_global.delta, component_block.delta,
    )

    # 4. One complete compatible momentum-component update. Resetting the
    # advected/source fields remains outside the measured operation.
    function measure_component!(traversal, workspace, operation)
        component = workspace.x
        measure!(
            rows, case_name, "momentum_component", traversal,
            operation,
            () -> sum(component.value.interior) + sum(component.value.boundary),
            prepare = () -> TP._prepare_momentum_advection!(workspace, velocity),
        )
        return copy(component.value)
    end
    production_result = measure_component!(
        "production_block_cells", production_workspace,
        () -> TP._advect_momentum_component!(
            component_production, work, interface, axis, density,
            work.ctu_threshold, scheme, amr_interface_scheme,
        ),
    )
    global_result = measure_component!(
        "indexed_global", global_workspace,
        () -> global_scheduled_component!(
            component_global, work, interface, axis, density,
            scheme, amr_interface_scheme, face_plan, cells,
        ),
    )
    block_result = measure_component!(
        "block_major", block_workspace,
        () -> block_component!(
            component_block, work, interface, axis, density,
            scheme, amr_interface_scheme, face_plan,
        ),
    )
    block_cell_result = measure_component!(
        "block_cells_flat_faces", block_cell_workspace,
        () -> block_cell_component!(
            component_block_cell, work, interface, axis, density,
            scheme, amr_interface_scheme,
        ),
    )
    assert_same(
        "production/global component interior",
        production_result.interior, global_result.interior,
    )
    assert_same(
        "production/global component boundary",
        production_result.boundary, global_result.boundary,
    )
    assert_same(
        "global/block component interior",
        global_result.interior, block_result.interior,
    )
    assert_same(
        "global/block component boundary",
        global_result.boundary, block_result.boundary,
    )
    assert_same(
        "global/block-cell component interior",
        global_result.interior, block_cell_result.interior,
    )
    assert_same(
        "global/block-cell component boundary",
        global_result.boundary, block_cell_result.boundary,
    )
    return nothing
end

function write_results(path, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(
            io,
            "case,kernel,traversal,time_ns,minimum_ns,bytes,allocations,checksum",
        )
        for row in rows
            println(
                io, row.case, ',', row.kernel, ',', row.traversal, ',',
                row.time_ns, ',', row.minimum_ns, ',', row.bytes, ',',
                row.allocations, ',', row.checksum,
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
    println("Building ", case_name, " locality case ...")
    benchmark_case!(rows, case_name, factory())
end

println("\nMedian serial times (", SAMPLES, " samples):")
for row in rows
    println(
        rpad("$(row.case)/$(row.kernel)/$(row.traversal)", 66),
        lpad(string(round(row.time_ns / 1.0e6; digits = 3)), 10), " ms  ",
        row.bytes, " B / ", row.allocations, " alloc",
    )
end

output_path = isempty(ARGS) ?
    joinpath(@__DIR__, "data-locality-results", "staggered-two-phase.csv") :
    abspath(ARGS[1])
write_results(output_path, rows)
println("Wrote ", output_path)

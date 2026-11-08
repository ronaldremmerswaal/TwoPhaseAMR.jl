using Test, TwoPhaseAMR

# Independent pre-cache curl reference: allocating cell neighbours and fit
# arrays deliberately remain here to detect donor/order/mask changes.
function _uncached_curl_reference(velocity; interface = nothing, phase = :liquid)
    mesh = velocity.mesh
    nx, ny = mesh.block_size
    count = ncells(mesh)
    index(cell) = (cell.block - 1) * nx * ny + (cell.j - 1) * nx + cell.i
    weighted, measures = zeros(4, count), zeros(4, count)
    for (f, face) in pairs(mesh.faces)
        isnothing(interface) || TwoPhaseAMR._phase_face_is_active(interface, f, face, phase) || continue
        value = velocity.interior[f]
        isnan(value) && continue
        for (cell, side) in ((face.minus, 1), (face.plus, -1))
            slot = 2face.axis - (side < 0 ? 1 : 0)
            weighted[slot, index(cell)] += value * face.area
            measures[slot, index(cell)] += face.area
        end
    end
    for (f, face) in pairs(mesh.boundary_faces)
        isnothing(interface) || TwoPhaseAMR._phase_cell_is_active(interface, face.cell, phase) || continue
        value = velocity.boundary[f]
        isnan(value) && continue
        slot = 2face.axis - (face.side < 0 ? 1 : 0)
        weighted[slot, index(face.cell)] += value * face.area
        measures[slot, index(face.cell)] += face.area
    end
    x, y = zeros(count), zeros(count)
    centers = NTuple{2, Float64}[]
    for b in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        c = CellRef(b, i, j)
        k = index(c)
        push!(centers, cellcenter(mesh, b, i, j))
        present = isnothing(interface) || TwoPhaseAMR._phase_cell_is_active(interface, c, phase)
        for (component, first_slot) in ((x, 1), (y, 3))
            total = 0.0; sides = 0
            for slot in first_slot:(first_slot + 1)
                measures[slot, k] > 0 || continue
                total += weighted[slot, k] / measures[slot, k]
                sides += 1
            end
            component[k] = present && sides > 0 ? total / sides : NaN
        end
    end
    known = isfinite.(x) .& isfinite.(y)
    neighbours = [Int[] for _ in 1:count]
    for face in mesh.faces
        minus, plus = index(face.minus), index(face.plus)
        push!(neighbours[minus], plus); push!(neighbours[plus], minus)
    end
    for target in eachindex(x)
        known[target] && continue
        any(k -> known[k], neighbours[target]) || continue
        candidates = Int[]
        for n in neighbours[target]
            push!(candidates, n); append!(candidates, neighbours[n])
        end
        sort!(candidates); unique!(candidates)
        sw = sx = sy = sxx = sxy = syy = 0.0
        su = sv = sxu = syu = sxv = syv = 0.0
        for donor in candidates
            known[donor] || continue
            dx = TwoPhaseAMR._wrapped_delta(centers[donor][1] - centers[target][1], mesh.extent[1], mesh.periodic[1])
            dy = TwoPhaseAMR._wrapped_delta(centers[donor][2] - centers[target][2], mesh.extent[2], mesh.periodic[2])
            distance_squared = dx^2 + dy^2
            distance_squared > 0 || continue
            w = inv(distance_squared)
            u, v = x[donor], y[donor]
            sw += w; sx += w * dx; sy += w * dy
            sxx += w * dx^2; sxy += w * dx * dy; syy += w * dy^2
            su += w * u; sv += w * v
            sxu += w * dx * u; syu += w * dy * u
            sxv += w * dx * v; syv += w * dy * v
        end
        iszero(sw) && continue
        mx, my = sx / sw, sy / sw
        u, v = su / sw, sv / sw
        a, b, c = sxx - sx * mx, sxy - sx * my, syy - sy * my
        determinant = a * c - b^2
        if determinant > 128eps() * max(abs(a * c), b^2)
            ux, uy = sxu - sx * u, syu - sy * u
            vx, vy = sxv - sx * v, syv - sy * v
            u -= ((c * ux - b * uy) * mx + (a * uy - b * ux) * my) / determinant
            v -= ((c * vx - b * vy) * mx + (a * vy - b * vx) * my) / determinant
        end
        x[target], y[target] = u, v
    end
    a11, a12, a22, bx1, bx2, by1, by2 = ntuple(_ -> zeros(count), 7)
    for face in mesh.faces
        minus, plus = index(face.minus), index(face.plus)
        all(isfinite, (x[minus], x[plus], y[minus], y[plus])) || continue
        dx = TwoPhaseAMR._wrapped_delta(centers[plus][1] - centers[minus][1], mesh.extent[1], mesh.periodic[1])
        dy = TwoPhaseAMR._wrapped_delta(centers[plus][2] - centers[minus][2], mesh.extent[2], mesh.periodic[2])
        distance_squared = dx^2 + dy^2
        distance_squared > 0 || continue
        w = face.area / distance_squared
        du, dv = x[plus] - x[minus], y[plus] - y[minus]
        for k in (minus, plus)
            a11[k] += w * dx^2; a12[k] += w * dx * dy; a22[k] += w * dy^2
            bx1[k] += w * du * dx; bx2[k] += w * du * dy
            by1[k] += w * dv * dx; by2[k] += w * dv * dy
        end
    end
    result = fill(NaN, count)
    for k in eachindex(result)
        known[k] || continue
        determinant = a11[k] * a22[k] - a12[k]^2
        determinant <= 128eps() * max(abs(a11[k] * a22[k]), a12[k]^2) && continue
        result[k] = (a22[k] * by1[k] - a12[k] * by2[k]) / determinant -
            (a11[k] * bx2[k] - a12[k] * bx1[k]) / determinant
    end
    return result
end

function _uncached_phase_transfer!(velocity)
    mesh = velocity.mesh
    for axis in 1:2
        field = axis == 1 ? FaceField{1}(velocity) : FaceField{2}(velocity)
        schedule = all(mesh.periodic) ?
            TwoPhaseAMR._dual_advection_schedule(mesh, Val(axis), Float64, true) :
            TwoPhaseAMR._bounded_phase_schedule(mesh, Val(axis), Float64)
        work = StaggeredAdvectionWorkspace(FaceField{axis}(mesh), FaceField{axis}(mesh), schedule, Ref(true))
        values = vcat(field.interior, field.boundary)
        TwoPhaseAMR._extend_phase_transfer_values!(values, work, mesh, axis)
        TwoPhaseAMR._unpack_phase_faces!(field, values)
        copyto!(velocity, field)
    end
    return velocity
end

@testset "Shared AMR connectivity preserves numerical kernels" begin
    for periodic in ((true, true), (true, false), (false, false)), mixed in (false, true), solid in (false, true)
        solid && any(periodic) && continue
        mesh = BlockTreeMesh(block_size = (6, 6), max_level = 2, periodic = periodic,
            refine = (key, origin, extent) -> mixed &&
                (key.level == 0 || (key.level == 1 && key.i == key.j == 0)))
        geometry = solid ? GeometryField(mesh, (x, y) -> x + 0.15y - 0.18) : nothing
        interface = InterfaceField(mesh, (x, y) -> y - 0.46 - 0.1sinpi(2x); geometry)
        velocity = (FaceVelocity(mesh, (x, y, t, a) -> sinpi(2x) + a * cospi(2y)),
            FaceVelocity(mesh, (x, y, t, a) -> 0.3x - 0.7y + a))
        TwoPhaseAMR._mark_inactive_phase_velocities!(velocity, interface)
        before = map(copy, velocity)
        scratch = TwoPhaseAMR.PhaseTransferScratch(Float64)
        curlwork = TwoPhaseAMR.VorticityWorkspace(Float64)
        for (phase, name) in ((1, :liquid), (2, :gas))
            expected = _uncached_phase_transfer!(copy(velocity[phase]))
            actual = copy(velocity[phase])
            TwoPhaseAMR._extend_phase_velocity_for_transfer!(actual, interface, scratch)
            @test isequal(actual.interior, expected.interior)
            @test isequal(actual.boundary, expected.boundary)
            expected_curl = _uncached_curl_reference(velocity[phase]; interface, phase = name)
            actual_curl = TwoPhaseAMR._vorticity_values!(curlwork, velocity[phase]; interface, phase = name)
            @test isequal(actual_curl, expected_curl)
            @test isequal(visualization_values(velocity[phase], Vorticity(); interface, phase = name), expected_curl)
            @test isequal(velocity[phase].interior, before[phase].interior)
            @test isequal(velocity[phase].boundary, before[phase].boundary)
        end
        for axis in 1:2
            graph = TwoPhaseAMR._phase_transport_connectivity(mesh, Val(axis), Float64)
            @test graph === TwoPhaseAMR._phase_transport_connectivity(mesh, Val(axis), Float64)
            raw = all(periodic) ? TwoPhaseAMR._dual_advection_schedule(mesh, Val(axis), Float64, true) :
                TwoPhaseAMR._bounded_phase_schedule(mesh, Val(axis), Float64)
            reference = StaggeredAdvectionWorkspace(FaceField{axis}(mesh), FaceField{axis}(mesh), raw, Ref(true))
            @test graph.schedule.faces == raw.faces
            @test graph.dual_offsets == reference.dual_offsets
            @test graph.dual_faces == reference.dual_faces
            @test graph.dual_signs == reference.dual_signs
            @test graph.boundary_faces == reference.boundary_faces
        end
        # Connectivity can be shared; evolving numerical/BC data cannot.
        q1, q2 = PhaseFaceScalar{1}(interface), PhaseFaceScalar{1}(interface)
        @test q1.workspace.dual_flux !== q2.workspace.dual_flux
        @test !hasproperty(q1.workspace, :boundary_conditions)
        if !solid
            @test q1.workspace.schedule === q2.workspace.schedule
            @test q1.workspace.dual_offsets === q2.workspace.dual_offsets
        else
            @test q1.workspace.schedule !== q2.workspace.schedule
            graph = TwoPhaseAMR._phase_transport_connectivity(mesh, Val(1), Float64)
            @test q1.workspace.schedule !== graph.schedule
        end
        first = visualization_values(velocity[1], Vorticity(); interface)
        second = visualization_values(velocity[1], Vorticity(); interface)
        @test first !== second
    end
end

@testset "Connectivity epochs include same-size AMR changes" begin
    mesh = BlockTreeMesh(block_size = (4, 4), max_level = 2, periodic = (false, false),
        adaptation = MeshAdaptationOptions(coarsen_delay = 0),
        refine = (key, origin, extent) -> key.level == 0)
    scalar = CellField(mesh, (x, y) -> x + y)
    work = AMRWorkspace(mesh, scalar)
    graph = TwoPhaseAMR._phase_transport_connectivity(mesh, Val(1), Float64)
    cells = TwoPhaseAMR._mesh_cell_connectivity(mesh)
    @test !adapt!(work, fill(:keep, nblocks(mesh)); step = 1, force = true).changed
    @test graph === TwoPhaseAMR._phase_transport_connectivity(mesh, Val(1), Float64)
    @test cells === TwoPhaseAMR._mesh_cell_connectivity(mesh)
    labels = [block.key.i == block.key.j == 0 ? :refine : :keep for block in mesh.blocks]
    @test adapt!(work, labels; step = 2, force = true).changed
    graph = TwoPhaseAMR._phase_transport_connectivity(mesh, Val(1), Float64)
    cells = TwoPhaseAMR._mesh_cell_connectivity(mesh)
    sizes = (nblocks(mesh), length(mesh.faces), length(mesh.boundary_faces))
    labels = [block.key.level == 2 ? :coarsen :
        block.key.i == 1 && block.key.j == 0 ? :refine : :keep for block in mesh.blocks]
    @test adapt!(work, labels; step = 3, force = true).changed
    @test sizes == (nblocks(mesh), length(mesh.faces), length(mesh.boundary_faces))
    updated = TwoPhaseAMR._phase_transport_connectivity(mesh, Val(1), Float64)
    @test updated !== graph
    @test updated.generation == graph.generation + 1
    @test cells !== TwoPhaseAMR._mesh_cell_connectivity(mesh)
    @test all(updated.centers[k] == mesh.faces[f].center for (k, f) in pairs(updated.interior_indices))
end

@testset "Moving phase masks do not invalidate or contaminate mesh connectivity" begin
    mesh = BlockTreeMesh(block_size = (6, 6), max_level = 1, periodic = (false, false),
        refine = (key, origin, extent) -> key.level == 0)
    interface = InterfaceField(mesh)
    graph = TwoPhaseAMR._phase_transport_connectivity(mesh, Val(1), Float64)
    cells = TwoPhaseAMR._mesh_cell_connectivity(mesh)
    transfer = TwoPhaseAMR.PhaseTransferScratch(Float64)
    curlwork = TwoPhaseAMR.VorticityWorkspace(Float64)
    for intercept in (0.23, 0.51, 0.79)
        initialize!(interface, (0.0, 1.0), intercept)
        velocity = (FaceVelocity(mesh, (x, y, t, a) -> a == 1 ? y^2 : x^2),
            FaceVelocity(mesh, (x, y, t, a) -> a == 1 ? -y : 2x))
        TwoPhaseAMR._mark_inactive_phase_velocities!(velocity, interface)
        for (phase, name) in ((1, :liquid), (2, :gas))
            reference = _uncached_phase_transfer!(copy(velocity[phase]))
            candidate = copy(velocity[phase])
            TwoPhaseAMR._extend_phase_velocity_for_transfer!(candidate, interface, transfer)
            @test isequal(candidate.interior, reference.interior)
            @test isequal(candidate.boundary, reference.boundary)
            expected = _uncached_curl_reference(velocity[phase]; interface, phase = name)
            @test isequal(TwoPhaseAMR._vorticity_values!(curlwork, velocity[phase]; interface, phase = name), expected)
            indicator = VorticityRefinementIndicator(velocity[phase]; interface, phase = name,
                refine_tolerance = 0.05, coarsen_tolerance = 0.01)
            labels = fill(:keep, nblocks(mesh))
            refinement_labels!(labels, indicator; _vorticity_workspace = curlwork)
            block_cells = prod(mesh.block_size)
            for (b, block) in pairs(mesh.blocks)
                error = 0.0
                unresolved = false
                for j in 1:mesh.block_size[2], i in 1:mesh.block_size[1]
                    TwoPhaseAMR._phase_cell_is_active(interface, CellRef(b, i, j), name) || continue
                    value = expected[(b - 1) * block_cells + (j - 1) * mesh.block_size[1] + i]
                    if isnan(value)
                        unresolved = true
                    else
                        error = max(error, maximum(block.spacing) * abs(value))
                    end
                end
                # These blocks are at the maximum level and above the
                # minimum: only the original coarsening rule can fire.
                @test labels[b] == (!unresolved && error < 0.01 ? :coarsen : :keep)
            end
        end
        @test graph === TwoPhaseAMR._phase_transport_connectivity(mesh, Val(1), Float64)
        @test cells === TwoPhaseAMR._mesh_cell_connectivity(mesh)
    end
end

@testset "Mesh-only connectivity respects numerical precision" begin
    mesh = BlockTreeMesh(block_size = (4, 4), periodic = (true, false),
        origin = (0.0f0, 0.0f0), extent = (1.0f0, 1.0f0))
    interface = InterfaceField(mesh; T = Float32)
    q = PhaseFaceScalar{1}(interface)
    graph32 = TwoPhaseAMR._phase_transport_connectivity(mesh, Val(1), Float32)
    @test q.workspace.schedule === graph32.schedule
    @test eltype(graph32.centers) == NTuple{2, Float32}
    @test eltype(TwoPhaseAMR._mesh_cell_connectivity(mesh).centers) == NTuple{2, Float32}
    graph64 = TwoPhaseAMR._phase_transport_connectivity(mesh, Val(1), Float64)
    @test graph64 !== graph32
    @test graph64 === TwoPhaseAMR._phase_transport_connectivity(mesh, Val(1), Float64)
    renewed32 = TwoPhaseAMR._phase_transport_connectivity(mesh, Val(1), Float32)
    @test renewed32 !== graph32
    @test renewed32.schedule.faces == graph32.schedule.faces
    @test q.workspace.schedule === graph32.schedule
    @test graph32.generation == graph64.generation == renewed32.generation
end

@testset "adaptive phase registry accepts heterogeneous interfaces" begin
    mesh = BlockTreeMesh(block_size = (4, 4), periodic = (false, false))
    plain = InterfaceField(mesh)
    geometry = GeometryField(mesh, (x, y) -> x - 0.3)
    cut = InterfaceField(mesh; geometry)
    plain_velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
    cut_velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
    workspace = AMRWorkspace(mesh)
    register_adaptive_field!(workspace, plain_velocity, plain)
    register_adaptive_field!(workspace, cut_velocity, cut)
    register_adaptive_field!(workspace, plain_velocity, plain)
    @test length(workspace.phase_velocities) == 2
    @test workspace.phase_velocities[1][3] === plain
    @test workspace.phase_velocities[2][3] === cut
    @test eltype(workspace.phase_velocities) !== Any
end

using Test
using TwoPhaseAMR

struct FrozenCellAdvectionWorkspace{T,M}
    increment::CellField{T,M}
    rates::CellField{T,M}
    interior_flux::Vector{T}
    boundary_flux::Vector{T}
    cell_offsets::Vector{Int}
    face_indices::Vector{Int}
    face_signs::Vector{Int8}
    boundary_entries::BitVector
end

function FrozenCellAdvectionWorkspace(field::CellField{T,M}) where {T,M}
    mesh = field.mesh
    nx, ny = mesh.block_size
    targets = Int[]
    face_indices = Int[]
    face_signs = Int8[]
    boundary_entries = BitVector()
    cell_number(ref) = (ref.block - 1) * nx * ny + (ref.j - 1) * nx + ref.i
    for (index, face) in pairs(mesh.faces)
        push!(targets, cell_number(face.minus))
        push!(face_indices, index)
        push!(face_signs, -1)
        push!(boundary_entries, false)
        push!(targets, cell_number(face.plus))
        push!(face_indices, index)
        push!(face_signs, 1)
        push!(boundary_entries, false)
    end
    for (index, face) in pairs(mesh.boundary_faces)
        push!(targets, cell_number(face.cell))
        push!(face_indices, index)
        push!(face_signs, -1)
        push!(boundary_entries, true)
    end
    permutation = sortperm(targets)
    offsets = Vector{Int}(undef, ncells(mesh) + 1)
    pointer = 1
    for cell in 1:ncells(mesh)
        offsets[cell] = pointer
        while pointer <= length(permutation) && targets[permutation[pointer]] == cell
            pointer += 1
        end
    end
    offsets[end] = pointer
    return FrozenCellAdvectionWorkspace(
        CellField(mesh, T), CellField(mesh, T),
        zeros(T, length(mesh.faces)), zeros(T, length(mesh.boundary_faces)),
        offsets, face_indices[permutation], face_signs[permutation],
        boundary_entries[permutation],
    )
end

# Frozen pre-compact flux path (30 September 2026): read owned face records,
# then gather the same signed stored fluxes in the workspace's CSR order.
function frozen_cell_advection_step!(scalar, velocity, dt, workspace;
                                     boundary=ZeroGradient(), time=0)
    mesh = scalar.mesh
    for (index, face) in pairs(mesh.faces)
        speed = velocity.interior[index]
        state = speed >= 0 ? TwoPhaseAMR._get(scalar, face.minus) :
                             TwoPhaseAMR._get(scalar, face.plus)
        workspace.interior_flux[index] = dt * speed * state * face.area
    end
    for (index, face) in pairs(mesh.boundary_faces)
        coordinate_speed = velocity.boundary[index]
        outward_speed = face.side * coordinate_speed
        state = outward_speed >= 0 ? TwoPhaseAMR._get(scalar, face.cell) :
            boundary isa ZeroGradient ? TwoPhaseAMR._get(scalar, face.cell) :
            boundary.value(face.center[1], face.center[2], time)
        workspace.boundary_flux[index] = dt * outward_speed * state * face.area
    end
    nx, ny = mesh.block_size
    h = mesh.halo
    cells_per_block = nx * ny
    for b in eachindex(mesh.blocks)
        scalar_block = scalar.blocks[b]
        increment_block = workspace.increment.blocks[b]
        volume = mesh.blocks[b].volume
        cell_offset = (b - 1) * cells_per_block
        for j in 1:ny, i in 1:nx
            cell = cell_offset + (j - 1) * nx + i
            integrated = zero(eltype(scalar))
            for pointer in workspace.cell_offsets[cell]:(workspace.cell_offsets[cell + 1] - 1)
                face_index = workspace.face_indices[pointer]
                flux = workspace.boundary_entries[pointer] ?
                    workspace.boundary_flux[face_index] :
                    workspace.interior_flux[face_index]
                integrated += workspace.face_signs[pointer] * flux
            end
            change = integrated / volume
            increment_block[i + h, j + h] = change
            scalar_block[i + h, j + h] += change
        end
    end
    scalar
end

function frozen_cell_cfl(velocity, workspace; cfl=0.9)
    mesh = velocity.mesh
    accum_type = promote_type(eltype(velocity), eltype(workspace.rates), typeof(cfl))
    for (index, face) in pairs(mesh.faces)
        workspace.interior_flux[index] = abs(velocity.interior[index]) * face.area
    end
    for (index, face) in pairs(mesh.boundary_faces)
        workspace.boundary_flux[index] = abs(velocity.boundary[index]) * face.area
    end
    nx, ny = mesh.block_size
    h = mesh.halo
    cells_per_block = nx * ny
    maxrate = zero(accum_type)
    for b in eachindex(mesh.blocks)
        volume = mesh.blocks[b].volume
        cell_offset = (b - 1) * cells_per_block
        for j in 1:ny, i in 1:nx
            cell = cell_offset + (j - 1) * nx + i
            integrated = zero(accum_type)
            for pointer in workspace.cell_offsets[cell]:(workspace.cell_offsets[cell + 1] - 1)
                face_index = workspace.face_indices[pointer]
                integrated += workspace.boundary_entries[pointer] ?
                    workspace.boundary_flux[face_index] :
                    workspace.interior_flux[face_index]
            end
            rate = integrated / volume
            workspace.rates.blocks[b][i + h, j + h] = rate
            maxrate = max(maxrate, workspace.rates.blocks[b][i + h, j + h])
        end
    end
    iszero(maxrate) ? accum_type(Inf) : accum_type(cfl) / maxrate
end

@testset "compact general face path matches frozen trajectory" begin
    meshes = (
        # The seam in a single periodic block connects distant array indices.
        BlockTreeMesh(base_mesh=(2, 2), block_size=(2, 2)),
        BlockTreeMesh(base_mesh=(2, 8), block_size=(2, 8),
            periodic=(true, false)),
        BlockTreeMesh(base_mesh=(8, 2), block_size=(8, 2),
            periodic=(false, true)),
        BlockTreeMesh(base_mesh=(16, 4), block_size=(2, 4)),
        BlockTreeMesh(base_mesh=(4, 16), block_size=(4, 2)),
        BlockTreeMesh(base_mesh=(16, 12), block_size=(16, 12)),
        BlockTreeMesh(base_mesh=(16, 12), block_size=(4, 4)),
        BlockTreeMesh(base_mesh=(16, 8), block_size=(8, 8), max_level=1,
            refine=(key, origin, extent)-> key.level == 0 && key.i == 0),
        BlockTreeMesh(base_mesh=(16, 8), block_size=(8, 8),
            origin=(-0.5, 0.25), extent=(2.0, 0.5)),
        BlockTreeMesh(base_mesh=(16, 12), block_size=(4, 4),
            periodic=(false, false)),
        # These four cases cross the package's 1,024-cell parallel grain.
        BlockTreeMesh(base_mesh=(64, 64), block_size=(64, 64)),
        BlockTreeMesh(base_mesh=(64, 64), block_size=(8, 8)),
        BlockTreeMesh(base_mesh=(64, 64), block_size=(8, 8), max_level=1,
            refine=(key, origin, extent)->key.level == 0 && key.i == 0 && key.j == 0),
        BlockTreeMesh(base_mesh=(64, 64), block_size=(8, 8),
            periodic=(false, false)),
    )
    for mesh in meshes, T in (Float32, Float64)
        scalar = CellField(mesh, T)
        initialize!(scalar, (x, y)->T(1 + 0.2sinpi(2x) - 0.1cospi(2y)))
        reference = copy(scalar)
        velocity = FaceVelocity(mesh, (x, y, t, axis)->
            T(axis == 1 ? 0.13 + 0.31sinpi(2x + t) - 0.17cospi(2y) :
                          -0.11 + 0.24cospi(2y - t) - 0.19sinpi(2x)); T=T)
        candidate_ws = AdvectionWorkspace(scalar)
        reference_ws = FrozenCellAdvectionWorkspace(reference)
        initial_mass = integrate(scalar)
        boundary = Dirichlet((x, y, t)->T(0.8 + 0.1x - 0.2y + 0.3t))
        for step in 1:5
            t = (step - 1) * 0.01
            set_velocities!(velocity, (x, y, time, axis)->
                T(axis == 1 ? 0.13 + 0.31sinpi(2x + time) - 0.17cospi(2y) :
                              -0.11 + 0.24cospi(2y - time) - 0.19sinpi(2x));
                time=t)
            # Face IDs remain authoritative even when callers edit the
            # carrier after the workspace has been constructed.
            velocity.interior[firstindex(velocity.interior)] = T(-0.41 + 0.03step)
            velocity.interior[lastindex(velocity.interior)] = T(0.27 - 0.02step)
            if !isempty(velocity.boundary)
                velocity.boundary[firstindex(velocity.boundary)] = T(-0.36 + 0.04step)
            end
            for cfl in (Float32(0.6), Float64(0.6))
                observed_dt = stable_timestep(velocity, candidate_ws; cfl)
                @test isequal(observed_dt,
                              frozen_cell_cfl(velocity, reference_ws; cfl))
                @test typeof(observed_dt) === promote_type(T, typeof(cfl))
                for b in eachindex(mesh.blocks)
                    @test isequal(interior(candidate_ws.rates, b),
                                  interior(reference_ws.rates, b))
                end
            end
            step_velocity = velocity
            if T === Float32 && step == 1
                wider_velocity = FaceVelocity(mesh, Float64; init=0)
                wider_velocity.interior .= velocity.interior
                wider_velocity.boundary .= velocity.boundary
                observed_dt = stable_timestep(wider_velocity, candidate_ws;
                                              cfl=Float32(0.6))
                @test isequal(observed_dt,
                              frozen_cell_cfl(wider_velocity, reference_ws;
                                              cfl=Float32(0.6)))
                @test typeof(observed_dt) === Float64
                for b in eachindex(mesh.blocks)
                    @test isequal(interior(candidate_ws.rates, b),
                                  interior(reference_ws.rates, b))
                end
                step_velocity = wider_velocity
            end
            frozen_cell_advection_step!(reference, step_velocity, 0.01, reference_ws;
                                        boundary, time=t)
            advect!(scalar, step_velocity, 0.01, candidate_ws; boundary, time=t)
            for b in eachindex(mesh.blocks)
                @test isequal(interior(scalar, b), interior(reference, b))
                @test isequal(interior(candidate_ws.increment, b),
                              interior(reference_ws.increment, b))
            end
        end
        @test isequal(integrate(scalar), integrate(reference))
        if all(mesh.periodic)
            @test abs(integrate(scalar) - initial_mass) <=
                (T === Float32 ? 5e-6 : 2e-13) * prod(mesh.extent)
        end
        TwoPhaseAMR._invalidate_mesh_topology_cache!(mesh)
        @test_throws ArgumentError advect!(scalar, velocity, 0.01, candidate_ws)
        @test_throws ArgumentError stable_timestep(velocity, candidate_ws)
    end
end

function frozen_cut_cell_step!(scalar, velocity, dt, workspace, geometry;
                               boundary=ZeroGradient(), time=0)
    mesh = scalar.mesh
    for (index, face) in pairs(mesh.faces)
        speed = velocity.interior[index]
        state = speed >= 0 ? TwoPhaseAMR._get(scalar, face.minus) :
                             TwoPhaseAMR._get(scalar, face.plus)
        workspace.interior_flux[index] =
            dt * speed * state * face.area * geometry.aperture[index]
    end
    for (index, face) in pairs(mesh.boundary_faces)
        coordinate_speed = velocity.boundary[index]
        outward_speed = face.side * coordinate_speed
        state = outward_speed >= 0 ? TwoPhaseAMR._get(scalar, face.cell) :
            boundary isa ZeroGradient ? TwoPhaseAMR._get(scalar, face.cell) :
            boundary.value(face.center[1], face.center[2], time)
        workspace.boundary_flux[index] = dt * outward_speed * state *
            face.area * geometry.boundary_aperture[index]
    end
    nx, ny = mesh.block_size
    h = mesh.halo
    cells_per_block = nx * ny
    for b in eachindex(mesh.blocks)
        scalar_block = scalar.blocks[b]
        increment_block = workspace.increment.blocks[b]
        open = geometry.open_fraction.blocks[b]
        volume = mesh.blocks[b].volume
        cell_offset = (b - 1) * cells_per_block
        for j in 1:ny, i in 1:nx
            cell = cell_offset + (j - 1) * nx + i
            integrated = zero(eltype(scalar))
            for pointer in workspace.cell_offsets[cell]:(workspace.cell_offsets[cell + 1] - 1)
                face_index = workspace.face_indices[pointer]
                flux = workspace.boundary_entries[pointer] ?
                    workspace.boundary_flux[face_index] :
                    workspace.interior_flux[face_index]
                integrated += workspace.face_signs[pointer] * flux
            end
            fraction = open[i + h, j + h]
            change = fraction > zero(eltype(scalar)) ?
                integrated / (volume * fraction) : zero(eltype(scalar))
            increment_block[i + h, j + h] = change
            scalar_block[i + h, j + h] += change
        end
    end
    scalar
end

function cut_cell_amount(field, geometry)
    total = zero(eltype(field))
    nx, ny = field.mesh.block_size
    h = field.mesh.halo
    for (b, block) in pairs(field.mesh.blocks), j in 1:ny, i in 1:nx
        total += field.blocks[b][i+h,j+h] *
                 geometry.open_fraction.blocks[b][i+h,j+h] * block.volume
    end
    total
end

@testset "live owned-face values and closed cells after workspace construction" begin
    mesh = BlockTreeMesh(base_mesh=(8, 4), block_size=(8, 4),
                         periodic=(true, false))
    scalar = CellField(mesh, (x, y)->1 + 0.3x - 0.2y)
    reference = copy(scalar)
    velocity = FaceVelocity(mesh, Float64; init=0.0)
    geometry = GeometryField(mesh, (x, y)->x - 0.2; lipschitz=1)
    candidate_ws = AdvectionWorkspace(scalar)
    reference_ws = FrozenCellAdvectionWorkspace(reference)

    # Exercise the original owned-face IDs, including a same-block periodic
    # seam and physical inflow. Direct writes bypass set_velocities! entirely.
    seam = findfirst(face -> face.axis == 1 &&
        face.minus.block == face.plus.block && face.plus.i != face.minus.i + 1,
        mesh.faces)
    @test !isnothing(seam)
    @test !isempty(mesh.boundary_faces)
    for step in 1:3
        for index in eachindex(velocity.interior)
            velocity.interior[index] = (isodd(index + step) ? 0.23 : -0.17)
            geometry.aperture[index] = (index == seam ? 0.37 :
                                        isodd(index + step) ? 0.7 : 0.2)
        end
        for index in eachindex(velocity.boundary)
            velocity.boundary[index] = isodd(index + step) ? -0.31 : 0.19
            geometry.boundary_aperture[index] = isodd(index) ? 0.4 : 0.8
        end
        # An isolated closed cell must receive an exactly zero increment.
        closed = geometry.open_fraction.blocks[1]
        closed[mesh.halo + 3, mesh.halo + 2] = 0.0
        boundary = Dirichlet((x, y, t)->0.4 + x - 2y + t)
        frozen_cut_cell_step!(reference, velocity, 0.003, reference_ws,
                              geometry; boundary, time=0.1step)
        advect!(scalar, velocity, 0.003, candidate_ws, geometry;
                boundary, time=0.1step)
        @test isequal(interior(scalar, 1), interior(reference, 1))
        @test isequal(interior(candidate_ws.increment, 1),
                      interior(reference_ws.increment, 1))
        @test candidate_ws.increment.blocks[1][mesh.halo + 3, mesh.halo + 2] == 0.0
    end
end

@testset "shared cut-cell flux remap matches independent frozen incidence" begin
    mesh = BlockTreeMesh(base_mesh=(64, 64), block_size=(8, 8),
                         periodic=(false, false))
    geometry = GeometryField(mesh,
        (x, y)->(x - 0.5)^2 + (y - 0.5)^2 - 0.17^2;
        lipschitz=1)
    @test any(a -> 0 < a < 1, geometry.aperture)
    scalar = CellField(mesh, (x, y)->1 + 0.2sinpi(2x) - 0.1cospi(2y))
    reference = copy(scalar)
    velocity = FaceVelocity(mesh,
        (x, y, t, axis)->axis == 1 ? 0.2 + 0.1sinpi(2y) :
                                        -0.1 + 0.07cospi(2x))
    enforce_no_permeation!(velocity)
    candidate_ws = AdvectionWorkspace(scalar)
    reference_ws = FrozenCellAdvectionWorkspace(reference)
    initial_amount = cut_cell_amount(scalar, geometry)
    for step in 1:4
        frozen_cut_cell_step!(reference, velocity, 0.005, reference_ws,
                              geometry; time=(step-1)*0.005)
        advect!(scalar, velocity, 0.005, candidate_ws, geometry;
                time=(step-1)*0.005)
        for b in eachindex(mesh.blocks)
            @test isequal(interior(scalar,b), interior(reference,b))
            @test isequal(interior(candidate_ws.increment,b),
                          interior(reference_ws.increment,b))
        end
    end
    @test isequal(cut_cell_amount(scalar, geometry),
                  cut_cell_amount(reference, geometry))
    @test abs(cut_cell_amount(scalar, geometry) - initial_amount) < 2e-13
    TwoPhaseAMR._invalidate_mesh_topology_cache!(mesh)
    @test_throws ArgumentError advect!(scalar, velocity, 0.005, candidate_ws, geometry)
end

@testset "independent general advection workspaces can advance concurrently" begin
    mesh = BlockTreeMesh(base_mesh=(64, 64), block_size=(8, 8))
    seed = CellField(mesh, (x,y)->1 + 0.2sinpi(2x) * cospi(2y))
    first, second, reference = copy(seed), copy(seed), copy(seed)
    velocity1 = FaceVelocity(mesh,
        (x,y,t,axis)->axis == 1 ? 0.4 + 0.1sinpi(2y) : -0.3)
    velocity2 = copy(velocity1)
    work1, work2 = AdvectionWorkspace(first), AdvectionWorkspace(second)
    frozen = FrozenCellAdvectionWorkspace(reference)
    frozen_cell_advection_step!(reference, velocity1, 0.005, frozen)
    @sync begin
        Threads.@spawn advect!(first, velocity1, 0.005, work1)
        Threads.@spawn advect!(second, velocity2, 0.005, work2)
    end
    for b in eachindex(mesh.blocks)
        @test isequal(interior(first,b), interior(reference,b))
        @test isequal(interior(second,b), interior(reference,b))
    end
end

@testset "regular face IDs select narrow storage with a native-width fallback" begin
    select_id_type = TwoPhaseAMR._cell_advection_face_id_type
    limit = Int(typemax(Int32))
    @test select_id_type(0) === Int32
    @test select_id_type(limit - 1) === Int32
    @test select_id_type(limit) === Int32
    if sizeof(Int) > sizeof(Int32)
        @test select_id_type(limit + 1) === Int
    end
    @test_throws ArgumentError select_id_type(-1)

    # Force the native-width builder on a small mesh. This checks the actual
    # fallback kernel without allocating a mesh with billions of faces.
    for T in (Float32, Float64)
        mesh = BlockTreeMesh(base_mesh=(8, 8), block_size=(8, 4),
                             periodic=(true, false))
        @test any(face -> face.minus.block == face.plus.block &&
                 face.axis == 1 && face.plus.i != face.minus.i + 1, mesh.faces)
        @test !isempty(mesh.boundary_faces)
        seed = CellField(mesh, (x, y)->T(1 + 0.2sinpi(2x) - 0.1cospi(2y)); T=T)
        narrow, native, reference = copy(seed), copy(seed), copy(seed)
        narrow_ws = AdvectionWorkspace(narrow)
        native_ws = TwoPhaseAMR._cell_advection_workspace(native, Int)
        reference_ws = FrozenCellAdvectionWorkspace(reference)
        @test eltype(narrow_ws.x_face_ids) === Int32
        @test eltype(narrow_ws.y_face_ids) === Int32
        @test eltype(native_ws.x_face_ids) === Int
        @test eltype(native_ws.y_face_ids) === Int
        @test !isempty(narrow_ws.interior_faces)
        for ws in (narrow_ws, native_ws)
            @test all(id -> id != 0, ws.x_face_ids)
            @test all(id -> id != 0, ws.y_face_ids)
            for (face_id, face) in pairs(mesh.faces)
                kind = TwoPhaseAMR._cell_advection_regular_kind(face)
                if kind == 1
                    slot = TwoPhaseAMR._cell_advection_x_slot(
                        face.minus.block, face.minus.i, face.minus.j,
                        mesh.block_size...)
                    @test Int(ws.x_face_ids[slot]) == face_id
                elseif kind == 2
                    slot = TwoPhaseAMR._cell_advection_y_slot(
                        face.minus.block, face.minus.i, face.minus.j,
                        mesh.block_size...)
                    @test Int(ws.y_face_ids[slot]) == face_id
                end
            end
        end

        velocity = FaceVelocity(mesh, T; init=zero(T))
        geometry = GeometryField(mesh, (x, y)->x - 0.2; T, lipschitz=1)
        boundary = Dirichlet((x, y, t)->T(0.4 + x - 2y + t))
        for step in 1:3
            for face_id in eachindex(velocity.interior)
                velocity.interior[face_id] = T(isodd(face_id + step) ? 0.23 : -0.17)
                geometry.aperture[face_id] = T(isodd(face_id + step) ? 0.7 : 0.2)
            end
            for face_id in eachindex(velocity.boundary)
                velocity.boundary[face_id] = T(isodd(face_id + step) ? -0.31 : 0.19)
                geometry.boundary_aperture[face_id] = T(isodd(face_id) ? 0.4 : 0.8)
            end
            for cfl in (T(0.6), Float64(0.6))
                expected = frozen_cell_cfl(velocity, reference_ws; cfl)
                for ws in (narrow_ws, native_ws)
                    observed = stable_timestep(velocity, ws; cfl)
                    @test isequal(observed, expected)
                    @test typeof(observed) === promote_type(T, typeof(cfl))
                    for b in eachindex(mesh.blocks)
                        @test isequal(interior(ws.rates, b),
                                      interior(reference_ws.rates, b))
                    end
                end
            end
            if step == 1
                frozen_cell_advection_step!(reference, velocity, T(0.003), reference_ws;
                                            boundary, time=T(0.1step))
                advect!(narrow, velocity, T(0.003), narrow_ws;
                        boundary, time=T(0.1step))
                advect!(native, velocity, T(0.003), native_ws;
                        boundary, time=T(0.1step))
            else
                if step == 2
                    geometry.open_fraction.blocks[1][mesh.halo + 3, mesh.halo + 2] = zero(T)
                end
                frozen_cut_cell_step!(reference, velocity, T(0.003), reference_ws,
                                      geometry; boundary, time=T(0.1step))
                advect!(narrow, velocity, T(0.003), narrow_ws, geometry;
                        boundary, time=T(0.1step))
                advect!(native, velocity, T(0.003), native_ws, geometry;
                        boundary, time=T(0.1step))
                for ws in (narrow_ws, native_ws)
                    @test isequal(ws.increment.blocks[1][mesh.halo + 3,
                                                       mesh.halo + 2], zero(T))
                end
            end
            for (field, ws) in ((narrow, narrow_ws), (native, native_ws))
                for b in eachindex(mesh.blocks)
                    @test isequal(interior(field, b), interior(reference, b))
                    @test isequal(interior(ws.increment, b),
                                  interior(reference_ws.increment, b))
                end
            end
        end
        TwoPhaseAMR._invalidate_mesh_topology_cache!(mesh)
        for (field, ws) in ((narrow, narrow_ws), (native, native_ws))
            @test_throws ArgumentError advect!(field, velocity, T(0.003), ws)
            @test_throws ArgumentError stable_timestep(velocity, ws)
        end
    end
end

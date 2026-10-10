"""A two-component velocity stored on owned coordinate-normal mesh faces."""
struct FaceVelocity{T, M <: BlockTreeMesh} <: AbstractMeshField{T}
    mesh::M
    interior::Vector{T}
    boundary::Vector{T}
end

Base.eltype(::Type{FaceVelocity{T, M}}) where {T, M} = T
Base.eltype(::FaceVelocity{T}) where {T} = T

"""
    FaceVelocity(mesh, [T=eltype(mesh.origin)]; init=zero(T))
    FaceVelocity(mesh, f; time=0, T=eltype(mesh.origin))

Create an owned coordinate-normal face velocity. The first form fills every
component with `init`; the second samples `f(x, y, time, axis)` on the immutable
face schedule. Coarse--fine faces have a single owner and a single value,
which is the conservation invariant used by advection.
"""
function FaceVelocity(
        mesh::M, ::Type{T} = eltype(mesh.origin);
        init = zero(T)
    ) where {M <: BlockTreeMesh, T}
    value = convert(T, init)
    return FaceVelocity{T, M}(
        mesh, fill(value, length(mesh.faces)),
        fill(value, length(mesh.boundary_faces))
    )
end

function FaceVelocity(mesh::M, f; time = 0, T = eltype(mesh.origin)) where {M <: BlockTreeMesh}
    interior = Vector{T}(undef, length(mesh.faces))
    boundary = Vector{T}(undef, length(mesh.boundary_faces))
    velocity = FaceVelocity{T, M}(mesh, interior, boundary)
    return set_velocities!(velocity, f; time)
end

function Base.copy(velocity::FaceVelocity{T, M}) where {T, M}
    return FaceVelocity{T, M}(velocity.mesh, copy(velocity.interior), copy(velocity.boundary))
end

"""Copy one directional component out of a `FaceVelocity`."""
function Base.copyto!(
        destination::FaceField{Axis},
        source::FaceVelocity
    ) where {Axis}
    destination.mesh === source.mesh ||
        throw(ArgumentError("destination and source must share a mesh"))
    local_index = 0
    for (global_index, face) in pairs(source.mesh.faces)
        face.axis == Axis || continue
        local_index += 1
        @inbounds destination.interior[local_index] = source.interior[global_index]
    end
    local_index = 0
    for (global_index, face) in pairs(source.mesh.boundary_faces)
        face.axis == Axis || continue
        local_index += 1
        @inbounds destination.boundary[local_index] = source.boundary[global_index]
    end
    return destination
end

"""Copy a directional `FaceField` into the corresponding velocity component."""
function Base.copyto!(
        destination::FaceVelocity,
        source::FaceField{Axis}
    ) where {Axis}
    destination.mesh === source.mesh ||
        throw(ArgumentError("destination and source must share a mesh"))
    local_index = 0
    for (global_index, face) in pairs(destination.mesh.faces)
        face.axis == Axis || continue
        local_index += 1
        @inbounds destination.interior[global_index] = source.interior[local_index]
    end
    local_index = 0
    for (global_index, face) in pairs(destination.mesh.boundary_faces)
        face.axis == Axis || continue
        local_index += 1
        @inbounds destination.boundary[global_index] = source.boundary[local_index]
    end
    return destination
end

function FaceField{Axis}(velocity::FaceVelocity{T}) where {Axis, T}
    field = FaceField{Axis}(velocity.mesh, T)
    return copyto!(field, velocity)
end

"""Assemble an owned staggered velocity from its two directional components."""
function FaceVelocity(x::FaceField{1, TX, M}, y::FaceField{2, TY, M}) where {TX, TY, M}
    x.mesh === y.mesh || throw(ArgumentError("face components must share a mesh"))
    T = promote_type(TX, TY)
    velocity = FaceVelocity{T, M}(
        x.mesh, zeros(T, length(x.mesh.faces)), zeros(T, length(x.mesh.boundary_faces))
    )
    copyto!(velocity, x)
    copyto!(velocity, y)
    return velocity
end

"""Resample `f(x, y, time, axis)` into an existing `FaceVelocity`."""
function set_velocities!(velocity::FaceVelocity, f; time = 0)
    @_parallel for index in 1:length(velocity.mesh.faces)
        face = velocity.mesh.faces[index]
        @inbounds velocity.interior[index] = f(face.center[1], face.center[2], time, face.axis)
    end
    @_parallel for index in 1:length(velocity.mesh.boundary_faces)
        face = velocity.mesh.boundary_faces[index]
        @inbounds velocity.boundary[index] = f(face.center[1], face.center[2], time, face.axis)
    end
    return velocity
end

"""
    enforce_no_permeation!(velocity)

Set the coordinate-normal velocity on every physical domain face to zero.
Periodic directions have no physical boundary entries, so a mesh with any
non-periodic direction automatically receives the impermeable-wall condition
without affecting periodic seams.
"""
function enforce_no_permeation!(velocity::FaceVelocity{T}) where {T}
    fill!(velocity.boundary, zero(T))
    return velocity
end

"""Compact mesh-only data for one owned interior face."""
struct _CompactCellAdvectionFace{T}
    face_id::Int
    minus_block::Int
    minus_index::Int
    plus_block::Int
    plus_index::Int
    area::T
end

@inline function _cell_advection_array_index(ref::CellRef, nx::Int, halo::Int)
    stride = nx + 2halo
    return ref.i + halo + (ref.j + halo - 1) * stride
end

@inline function _cell_advection_upwind_state(
        scalar::CellField, face::_CompactCellAdvectionFace, speed
    )
    return speed >= 0 ?
        @inbounds(scalar.blocks[face.minus_block][face.minus_index]) :
        @inbounds(scalar.blocks[face.plus_block][face.plus_index])
end

"""Reusable regular-interior and residual-face storage for cell advection.

Each leaf has dense x/y face slots for true arithmetic neighbours. The slots
hold original mesh-face IDs, so carrier and aperture changes remain live.
Cells touching a block edge retain ordered incidence entries; other cells use
four arithmetic flux neighbours and a packed original-face reduction order.
"""
struct AdvectionWorkspace{T, M <: BlockTreeMesh, TM, TI <: Integer}
    increment::CellField{T, M}
    rates::CellField{T, M}
    interior_faces::Vector{_CompactCellAdvectionFace{TM}}
    interior_flux::Vector{T}
    x_face_ids::Vector{TI}
    y_face_ids::Vector{TI}
    x_areas::Vector{TM}
    y_areas::Vector{TM}
    x_flux::Vector{T}
    y_flux::Vector{T}
    interior_order::Vector{UInt8}
    boundary_flux::Vector{T}
    cell_offsets::Vector{Int}
    face_indices::Vector{Int}
    face_signs::Vector{Int8}
    face_kinds::Vector{UInt8}
    generation::UInt
end

@inline _cell_advection_x_slot(b, i, j, nx, ny) =
    (b - 1) * (nx - 1) * ny + (j - 1) * (nx - 1) + i
@inline _cell_advection_y_slot(b, i, j, nx, ny) =
    (b - 1) * nx * (ny - 1) + (j - 1) * nx + i

@inline function _cell_advection_regular_kind(face::InteriorFace)
    face.minus.block == face.plus.block || return UInt8(0)
    if face.axis == 1 && face.plus.i == face.minus.i + 1 &&
            face.plus.j == face.minus.j
        return UInt8(1)
    elseif face.axis == 2 && face.plus.j == face.minus.j + 1 &&
            face.plus.i == face.minus.i
        return UInt8(2)
    end
    return UInt8(0)
end

@inline function _cell_advection_order(left, right, down, up)
    ids = (left, right, down, up)
    order = zero(UInt8)
    used = zero(UInt8)
    for position in 0:3
        best = typemax(Int)
        selected = 0
        for direction in 0:3
            if iszero(used & (UInt8(1) << direction)) && ids[direction + 1] < best
                best = ids[direction + 1]
                selected = direction
            end
        end
        order |= UInt8(selected) << (2position)
        used |= UInt8(1) << selected
    end
    return order
end

function _cell_advection_face_id_type(nfaces::Integer)
    nfaces >= 0 || throw(ArgumentError("face count must be nonnegative"))
    return nfaces <= typemax(Int32) ? Int32 : Int
end

AdvectionWorkspace(field::CellField) =
    _cell_advection_workspace(field, _cell_advection_face_id_type(length(field.mesh.faces)))

function _cell_advection_workspace(field::CellField{T, M}, ::Type{TI}) where {T, M, TI <: Integer}
    mesh = field.mesh
    targets = Int[]
    face_indices = Int[]
    face_signs = Int8[]
    face_kinds = UInt8[]
    nx, ny = mesh.block_size
    h = mesh.halo
    TM = eltype(mesh.origin)
    x_face_ids = zeros(TI, nblocks(mesh) * (nx - 1) * ny)
    y_face_ids = zeros(TI, nblocks(mesh) * nx * (ny - 1))
    x_areas = zeros(TM, nblocks(mesh))
    y_areas = zeros(TM, nblocks(mesh))
    face_to_kind = Vector{UInt8}(undef, length(mesh.faces))
    face_to_slot = Vector{Int}(undef, length(mesh.faces))
    compact_faces = _CompactCellAdvectionFace{TM}[]
    for (index, face) in pairs(mesh.faces)
        kind = _cell_advection_regular_kind(face)
        face_to_kind[index] = kind
        if kind == 1
            slot = _cell_advection_x_slot(face.minus.block, face.minus.i,
                                          face.minus.j, nx, ny)
            x_face_ids[slot] = convert(TI, index)
            x_areas[face.minus.block] = face.area
        elseif kind == 2
            slot = _cell_advection_y_slot(face.minus.block, face.minus.i,
                                          face.minus.j, nx, ny)
            y_face_ids[slot] = convert(TI, index)
            y_areas[face.minus.block] = face.area
        else
            push!(compact_faces, _CompactCellAdvectionFace(
                index, face.minus.block, _cell_advection_array_index(face.minus, nx, h),
                face.plus.block, _cell_advection_array_index(face.plus, nx, h),
                face.area,
            ))
            slot = length(compact_faces)
        end
        face_to_slot[index] = slot
    end
    # Edge cells can see arbitrary AMR and periodic incidences. Keep their
    # complete original reduction sequence, including adjacent regular faces.
    interior_order = zeros(UInt8, ncells(mesh))
    for b in eachindex(mesh.blocks), j in 2:(ny - 1), i in 2:(nx - 1)
        cell = (b - 1) * nx * ny + (j - 1) * nx + i
        left = x_face_ids[_cell_advection_x_slot(b, i - 1, j, nx, ny)]
        right = x_face_ids[_cell_advection_x_slot(b, i, j, nx, ny)]
        down = y_face_ids[_cell_advection_y_slot(b, i, j - 1, nx, ny)]
        up = y_face_ids[_cell_advection_y_slot(b, i, j, nx, ny)]
        # Face IDs encode the exact four-term reduction order of the old CSR.
        interior_order[cell] = _cell_advection_order(left, right, down, up)
    end
    for (index, face) in pairs(mesh.faces)
        for (ref, sign) in ((face.minus, Int8(-1)), (face.plus, Int8(1)))
            if ref.i == 1 || ref.i == nx || ref.j == 1 || ref.j == ny
                push!(targets, _cell_linear_index(ref, nx, ny))
                push!(face_indices, face_to_slot[index])
                push!(face_signs, sign)
                push!(face_kinds, face_to_kind[index])
            end
        end
    end
    for (index, face) in pairs(mesh.boundary_faces)
        push!(targets, _cell_linear_index(face.cell, nx, ny))
        push!(face_indices, index)
        push!(face_signs, -1)
        push!(face_kinds, 4)
    end
    offsets, permutation = _grouped_output_schedule(targets, ncells(mesh))
    return AdvectionWorkspace(
        CellField(mesh, T), CellField(mesh, T),
        compact_faces,
        zeros(T, length(compact_faces)), x_face_ids, y_face_ids,
        x_areas, y_areas,
        zeros(T, length(x_face_ids)), zeros(T, length(y_face_ids)),
        interior_order, zeros(T, length(mesh.boundary_faces)),
        offsets, face_indices[permutation], face_signs[permutation],
        face_kinds[permutation], mesh.topology_cache.generation,
    )
end

@inline function _validate_advection_workspace_generation(workspace::AdvectionWorkspace)
    workspace.generation == workspace.increment.mesh.topology_cache.generation ||
        throw(ArgumentError("advection workspace has stale mesh topology"))
    return nothing
end

@inline function _cell_advection_face_flux(workspace::AdvectionWorkspace, kind::UInt8, slot::Int)
    if kind == 1
        return @inbounds workspace.x_flux[slot]
    elseif kind == 2
        return @inbounds workspace.y_flux[slot]
    elseif kind == 4
        return @inbounds workspace.boundary_flux[slot]
    end
    return @inbounds workspace.interior_flux[slot]
end

@inline function _cell_advection_ordered_sum(a, b, c, d, order::UInt8, ::Type{T}) where {T}
    fluxes = (a, b, c, d)
    integrated = zero(T)
    integrated += fluxes[Int(order & 0x03) + 1]
    integrated += fluxes[Int((order >> 2) & 0x03) + 1]
    integrated += fluxes[Int((order >> 4) & 0x03) + 1]
    integrated += fluxes[Int(order >> 6) + 1]
    return integrated
end

@inline function _cell_advection_regular_sum(
        workspace::AdvectionWorkspace, b::Int, i::Int, j::Int,
        nx::Int, ny::Int, cell::Int, signed::Bool, ::Type{T}
    ) where {T}
    left = @inbounds workspace.x_flux[_cell_advection_x_slot(b, i - 1, j, nx, ny)]
    right = @inbounds workspace.x_flux[_cell_advection_x_slot(b, i, j, nx, ny)]
    down = @inbounds workspace.y_flux[_cell_advection_y_slot(b, i, j - 1, nx, ny)]
    up = @inbounds workspace.y_flux[_cell_advection_y_slot(b, i, j, nx, ny)]
    order = @inbounds workspace.interior_order[cell]
    return signed ? _cell_advection_ordered_sum(left, -right, down, -up, order, T) :
        _cell_advection_ordered_sum(left, right, down, up, order, T)
end

@inline function _cell_advection_residual_sum(
        workspace::AdvectionWorkspace, cell::Int, signed::Bool, ::Type{T}
    ) where {T}
    integrated = zero(T)
    @inbounds for pointer in workspace.cell_offsets[cell]:(workspace.cell_offsets[cell + 1] - 1)
        flux = _cell_advection_face_flux(workspace,
            workspace.face_kinds[pointer], workspace.face_indices[pointer])
        integrated += signed ? workspace.face_signs[pointer] * flux : flux
    end
    return integrated
end

@inline function _cell_advection_sum(
        workspace::AdvectionWorkspace, b::Int, i::Int, j::Int,
        nx::Int, ny::Int, cell::Int, signed::Bool, ::Type{T}
    ) where {T}
    if 1 < i < nx && 1 < j < ny
        return _cell_advection_regular_sum(workspace, b, i, j, nx, ny, cell, signed, T)
    end
    return _cell_advection_residual_sum(workspace, cell, signed, T)
end

@inline _cell_advection_regular_flux(dt, speed, state, area, face_id, ::Nothing) =
    dt * speed * state * area

@inline _cell_advection_regular_flux(dt, speed, state, area, face_id, geometry) =
    @inbounds(dt * speed * state * area * geometry.aperture[face_id])

function _cell_advection_regular_fluxes!(
        scalar::CellField, velocity::FaceVelocity, dt,
        workspace::AdvectionWorkspace, geometry
    )
    mesh = scalar.mesh
    nx, ny = mesh.block_size
    h = mesh.halo
    stride = nx + 2h
    _parallel_block_columns(mesh) do b, first_j, last_j
        values = @inbounds scalar.blocks[b]
        x_area = @inbounds workspace.x_areas[b]
        y_area = @inbounds workspace.y_areas[b]
        @inbounds for j in first_j:last_j
            row = (j + h - 1) * stride + h
            for i in 1:(nx - 1)
                slot = _cell_advection_x_slot(b, i, j, nx, ny)
                face_id = workspace.x_face_ids[slot]
                speed = velocity.interior[face_id]
                state = speed >= 0 ? values[row + i] : values[row + i + 1]
                workspace.x_flux[slot] =
                    _cell_advection_regular_flux(dt, speed, state, x_area, face_id, geometry)
            end
            if j < ny
                for i in 1:nx
                    slot = _cell_advection_y_slot(b, i, j, nx, ny)
                    face_id = workspace.y_face_ids[slot]
                    speed = velocity.interior[face_id]
                    state = speed >= 0 ? values[row + i] : values[row + i + stride]
                    workspace.y_flux[slot] =
                        _cell_advection_regular_flux(dt, speed, state, y_area, face_id, geometry)
                end
            end
        end
    end
    return nothing
end

function _cell_advection_regular_rates!(velocity::FaceVelocity, workspace::AdvectionWorkspace)
    mesh = velocity.mesh
    nx, ny = mesh.block_size
    _parallel_block_columns(mesh) do b, first_j, last_j
        x_area = @inbounds workspace.x_areas[b]
        y_area = @inbounds workspace.y_areas[b]
        @inbounds for j in first_j:last_j
            for i in 1:(nx - 1)
                slot = _cell_advection_x_slot(b, i, j, nx, ny)
                workspace.x_flux[slot] = abs(velocity.interior[workspace.x_face_ids[slot]]) * x_area
            end
            if j < ny
                for i in 1:nx
                    slot = _cell_advection_y_slot(b, i, j, nx, ny)
                    workspace.y_flux[slot] = abs(velocity.interior[workspace.y_face_ids[slot]]) * y_area
                end
            end
        end
    end
    return nothing
end

function _zero_interior!(field::CellField)
    nx, ny = field.mesh.block_size
    h = field.mesh.halo
    _parallel_block_columns(field.mesh) do b, first_j, last_j
        values = @inbounds field.blocks[b]
        @inbounds for j in first_j:last_j, i in 1:nx
            values[i + h, j + h] = zero(eltype(field))
        end
    end
    return field
end

"""
    divergence!(destination, velocity)

Compute the finite-volume divergence of an owned face-normal velocity field at
cell centres. The same face schedule and metric weights are used by scalar and
staggered transport.
"""
function divergence!(destination::CellField{T}, velocity::FaceVelocity) where {T}
    destination.mesh === velocity.mesh ||
        throw(ArgumentError("destination and velocity must share a mesh"))
    _zero_interior!(destination)
    mesh = destination.mesh
    for (index, face) in pairs(mesh.faces)
        @inbounds flux = velocity.interior[index] * face.area
        _add!(destination, flux / mesh.blocks[face.minus.block].volume, face.minus)
        _add!(destination, -flux / mesh.blocks[face.plus.block].volume, face.plus)
    end
    for (index, face) in pairs(mesh.boundary_faces)
        @inbounds flux = face.side * velocity.boundary[index] * face.area
        _add!(destination, flux / mesh.blocks[face.cell.block].volume, face.cell)
    end
    return destination
end

@inline function _inflow_value(::ZeroGradient, scalar, face, time)
    return _get(scalar, face.cell)
end

@inline function _inflow_value(bc::Dirichlet, scalar, face, time)
    return bc.value(face.center[1], face.center[2], time)
end

"""
    advect!(scalar, velocity, dt, workspace; boundary=ZeroGradient(), time=0)

Advance cell averages with forward Euler and a first-order upwind finite-volume
flux. Every interior face, including every coarse--fine face segment, is owned
exactly once and applies equal-and-opposite integrated fluxes to its two cells.
After workspace construction the update is allocation-free.
"""
function advect!(
        scalar::CellField{T}, velocity::FaceVelocity, dt,
        workspace::AdvectionWorkspace{T};
        boundary::AbstractBoundaryCondition = ZeroGradient(), time = 0
    ) where {T}
    scalar.mesh === velocity.mesh === workspace.increment.mesh ||
        throw(ArgumentError("scalar, velocity, and workspace must share a mesh"))
    _validate_advection_workspace_generation(workspace)
    increment = workspace.increment
    mesh = scalar.mesh
    @_parallel for index in eachindex(workspace.interior_faces)
        face = workspace.interior_faces[index]
        @inbounds speed = velocity.interior[face.face_id]
        state = _cell_advection_upwind_state(scalar, face, speed)
        @inbounds workspace.interior_flux[index] =
            dt * speed * state * face.area
    end
    _cell_advection_regular_fluxes!(scalar, velocity, dt, workspace, nothing)
    @_parallel for index in 1:length(mesh.boundary_faces)
        face = mesh.boundary_faces[index]
        @inbounds coordinate_speed = velocity.boundary[index]
        outward_speed = face.side * coordinate_speed
        state = outward_speed >= 0 ? _get(scalar, face.cell) :
            _inflow_value(boundary, scalar, face, time)
        @inbounds workspace.boundary_flux[index] =
            dt * outward_speed * state * face.area
    end
    nx, ny = mesh.block_size
    h = mesh.halo
    cells_per_block = nx * ny
    _parallel_block_columns(mesh) do b, first_j, last_j
        increment_block = @inbounds increment.blocks[b]
        scalar_block = @inbounds scalar.blocks[b]
        volume = mesh.blocks[b].volume
        cell_offset = (b - 1) * cells_per_block
        @inbounds for j in first_j:last_j, i in 1:nx
            cell = cell_offset + (j - 1) * nx + i
            integrated = _cell_advection_sum(workspace, b, i, j, nx, ny, cell, true, T)
            change = integrated / volume
            increment_block[i + h, j + h] = change
            scalar_block[i + h, j + h] += change
        end
    end
    return scalar
end

"""
    stable_timestep(velocity, workspace; cfl=0.9)

Return a conservative multidimensional upwind CFL bound. Absolute face rates
are accumulated, making the result safe for arbitrary prescribed velocities.
"""
function stable_timestep(velocity::FaceVelocity{TV}, workspace::AdvectionWorkspace{TW}; cfl = 0.9) where {TV, TW}
    velocity.mesh === workspace.rates.mesh ||
        throw(ArgumentError("velocity and workspace must share a mesh"))
    _validate_advection_workspace_generation(workspace)
    T = promote_type(TV, TW, typeof(cfl))
    zero(T) < cfl <= one(T) || throw(ArgumentError("cfl must lie in (0, 1]"))
    return _stable_timestep_promoted(velocity, workspace, T, cfl)
end

# Capture the promoted type as a method parameter. Capturing a runtime DataType
# in the column closure boxes zero(T) and the accumulated rate for every cell.
function _stable_timestep_promoted(
        velocity::FaceVelocity, workspace::AdvectionWorkspace,
        ::Type{T}, cfl
    ) where {T}
    rates = workspace.rates
    mesh = velocity.mesh
    @_parallel for index in eachindex(workspace.interior_faces)
        face = workspace.interior_faces[index]
        @inbounds workspace.interior_flux[index] =
            abs(velocity.interior[face.face_id]) * face.area
    end
    _cell_advection_regular_rates!(velocity, workspace)
    @_parallel for index in 1:length(mesh.boundary_faces)
        face = mesh.boundary_faces[index]
        @inbounds workspace.boundary_flux[index] =
            abs(velocity.boundary[index]) * face.area
    end
    nx, ny = mesh.block_size
    h = mesh.halo
    cells_per_block = nx * ny
    _parallel_block_columns(mesh) do b, first_j, last_j
        rates_block = @inbounds rates.blocks[b]
        volume = mesh.blocks[b].volume
        cell_offset = (b - 1) * cells_per_block
        @inbounds for j in first_j:last_j, i in 1:nx
            cell = cell_offset + (j - 1) * nx + i
            integrated = _cell_advection_sum(workspace, b, i, j, nx, ny, cell, false, T)
            rates_block[i + h, j + h] = integrated / volume
        end
    end
    maxrate = zero(T)
    for b in eachindex(mesh.blocks)
        rates_block = @inbounds rates.blocks[b]
        @inbounds for j in 1:ny, i in 1:nx
            maxrate = max(maxrate, rates_block[i + h, j + h])
        end
    end
    return iszero(maxrate) ? T(Inf) : T(cfl) / maxrate
end

"""
    WYAdvectionWorkspace(interface; ctu_threshold=0.5)

Reusable WY sweep arrays; reconstruct after regridding by creating a new workspace.
`liquid_flux` and `gas_flux` contain signed, time-integrated volumes on owned
faces from the most recent complete step, in the same order as `mesh.faces`.
Intrinsic phase values are recovered with the interface's
`fraction_floor`; the same threshold removes small VOF fragments.
The modified staggered flux interpolant switches to CTU when either adjacent
predicted phase fraction is below `ctu_threshold`.
"""
struct WYAdvectionWorkspace{T, M <: BlockTreeMesh, P, F}
    mesh::M
    interface::F
    indicator::CellField{T, M}
    increment::CellField{T, M}
    incoming::CellField{T, M}
    outgoing::CellField{T, M}
    liquid_dilation::CellField{T, M}
    gas_dilation::CellField{T, M}
    gas_carrier::CellField{T, M}
    gas_increment::CellField{T, M}
    open_flux::Vector{T}
    liquid_flux::Vector{T}
    gas_flux::Vector{T}
    liquid_aperture::Vector{T}
    gas_aperture::Vector{T}
    cell_face_offsets::Vector{Int}
    cell_faces::Vector{Int}
    cell_face_orientations::Vector{Int8}
    geometry::Any
    topology::Vector{BlockKey}
    ctu_threshold::T
    interface_transport::P
    diagnostics::PhaseFluxDiagnostics{T}
    method_workspace::Base.RefValue{Any}
end
function WYAdvectionWorkspace(
        field::InterfaceField{T, M};
        ctu_threshold = 0.5,
        interface_transport = nothing, interface_method = nothing,
    ) where {T, M}
    mesh = field.mesh
    transport = _select_interface_transport(
        interface_transport, interface_method,
        _OneVelocityInterfaceAdvection(),
    )
    transport isa AbstractInterfaceTransportMethod || throw(
        ArgumentError(
            "interface_transport must implement " *
                "AbstractInterfaceTransportMethod",
        )
    )
    threshold = T(ctu_threshold)
    isfinite(threshold) && zero(T) < threshold <= one(T) || throw(
        ArgumentError(
            "ctu_threshold must be finite and in (0, 1]"
        )
    )
    offsets, faces, signs = _cell_face_incidence(mesh)
    number_of_faces = length(mesh.faces)
    return WYAdvectionWorkspace(
        mesh, field, CellField(mesh, T), CellField(mesh, T),
        CellField(mesh, T), CellField(mesh, T),
        CellField(mesh, T), CellField(mesh, T), CellField(mesh, T),
        CellField(mesh, T),
        zeros(T, number_of_faces),
        zeros(T, number_of_faces), zeros(T, number_of_faces),
        zeros(T, number_of_faces), zeros(T, number_of_faces),
        offsets, faces, Int8.(-signs),
        field.geometry, [block.key for block in mesh.blocks], threshold,
        transport, PhaseFluxDiagnostics(T),
        Ref{Any}(_interface_transport_workspace(field, transport)),
    )
end

function Base.getproperty(workspace::WYAdvectionWorkspace, name::Symbol)
    name === :interface_method && return getfield(workspace, :interface_transport)
    return getfield(workspace, name)
end

function Base.propertynames(workspace::WYAdvectionWorkspace, private::Bool = false)
    names = fieldnames(typeof(workspace))
    return private ? (names..., :interface_method) : (names..., :interface_method)
end

function _resize_zero_cell_field!(field::CellField{T}) where {T}
    mesh = field.mesh
    nx, ny = mesh.block_size
    halo = mesh.halo
    old_length = length(field.blocks)
    resize!(field.blocks, nblocks(mesh))
    for block in eachindex(field.blocks)
        if block > old_length
            field.blocks[block] = zeros(T, nx + 2halo, ny + 2halo)
        else
            fill!(field.blocks[block], zero(T))
        end
    end
    return field
end

"""Retarget WY scratch after AMR while retaining all growable storage."""
function _reconfigure_wy_workspace!(
        workspace::WYAdvectionWorkspace{T}, field::InterfaceField{T},
    ) where {T}
    workspace.interface === field || return WYAdvectionWorkspace(
        field; ctu_threshold = workspace.ctu_threshold,
        interface_transport = workspace.interface_transport,
    )
    workspace.mesh === field.mesh || return WYAdvectionWorkspace(
        field; ctu_threshold = workspace.ctu_threshold,
        interface_transport = workspace.interface_transport,
    )
    workspace.geometry === field.geometry || return WYAdvectionWorkspace(
        field; ctu_threshold = workspace.ctu_threshold,
        interface_transport = workspace.interface_transport,
    )
    isnothing(workspace.method_workspace[]) || return WYAdvectionWorkspace(
        field; ctu_threshold = workspace.ctu_threshold,
        interface_transport = workspace.interface_transport,
    )
    for cell_field in (
            workspace.indicator, workspace.increment,
            workspace.incoming, workspace.outgoing,
            workspace.liquid_dilation, workspace.gas_dilation,
            workspace.gas_carrier, workspace.gas_increment,
        )
        _resize_zero_cell_field!(cell_field)
    end
    for flux in (
            workspace.open_flux, workspace.liquid_flux, workspace.gas_flux,
            workspace.liquid_aperture, workspace.gas_aperture,
        )
        resize!(flux, length(field.mesh.faces))
        fill!(flux, zero(T))
    end
    offsets, faces, signs = _cell_face_incidence(field.mesh)
    empty!(workspace.cell_face_offsets)
    append!(workspace.cell_face_offsets, offsets)
    empty!(workspace.cell_faces)
    append!(workspace.cell_faces, faces)
    empty!(workspace.cell_face_orientations)
    append!(workspace.cell_face_orientations, Int8.(-signs))
    empty!(workspace.topology)
    append!(workspace.topology, (block.key for block in field.mesh.blocks))
    _reset!(workspace.diagnostics)
    return workspace
end

function _update_phase_flux_diagnostics!(workspace::WYAdvectionWorkspace{T}) where {T}
    maximum_defect = zero(T)
    absolute_defect = zero(T)
    signed_defect = zero(T)
    @inbounds for face in eachindex(workspace.open_flux)
        defect = workspace.liquid_flux[face] + workspace.gas_flux[face] -
            workspace.open_flux[face]
        maximum_defect = max(maximum_defect, abs(defect))
        absolute_defect += abs(defect)
        signed_defect += defect
    end
    diagnostics = workspace.diagnostics
    diagnostics.maximum_partition_defect = maximum_defect
    diagnostics.absolute_partition_defect = absolute_defect
    diagnostics.signed_partition_defect = signed_defect
    return diagnostics
end

function _update_gas_reset_diagnostics!(
        field::InterfaceField{T}, workspace::WYAdvectionWorkspace{T};
        accumulate = false,
    ) where {T}
    signed = zero(T)
    absolute = zero(T)
    mesh = field.mesh
    nx, ny = mesh.block_size
    halo = mesh.halo
    geometry = field.geometry
    for (block, leaf) in pairs(mesh.blocks)
        conditional = @inbounds field.conditional_fraction.blocks[block]
        carrier = @inbounds workspace.gas_carrier.blocks[block]
        open = isnothing(geometry) ? nothing :
            @inbounds(geometry.open_fraction.blocks[block])
        @inbounds for j in 1:ny, i in 1:nx
            ii, jj = i + halo, j + halo
            open_fraction = isnothing(open) ? one(T) : open[ii, jj]
            discrepancy = leaf.volume * open_fraction *
                (carrier[ii, jj] - (one(T) - conditional[ii, jj]))
            signed += discrepancy
            absolute += abs(discrepancy)
        end
    end
    if accumulate
        workspace.diagnostics.gas_reset_discrepancy += signed
        workspace.diagnostics.absolute_gas_reset_discrepancy += absolute
    else
        workspace.diagnostics.gas_reset_discrepancy = signed
        workspace.diagnostics.absolute_gas_reset_discrepancy = absolute
    end
    return workspace.diagnostics
end

function _reset_auxiliary_gas_carrier!(
        field::InterfaceField{T}, workspace::WYAdvectionWorkspace{T},
    ) where {T}
    mesh = field.mesh
    nx, ny = mesh.block_size
    halo = mesh.halo
    _parallel_block_columns(mesh) do block, first_j, last_j
        liquid = @inbounds field.conditional_fraction.blocks[block]
        gas = @inbounds workspace.gas_carrier.blocks[block]
        @inbounds for j in first_j:last_j, i in 1:nx
            ii, jj = i + halo, j + halo
            gas[ii, jj] = one(T) - liquid[ii, jj]
        end
    end
    return workspace.gas_carrier
end
function _check_wy_workspace(velocity, work)
    velocity.mesh === work.mesh || throw(ArgumentError("velocity and workspace must share a mesh"))
    length(work.topology) == nblocks(work.mesh) &&
        all(work.topology[i] == work.mesh.blocks[i].key for i in eachindex(work.topology)) ||
        throw(ArgumentError("rebuild WYAdvectionWorkspace after mesh adaptation"))
    all(isfinite, velocity.interior) && all(isfinite, velocity.boundary) ||
        throw(ArgumentError("velocity must be finite"))
    return all(iszero, velocity.boundary) ||
        throw(ArgumentError("WY currently requires impermeable or periodic boundaries"))
end

"""
    stable_timestep(velocity, work::WYAdvectionWorkspace; cfl=0.5)

WY incoming-volume CFL: `dt*sum(max(-outward_volume_rate,0))/cell_volume`.
Also constrain outgoing volume and each face extrusion to half the donor cell
width; the latter protects subfaces on mixed-level meshes. `0 < cfl <= 0.5`.
"""
function stable_timestep(velocity::FaceVelocity, work::WYAdvectionWorkspace{T}; cfl = 0.5) where {T}
    _check_wy_workspace(velocity, work)
    isfinite(cfl) && 0 < cfl <= 0.5 || throw(ArgumentError("WY requires 0 < cfl <= 0.5"))
    mesh = work.mesh
    geometry = work.geometry
    nx, ny = mesh.block_size
    halo = mesh.halo
    cells_per_block = nx * ny
    _parallel_block_columns(mesh) do b, first_j, last_j
        incoming_values = @inbounds work.incoming.blocks[b]
        outgoing_values = @inbounds work.outgoing.blocks[b]
        volume = mesh.blocks[b].volume
        offset = (b - 1) * cells_per_block
        @inbounds for j in first_j:last_j, i in 1:nx
            cell = offset + (j - 1) * nx + i
            incoming = zero(T)
            outgoing = zero(T)
            for pointer in work.cell_face_offsets[cell]:(
                    work.cell_face_offsets[cell + 1] - 1
                )
                face_index = work.cell_faces[pointer]
                orientation = work.cell_face_orientations[pointer]
                face = mesh.faces[face_index]
                aperture = isnothing(geometry) ? one(T) :
                    geometry.aperture[face_index]
                outward_rate = orientation * velocity.interior[face_index] *
                    face.area * aperture
                outgoing += max(outward_rate, zero(T))
                incoming += max(-outward_rate, zero(T))
            end
            open = isnothing(geometry) ? one(T) :
                geometry.open_fraction.blocks[b][i + halo, j + halo]
            effective_volume = volume * open
            incoming_values[i + halo, j + halo] =
                effective_volume > zero(T) ? incoming / effective_volume : zero(T)
            outgoing_values[i + halo, j + halo] =
                effective_volume > zero(T) ? outgoing / effective_volume : zero(T)
        end
    end
    maxrate = zero(T)
    for (i, face) in pairs(mesh.faces)
        u = velocity.interior[i]
        donor = u >= 0 ? face.minus : face.plus
        maxrate = max(
            maxrate,
            abs(u) / mesh.blocks[donor.block].spacing[face.axis],
        )
    end
    for b in eachindex(mesh.blocks)
        incoming_values = @inbounds work.incoming.blocks[b]
        outgoing_values = @inbounds work.outgoing.blocks[b]
        @inbounds for j in 1:ny, i in 1:nx
            maxrate = max(
                maxrate, incoming_values[i + halo, j + halo],
                outgoing_values[i + halo, j + halo],
            )
        end
    end
    return maxrate == 0 ? T(Inf) : T(cfl) / maxrate
end

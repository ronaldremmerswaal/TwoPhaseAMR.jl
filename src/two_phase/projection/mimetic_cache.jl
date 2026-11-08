"""Topology/solid-geometry data and reusable numerical scratch for mimetic projection.

Pressure topology and solid-geometry data are frozen by block keys, geometry
identity and revision. The embedded-cell mask also depends on the interface's
fraction floor. Interface-dependent coefficients and active faces are refreshed
on every projection; row slices are rebuilt when the active space changes.
"""
mutable struct MimeticProjectionCache{T, M <: BlockTreeMesh}
    pressure::CMDGFMPressureCache{T, M}
    keys::Vector{BlockKey}
    geometry::Union{Nothing, GeometryField}
    geometry_revision::UInt
    embedded_fraction_floor::T
    initialized::Bool
    incidence_pressure::SparseMatrixCSC{T, Int}
    active_incidence::SparseMatrixCSC{T, Int}
    inactive_incidence::SparseMatrixCSC{T, Int}
    active_faces::Vector{Int}
    inactive_faces::Vector{Int}
    previous_active_faces::Vector{Int}
    previous_inactive_faces::Vector{Int}
    active_mask::BitVector
    embedded_cells::BitVector
    face_dual_volumes::Vector{T}
    krylov::Union{Nothing, Krylov.MinresWorkspace{T, T, Vector{T}}}
    correction_rhs::Vector{T}
    residual::Vector{T}
    pressure_rhs::Vector{T}
    multiplier_rhs::Vector{T}
    fit::MimeticJointFitScratch{T}
end

function MimeticProjectionCache(::Type{T}, ::Type{M}) where {T, M <: BlockTreeMesh}
    return MimeticProjectionCache{T, M}(
        CMDGFMPressureCache(T, M), BlockKey[], nothing, UInt(0), zero(T), false,
        spzeros(T, 0, 0), spzeros(T, 0, 0),
        spzeros(T, 0, 0), Int[], Int[], Int[], Int[], falses(0), falses(0),
        T[], nothing, T[], T[], T[], T[], MimeticJointFitScratch(T),
    )
end

function _mimetic_pressure_cache!(cache::MimeticProjectionCache{T}, solver, data) where {T}
    interface = solver.interface
    mesh = interface.mesh
    pressure = cache.pressure
    geometry = interface.geometry
    revision = _interface_geometry_revision(geometry)
    current = cache.initialized && cache.geometry === geometry &&
        cache.geometry_revision == revision && _cmdgfm_same_topology(cache.keys, mesh)
    # The shared pressure cache marks itself initialized before every part of
    # its rebuild can succeed. Invalidate our own stamp until *all* mimetic
    # data is ready; after an exception, the next call rebuilds both caches.
    cache.initialized = false
    current || (pressure.initialized = false)
    _cmdgfm_pressure_cache!(pressure, solver, data)
    if !current
        cache.incidence_pressure = pressure.incidence[:, pressure.pressure_indices]
        cache.face_dual_volumes = isnothing(interface.geometry) ?
            _uncut_face_dual_volumes(mesh, T) :
            _geometry_face_dual_volumes(interface.geometry)
        resize!(cache.active_mask, length(mesh.faces))
        # Empty previous lists are not sufficient to force a rebuild when
        # there are no active or inactive faces; reset both slices as well.
        cache.active_incidence = spzeros(T, 0, length(pressure.pressure_indices))
        cache.inactive_incidence = spzeros(T, 0, length(pressure.pressure_indices))
        empty!(cache.previous_active_faces)
        empty!(cache.previous_inactive_faces)
        resize!(cache.keys, nblocks(mesh))
        for index in eachindex(cache.keys)
            cache.keys[index] = mesh.blocks[index].key
        end
        cache.geometry = geometry
        cache.geometry_revision = revision
    end
    if !current || cache.embedded_fraction_floor != interface.fraction_floor
        cache.embedded_cells = _local_embedded_cells(interface)
        cache.embedded_fraction_floor = interface.fraction_floor
    end
    cache.initialized = true
    return cache
end

function _mimetic_face_slices!(cache::MimeticProjectionCache)
    active, inactive = cache.active_faces, cache.inactive_faces
    if active != cache.previous_active_faces
        cache.active_incidence = cache.incidence_pressure[active, :]
        resize!(cache.previous_active_faces, length(active))
        copyto!(cache.previous_active_faces, active)
    end
    if inactive != cache.previous_inactive_faces
        cache.inactive_incidence = cache.incidence_pressure[inactive, :]
        resize!(cache.previous_inactive_faces, length(inactive))
        copyto!(cache.previous_inactive_faces, inactive)
    end
    return cache
end

# MINRES uses the same grow-only Krylov workspace protocol as the other
# solvers; its error window is not a problem-sized vector.
_krylov_traits(::Krylov.MinresWorkspace) = (;
    method = :minres, solve = Krylov.minres!, preconditioner = :M,
    callback = true, required = (:x, :r1, :r2, :w1, :w2, :y),
    optional = (:Δx, :npc_dir, :v), bases = (),
    retry_unpreconditioned = false,
)

function _mimetic_krylov_workspace!(cache::MimeticProjectionCache, matrix, rhs)
    krylov = cache.krylov
    if isnothing(krylov)
        krylov = Krylov.MinresWorkspace(matrix, rhs)
        cache.krylov = krylov
    else
        _resize_krylov_workspace!(krylov, length(rhs))
    end
    return krylov
end

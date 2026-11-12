struct RawDualFace{T}
    minus::Int
    plus::Int
    flux_a::Int
    flux_b::Int
    axis::Int
    area::T
    distance::T
end

struct RawAMRDualFace
    minus::Int
    plus::Int
    first_weight::Int
    weight_count::Int
    axis::Int
    refinement_interface::Bool
end


struct ComponentFaceIncidence{T}
    local_index::Int
    global_index::Int
    side::Int
    fraction::T
end

"""
    StaggeredAdvectionWorkspace(field)

Build the dual-control-volume schedule for a directional `FaceField`. Fully
periodic uniform and 2:1-balanced mixed-level meshes are supported. On AMR,
dual fluxes are factored algebraically from the primary face fluxes so that
their divergence is exactly the volume-weighted interpolation of the primary
cell divergence. Contributions are kept separate by coordinate direction so
that an LW sweep never combines x and y fluxes into one update. The schedule
also stores the metric-aware upstream predecessor used by three-point fluxes,
with an upwind fallback for degenerate hanging connections. It is built once;
staggered transport does not traverse the tree.
"""
struct StaggeredAdvectionWorkspace{
        Axis, T, M <: BlockTreeMesh, S <: AbstractDualAdvectionSchedule{T},
    }
    tendency::FaceField{Axis, T, M}
    rates::FaceField{Axis, T, M}
    schedule::S
    dual_flux::Vector{T}
    dual_offsets::Vector{Int}
    dual_faces::Vector{Int}
    dual_signs::Vector{Int8}
    sweep_forward::Base.RefValue{Bool}
    boundary_faces::Vector{BoundaryDualFace{T}}
    interior_aperture::Union{Nothing, Vector{T}}
    boundary_aperture::Union{Nothing, Vector{T}}
end

"""Workspace for simultaneous transport of both velocity components."""
struct StaggeredVelocityAdvectionWorkspace{T, M <: BlockTreeMesh, WX, WY}
    x::FaceField{1, T, M}
    y::FaceField{2, T, M}
    xworkspace::WX
    yworkspace::WY
    sweep_forward::Base.RefValue{Bool}
end


function _dual_adjacency(schedule, number_of_outputs)
    targets = Vector{Int}(undef, 2length(schedule.faces))
    dual_faces = Vector{Int}(undef, length(targets))
    dual_signs = Vector{Int8}(undef, length(targets))
    for (index, face) in pairs(schedule.faces)
        targets[2index - 1] = face.minus
        targets[2index] = face.plus
        dual_faces[2index - 1] = dual_faces[2index] = index
        dual_signs[2index - 1] = 1
        dual_signs[2index] = -1
    end
    offsets, permutation = _grouped_output_schedule(targets, number_of_outputs)
    return offsets, dual_faces[permutation], dual_signs[permutation]
end

function StaggeredAdvectionWorkspace(
        tendency::FaceField{Axis, T, M}, rates::FaceField{Axis, T, M},
        schedule::S, sweep_forward::Base.RefValue{Bool}; apertures = nothing,
    ) where {Axis, T, M, S <: AbstractDualAdvectionSchedule{T}}
    number_of_outputs = schedule isa AMRDualAdvectionSchedule ?
        length(schedule.volumes) : length(tendency.interior)
    offsets, faces, signs = _dual_adjacency(schedule, number_of_outputs)
    interior_aperture, boundary_aperture = isnothing(apertures) ?
        (nothing, nothing) : apertures
    return StaggeredAdvectionWorkspace(
        tendency, rates, schedule, zeros(T, length(schedule.faces)),
        offsets, faces, signs,
        sweep_forward, _boundary_dual_faces(tendency.mesh, Val(Axis), T),
        interior_aperture, boundary_aperture,
    )
end

function StaggeredAdvectionWorkspace(
        tendency::FaceField{Axis, T, M}, rates::FaceField{Axis, T, M},
        graph::PhaseTransportConnectivity{Axis, T, S},
        sweep_forward::Base.RefValue{Bool}; reuse_flux = nothing,
    ) where {Axis, T, M, S}
    flux = reuse_flux isa Vector{T} ? reuse_flux : T[]
    resize!(flux, length(graph.schedule.faces))
    fill!(flux, zero(T))
    return StaggeredAdvectionWorkspace(
        tendency, rates, graph.schedule, flux,
        graph.dual_offsets, graph.dual_faces, graph.dual_signs, sweep_forward,
        graph.boundary_faces, nothing, nothing,
    )
end

function _phase_transport_connectivity(mesh::BlockTreeMesh{MT}, ::Val{Axis}, ::Type{T}) where {MT, Axis, T}
    cache = mesh.topology_cache
    graph = cache.phase_transport[Axis]
    graph isa PhaseTransportConnectivity{Axis, T} && return graph
    # Bounded phase transport includes boundary half-volumes; both schedules
    # retain the regular-face classification and three-point stencils.
    schedule = all(mesh.periodic) ?
        _dual_advection_schedule(mesh, Val(Axis), T, true) :
        _bounded_phase_schedule(mesh, Val(Axis), T)
    interior = Int[index for (index, face) in pairs(mesh.faces) if face.axis == Axis]
    boundary = Int[index for (index, face) in pairs(mesh.boundary_faces) if face.axis == Axis]
    centers = NTuple{2, T}[mesh.faces[index].center for index in interior]
    append!(centers, (mesh.boundary_faces[index].center for index in boundary))
    outputs = schedule isa AMRDualAdvectionSchedule ? length(schedule.volumes) : length(interior)
    offsets, faces, signs = _dual_adjacency(schedule, outputs)
    graph = PhaseTransportConnectivity{Axis, T, typeof(schedule)}(
        cache.generation, schedule, offsets, faces, signs,
        _boundary_dual_faces(mesh, Val(Axis), T), interior, boundary, centers
    )
    # Foreign numeric types not represented in the bounded cache are still
    # supported; only their optional connectivity reuse is skipped.
    cache.phase_transport = Base.setindex(
        cache.phase_transport,
        graph isa _PhaseTransportCacheEntry{Axis, MT} ? graph : nothing, Axis
    )
    return graph
end

function _boundary_dual_faces(
        mesh::BlockTreeMesh, ::Val{Axis}, ::Type{T},
    ) where {Axis, T}
    nx, ny = mesh.block_size
    incidences, _, _ = _component_incidences(mesh, Val(Axis), T)
    boundary_local = zeros(Int, length(mesh.boundary_faces))
    local_index = 0
    for (index, face) in pairs(mesh.boundary_faces)
        face.axis == Axis || continue
        local_index += 1
        boundary_local[index] = local_index
    end
    connections = BoundaryDualFace{T}[]
    for (boundary_index, boundary_face) in pairs(mesh.boundary_faces)
        cell_index = _cell_index(boundary_face.cell, nx, ny)
        normal_component = boundary_face.axis == Axis
        for incidence in incidences[cell_index]
            if normal_component
                incidence.side == -boundary_face.side || continue
                component_face = mesh.faces[incidence.global_index]
                push!(
                    connections, BoundaryDualFace(
                        incidence.local_index, boundary_index,
                        boundary_local[boundary_index], incidence.global_index,
                        boundary_face.axis, boundary_face.side,
                        T(component_face.area), true,
                    )
                )
            else
                push!(
                    connections, BoundaryDualFace(
                        incidence.local_index, boundary_index, 0, 0,
                        boundary_face.axis, boundary_face.side,
                        T(boundary_face.area) * incidence.fraction / 2, false,
                    )
                )
            end
        end
    end
    iszero(local_index) && return connections
    # Complete transverse connections against boundary component faces. At a
    # level transition a fine interior component face may meet the coarse
    # cell's wall-side half-volume, even though that wall face is not evolved
    # by single-phase convection. Store that endpoint as a boundary value.
    for (index, face) in pairs(mesh.boundary_faces)
        face.axis == Axis || continue
        block = mesh.blocks[face.cell.block]
        push!(
            incidences[_cell_index(face.cell, nx, ny)],
            ComponentFaceIncidence(
                -boundary_local[index], length(mesh.faces) + index,
                face.side, T(face.area / block.spacing[3 - Axis]),
            ),
        )
    end
    for (flux_index, face) in pairs(mesh.faces)
        face.axis == Axis && continue
        minus = incidences[_cell_index(face.minus, nx, ny)]
        plus = incidences[_cell_index(face.plus, nx, ny)]
        for side in (-1, 1), a in minus, b in plus
            a.side == side == b.side || continue
            if a.local_index > 0 && b.local_index < 0
                interior, wall, sign = a, b, 1
            elseif a.local_index < 0 && b.local_index > 0
                interior, wall, sign = b, a, -1
            else
                continue
            end
            # Here wall_axis identifies the transverse flux/sweep direction
            # and side is its outward orientation from the interior unknown;
            # boundary_index still identifies the physical component wall.
            push!(
                connections, BoundaryDualFace(
                    interior.local_index, wall.global_index - length(mesh.faces),
                    -wall.local_index, flux_index, face.axis, sign,
                    T(face.area) * a.fraction * b.fraction / 2, true,
                ),
            )
        end
    end
    return connections
end


function _uniform_periodic_cell_faces(mesh::BlockTreeMesh)
    all(mesh.periodic) || throw(
        ArgumentError(
            "staggered advection currently requires periodic boundaries in both directions"
        )
    )
    level = first(mesh.blocks).key.level
    all(block -> block.key.level == level, mesh.blocks) || throw(
        ArgumentError(
            "staggered advection currently requires a uniform tree level"
        )
    )

    nx, ny = mesh.block_size
    n = ncells(mesh)
    xminus = zeros(Int, n)
    xplus = zeros(Int, n)
    yminus = zeros(Int, n)
    yplus = zeros(Int, n)
    for (index, face) in pairs(mesh.faces)
        minus = _cell_index(face.minus, nx, ny)
        plus = _cell_index(face.plus, nx, ny)
        if face.axis == 1
            iszero(xplus[minus]) || throw(ArgumentError("nonconforming x-face schedule"))
            iszero(xminus[plus]) || throw(ArgumentError("nonconforming x-face schedule"))
            xplus[minus] = index
            xminus[plus] = index
        else
            iszero(yplus[minus]) || throw(ArgumentError("nonconforming y-face schedule"))
            iszero(yminus[plus]) || throw(ArgumentError("nonconforming y-face schedule"))
            yplus[minus] = index
            yminus[plus] = index
        end
    end
    all(>(0), xminus) && all(>(0), xplus) &&
        all(>(0), yminus) && all(>(0), yplus) ||
        throw(ArgumentError("incomplete periodic face schedule"))
    return xminus, xplus, yminus, yplus
end

function _component_face_indices(mesh::BlockTreeMesh, axis::Int)
    global_to_local = zeros(Int, length(mesh.faces))
    local_index = 0
    for (global_index, face) in pairs(mesh.faces)
        face.axis == axis || continue
        local_index += 1
        global_to_local[global_index] = local_index
    end
    return global_to_local, local_index
end

function _component_face_centres(mesh::BlockTreeMesh, axis::Int, ::Type{T}) where {T}
    centres = NTuple{2, T}[]
    sizehint!(centres, count(face -> face.axis == axis, mesh.faces))
    for face in mesh.faces
        face.axis == axis || continue
        push!(centres, (T(face.center[1]), T(face.center[2])))
    end
    return centres
end

@inline function _dual_displacement(mesh::BlockTreeMesh, first, second)
    return (
        _wrapped_delta(second[1] - first[1], mesh.extent[1], mesh.periodic[1]),
        _wrapped_delta(second[2] - first[2], mesh.extent[2], mesh.periodic[2]),
    )
end

function _three_point_stencils(
        mesh::BlockTreeMesh, centres,
        connections
    )
    T = typeof(first(centres)[1])
    # Compressed adjacency avoids one growable Vector allocation per face.
    # This graph is temporary and was the dominant allocation-count increase
    # when three-point metadata was added to AMR workspace construction.
    adjacency_cursor = zeros(Int, length(centres))
    for connection in connections
        adjacency_cursor[connection.minus] += 1
        adjacency_cursor[connection.plus] += 1
    end
    adjacency_offsets = _csr_offsets(adjacency_cursor)
    copyto!(adjacency_cursor, 1, adjacency_offsets, 1, length(adjacency_cursor))
    adjacency = Vector{Int}(undef, adjacency_offsets[end] - 1)
    for (connection_index, connection) in pairs(connections)
        adjacency[adjacency_cursor[connection.minus]] = connection_index
        adjacency_cursor[connection.minus] += 1
        adjacency[adjacency_cursor[connection.plus]] = connection_index
        adjacency_cursor[connection.plus] += 1
    end

    function outer_stencil(
            connection_index, endpoint::Int, neighbour::Int,
            direction
        )
        connection = connections[connection_index]
        axis = connection.axis
        tangent_axis = 3 - axis
        displacement = _dual_displacement(
            mesh, centres[endpoint], centres[neighbour]
        )
        axial_distance = abs(displacement[axis])
        tolerance = 64eps(T) * max(one(T), mesh.extent[axis])
        axial_distance > tolerance || return ThreePointStencil(0, zero(T))

        # Continue the algebraic edge backwards through the endpoint. At a
        # hanging face several graph neighbours can exist, so choose the one
        # nearest the reflected target while requiring the correct upstream
        # half-space. This is deterministic and uses geometry only while the
        # schedule is built; the time-step loop retains a three-value stencil.
        best_neighbour = 0
        best_distance = T(Inf)
        best_axial_distance = zero(T)
        adjacency_stop = adjacency_offsets[endpoint + 1] - 1
        adjacency_range = adjacency_offsets[endpoint]:adjacency_stop
        for adjacency_index in adjacency_range
            candidate_connection_index = adjacency[adjacency_index]
            candidate_connection_index == connection_index && continue
            candidate_connection = connections[candidate_connection_index]
            candidate_connection.axis == axis || continue
            candidate = candidate_connection.minus == endpoint ?
                candidate_connection.plus : candidate_connection.minus
            candidate_displacement = _dual_displacement(
                mesh, centres[endpoint], centres[candidate]
            )
            direction * candidate_displacement[axis] > tolerance || continue
            axial = abs(candidate_displacement[axis])
            target_axis = direction * axial_distance
            target_tangent = -displacement[tangent_axis]
            distance = (candidate_displacement[axis] - target_axis)^2 +
                (candidate_displacement[tangent_axis] - target_tangent)^2
            if distance < best_distance ||
                    (distance == best_distance && candidate < best_neighbour)
                best_neighbour = candidate
                best_distance = distance
                best_axial_distance = axial
            end
        end
        iszero(best_neighbour) && return ThreePointStencil(0, zero(T))
        return ThreePointStencil(
            best_neighbour, T(axial_distance / best_axial_distance)
        )
    end

    # Each connection owns its stencil pair; the searches only read.
    result = Vector{NTuple{2, ThreePointStencil{T}}}(undef, length(connections))
    _parallel_for(length(connections)) do index
        connection = connections[index]
        displacement = _dual_displacement(
            mesh, centres[connection.minus], centres[connection.plus]
        )
        direction = sign(displacement[connection.axis])
        if iszero(direction)
            empty_stencil = ThreePointStencil(0, zero(T))
            result[index] = (empty_stencil, empty_stencil)
        else
            result[index] = (
                outer_stencil(index, connection.minus, connection.plus, -direction),
                outer_stencil(index, connection.plus, connection.minus, direction),
            )
        end
    end
    return result
end

function _empty_three_point_stencils(::Type{T}, count::Int) where {T}
    empty_stencil = ThreePointStencil(0, zero(T))
    return fill((empty_stencil, empty_stencil), count)
end

function _uniform_dual_schedule(
        mesh::BlockTreeMesh, ::Val{Axis}, ::Type{T};
        build_three_point_stencils::Bool = true
    ) where {Axis, T}
    xminus, xplus, yminus, yplus = _uniform_periodic_cell_faces(mesh)
    global_to_local, component_count = _component_face_indices(mesh, Axis)

    nx, ny = mesh.block_size
    n = ncells(mesh)
    dx, dy = first(mesh.blocks).spacing
    raw_faces = Vector{RawDualFace{T}}(undef, 2n)
    for cell in 1:n
        right = _cell_index(mesh.faces[xplus[cell]].plus, nx, ny)
        above = _cell_index(mesh.faces[yplus[cell]].plus, nx, ny)
        if Axis == 1
            # Normal dual face through the cell centre.
            raw_faces[cell] = RawDualFace(
                global_to_local[xminus[cell]], global_to_local[xplus[cell]],
                xminus[cell], xplus[cell], 1, T(dy), T(dx)
            )
            # Transverse dual face through the cell's upper-right vertex.
            raw_faces[n + cell] = RawDualFace(
                global_to_local[xplus[cell]], global_to_local[xplus[above]],
                yplus[cell], yplus[right], 2, T(dx), T(dy)
            )
        else
            # Normal dual face through the cell centre.
            raw_faces[cell] = RawDualFace(
                global_to_local[yminus[cell]], global_to_local[yplus[cell]],
                yminus[cell], yplus[cell], 2, T(dx), T(dy)
            )
            # Transverse dual face through the cell's upper-right vertex.
            raw_faces[n + cell] = RawDualFace(
                global_to_local[yplus[cell]], global_to_local[yplus[right]],
                xplus[cell], xplus[above], 1, T(dy), T(dx)
            )
        end
    end

    stencils = if build_three_point_stencils
        centres = _component_face_centres(mesh, Axis, T)
        connections = [
            (minus = face.minus, plus = face.plus, axis = face.axis)
                for face in raw_faces
        ]
        _three_point_stencils(mesh, centres, connections)
    else
        _empty_three_point_stencils(T, length(raw_faces))
    end
    dual_faces = Vector{DualFace{T}}(undef, length(raw_faces))
    for (index, face) in pairs(raw_faces)
        minus_stencil, plus_stencil = stencils[index]
        dual_faces[index] = DualFace(
            face.minus, face.plus, face.flux_a, face.flux_b,
            face.axis, face.area, face.distance,
            minus_stencil, plus_stencil
        )
    end
    any(face -> iszero(face.minus) || iszero(face.plus), dual_faces) &&
        throw(ArgumentError("invalid directional face mapping"))
    return UniformDualAdvectionSchedule(dual_faces, T(dx * dy))
end

"""
    _AMRFluxWeights{T}

Algebraic flux weights of an AMR dual schedule, grouped by the lower dual
unknown `minus` of each key `(minus, plus, flux)` (`minus < plus`): group
`minus` holds `counts[minus]` distinct keys at `offsets[minus]` onwards in
`plus`, `flux` and `weights`, sorted by `(plus, axis of flux, flux)`. A weight
is the sum of its contributions in generation order, starting from zero; a
contribution oriented from the higher to the lower unknown enters with a
minus sign. Zero sums are kept.
"""
struct _AMRFluxWeights{T}
    offsets::Vector{Int}
    counts::Vector{Int}
    plus::Vector{Int}
    flux::Vector{Int}
    weights::Vector{T}
end

"""Number of flux-weight contributions of interior primal face `flux_index`
to the dual schedule of component `Axis` (see `_flux_weight_contributions!`)."""
@inline function _flux_weight_count(
        faces, incidences, nx, ny, ::Val{Axis}, flux_index,
    ) where {Axis}
    flux_face = @inbounds faces[flux_index]
    contributions = 0
    if flux_face.axis == Axis
        for cell in (flux_face.minus, flux_face.plus)
            for component_face in @inbounds incidences[_cell_index(cell, nx, ny)]
                component_face.global_index == flux_index || (contributions += 1)
            end
        end
    else
        minus_faces = @inbounds incidences[_cell_index(flux_face.minus, nx, ny)]
        plus_faces = @inbounds incidences[_cell_index(flux_face.plus, nx, ny)]
        for side in (-1, 1), minus_face in minus_faces, plus_face in plus_faces
            minus_face.side == side == plus_face.side && (contributions += 1)
        end
    end
    return contributions
end

@inline function _store_flux_weight!(
        minus, plus, coefficients, position, first::Int, second::Int, coefficient,
    )
    # A degenerate connection stores `minus == 0`; the caller throws.
    if first < second
        @inbounds minus[position], plus[position] = first, second
        @inbounds coefficients[position] = coefficient
    else
        @inbounds minus[position] = first == second ? 0 : second
        @inbounds plus[position] = first
        # `x - c` and `x + (-c)` are the same IEEE operation.
        @inbounds coefficients[position] = -coefficient
    end
    return position + 1
end

"""Store the flux-weight contributions of interior primal face `flux_index`
from `position` on. A normal primal flux is routed through the two cells
adjacent to it: stars centred on the transported face reproduce the regular
two-point average and remain valid when either cell side splits. Across a
transverse primal face the same normal side of both cells is coupled: the
product weights are a conservative transport plan between possibly split side
partitions, whose marginals are exactly one half of the incidence fractions."""
@inline function _flux_weight_contributions!(
        minus, plus, coefficients, position, faces, incidences, global_to_local,
        nx, ny, ::Val{Axis}, flux_index,
    ) where {Axis}
    flux_face = @inbounds faces[flux_index]
    T = eltype(coefficients)
    if flux_face.axis == Axis
        centre = @inbounds global_to_local[flux_index]
        for (cell, sign) in ((flux_face.minus, one(T)), (flux_face.plus, -one(T)))
            for component_face in @inbounds incidences[_cell_index(cell, nx, ny)]
                component_face.global_index == flux_index && continue
                position = _store_flux_weight!(
                    minus, plus, coefficients, position, component_face.local_index,
                    centre, sign * component_face.fraction / 2,
                )
            end
        end
    else
        minus_faces = @inbounds incidences[_cell_index(flux_face.minus, nx, ny)]
        plus_faces = @inbounds incidences[_cell_index(flux_face.plus, nx, ny)]
        for side in (-1, 1), minus_face in minus_faces, plus_face in plus_faces
            minus_face.side == side == plus_face.side || continue
            position = _store_flux_weight!(
                minus, plus, coefficients, position, minus_face.local_index,
                plus_face.local_index, minus_face.fraction * plus_face.fraction / 2,
            )
        end
    end
    return position
end

"""Fixed number of dual unknowns per chunk when sorting flux-weight groups."""
const _FLUX_WEIGHT_GROUP_CHUNK = 256

"""
    _amr_flux_weights(mesh, Val(Axis), T, incidences, global_to_local, number_of_unknowns)

Flux weights of the AMR dual schedule of component `Axis` from the cell
`incidences` of its `number_of_unknowns` dual unknowns. The contributions of
each interior primal face are counted, placed by a prefix sum and stored in
face order, threaded over faces; a stable counting sort groups them by their
lower unknown, and each group is sorted stably and summed per key, threaded
over fixed chunks of groups. The summation order is the face order of the
contributions, so the weights are the same for every thread count.
"""
function _amr_flux_weights(
        mesh::BlockTreeMesh, ::Val{Axis}, ::Type{T}, incidences, global_to_local,
        number_of_unknowns::Int,
    ) where {Axis, T}
    nx, ny = mesh.block_size
    faces = mesh.faces
    number_of_faces = length(faces)
    face_counts = Vector{Int}(undef, number_of_faces)
    @_parallel for flux_index in 1:number_of_faces
        @inbounds face_counts[flux_index] = _flux_weight_count(
            faces, incidences, nx, ny, Val(Axis), flux_index,
        )
    end
    face_offsets = _csr_offsets(face_counts)
    number_of_contributions = face_offsets[end] - 1
    contribution_minus = Vector{Int}(undef, number_of_contributions)
    contribution_plus = Vector{Int}(undef, number_of_contributions)
    contribution_weights = Vector{T}(undef, number_of_contributions)
    @_parallel for flux_index in 1:number_of_faces
        _flux_weight_contributions!(
            contribution_minus, contribution_plus, contribution_weights,
            face_offsets[flux_index], faces, incidences, global_to_local,
            nx, ny, Val(Axis), flux_index,
        )
    end
    any(iszero, contribution_minus) &&
        throw(ArgumentError("degenerate AMR dual connection"))
    # The group counts below are indexed without bounds checks. A lower
    # unknown is a positive component-local index by construction; the
    # degenerate zero was rejected above.
    !isempty(contribution_minus) &&
        maximum(contribution_minus) > number_of_unknowns && throw(
        ArgumentError("AMR dual connection lies outside the dual unknowns"),
    )
    # Stable counting sort by the lower unknown (`face_counts` is reused for
    # the contribution order).
    group_counts = zeros(Int, number_of_unknowns)
    for minus in contribution_minus
        @inbounds group_counts[minus] += 1
    end
    offsets = _csr_offsets(group_counts)
    copyto!(group_counts, 1, offsets, 1, number_of_unknowns)
    plus = Vector{Int}(undef, number_of_contributions)
    flux = Vector{Int}(undef, number_of_contributions)
    weights = Vector{T}(undef, number_of_contributions)
    for flux_index in 1:number_of_faces, contribution in
            face_offsets[flux_index]:(face_offsets[flux_index + 1] - 1)
        minus = @inbounds contribution_minus[contribution]
        position = @inbounds group_counts[minus]
        @inbounds group_counts[minus] = position + 1
        @inbounds plus[position] = contribution_plus[contribution]
        @inbounds flux[position] = flux_index
        @inbounds weights[position] = contribution_weights[contribution]
    end
    chunks = cld(number_of_unknowns, _FLUX_WEIGHT_GROUP_CHUNK)
    @_parallel work = _FLUX_WEIGHT_GROUP_CHUNK for chunk in 1:chunks
        last_group = min(number_of_unknowns, chunk * _FLUX_WEIGHT_GROUP_CHUNK)
        for group in ((chunk - 1) * _FLUX_WEIGHT_GROUP_CHUNK + 1):last_group
            @inbounds group_counts[group] = _combine_flux_weight_group!(
                plus, flux, weights, faces, offsets[group], offsets[group + 1] - 1,
            )
        end
    end
    return _AMRFluxWeights{T}(offsets, group_counts, plus, flux, weights)
end

@inline _flux_weight_key(plus, flux, faces, position) =
    (@inbounds(plus[position]), @inbounds(faces[flux[position]].axis), @inbounds(flux[position]))

"""Sort one group of contributions stably by `(plus, axis, flux)` (insertion
sort; groups are short) and sum each key's contributions in order into the
group's leading entries. Return the number of distinct keys."""
@inline function _combine_flux_weight_group!(plus, flux, weights, faces, start, stop)
    for position in (start + 1):stop
        key = _flux_weight_key(plus, flux, faces, position)
        moving_plus = @inbounds plus[position]
        moving_flux = @inbounds flux[position]
        moving_weight = @inbounds weights[position]
        target = position
        while target > start && _flux_weight_key(plus, flux, faces, target - 1) > key
            @inbounds plus[target] = plus[target - 1]
            @inbounds flux[target] = flux[target - 1]
            @inbounds weights[target] = weights[target - 1]
            target -= 1
        end
        @inbounds plus[target] = moving_plus
        @inbounds flux[target] = moving_flux
        @inbounds weights[target] = moving_weight
    end
    distinct = 0
    position = start
    while position <= stop
        key_plus = @inbounds plus[position]
        key_flux = @inbounds flux[position]
        weight = zero(eltype(weights))
        while position <= stop && (@inbounds plus[position]) == key_plus &&
                (@inbounds flux[position]) == key_flux
            weight += @inbounds weights[position]
            position += 1
        end
        @inbounds plus[start + distinct] = key_plus
        @inbounds flux[start + distinct] = key_flux
        @inbounds weights[start + distinct] = weight
        distinct += 1
    end
    return distinct
end

"""Combined weight of the key `(minus, plus, flux)` of `weights`."""
function _amr_flux_weight(weights::_AMRFluxWeights, minus, plus, flux)
    start = weights.offsets[minus]
    for position in start:(start + weights.counts[minus] - 1)
        weights.plus[position] == plus && weights.flux[position] == flux &&
            return weights.weights[position]
    end
    throw(KeyError((minus, plus, flux)))
end

function _component_incidences(mesh::BlockTreeMesh, ::Val{Axis}, ::Type{T}) where {Axis, T}
    nx, ny = mesh.block_size
    global_to_local, component_count = _component_face_indices(mesh, Axis)
    incidences = [ComponentFaceIncidence{T}[] for _ in 1:ncells(mesh)]
    volumes = Vector{T}(undef, component_count)
    tangent_axis = 3 - Axis
    lattice = _finest_lattice_size(mesh)
    for (global_index, face) in pairs(mesh.faces)
        face.axis == Axis || continue
        local_index = global_to_local[global_index]
        _, lo, hi = _face_lattice_geometry(mesh, face, lattice)
        for (cell, side) in ((face.minus, 1), (face.plus, -1))
            cell_index = _cell_index(cell, nx, ny)
            _, scale = _cell_lattice_bounds(mesh, mesh.blocks, cell, lattice)
            fraction = T(hi - lo) / T(scale[tangent_axis])
            push!(
                incidences[cell_index], ComponentFaceIncidence(
                    local_index, global_index, side, fraction
                )
            )
        end
        # Advection's interior dual volume is the same mesh-only quantity
        # used by diffusion, projection and diagnostics. The face order here
        # is the component-local subsequence of the global face order.
        volumes[local_index] = T(mesh.face_dual_volumes[global_index])
    end

    tolerance = 512eps(T)
    for cell_faces in incidences, side in (-1, 1)
        fraction_sum = sum(
            face.fraction for face in cell_faces if face.side == side;
            init = zero(T)
        )
        # A zero sum is the physical wall normal to the transported component;
        # its prescribed face value is not an evolved dual-volume unknown.
        (
            isapprox(fraction_sum, one(T); atol = tolerance, rtol = tolerance) ||
                isapprox(fraction_sum, zero(T); atol = tolerance, rtol = tolerance)
        ) ||
            throw(ArgumentError("nonconforming AMR component-face partition"))
    end
    all(>(zero(T)), volumes) ||
        throw(ArgumentError("AMR staggered control volume has non-positive size"))
    return incidences, global_to_local, volumes
end

function _amr_dual_schedule(
        mesh::BlockTreeMesh, ::Val{Axis}, ::Type{T};
        build_three_point_stencils::Bool = true
    ) where {Axis, T}
    incidences, global_to_local, volumes =
        _component_incidences(mesh, Val(Axis), T)
    weights = _amr_flux_weights(
        mesh, Val(Axis), T, incidences, global_to_local, length(volumes),
    )
    return _assemble_amr_dual_schedule(
        mesh, Val(Axis), T, weights, volumes; build_three_point_stencils,
    )
end

"""Count (`outputs === nothing`) or fill the dual connections of the
nonzero flux weights of group `group` (see `_assemble_amr_dual_schedule`).
Return the numbers of flux weights and connections of the group."""
@inline function _dual_group_connections!(
        outputs, group, weights::_AMRFluxWeights{T}, faces, dual_sizes, interior_count,
    ) where {T}
    group_plus, group_flux, group_weights = weights.plus, weights.flux, weights.weights
    start = @inbounds weights.offsets[group]
    stop = start + (@inbounds weights.counts[group]) - 1
    if outputs !== nothing
        flux_indices, flux_weights, dual_connections, entry_offsets,
            connection_offsets = outputs
        first_entry = @inbounds entry_offsets[group]
        first_connection = @inbounds connection_offsets[group]
    end
    entries = 0
    connections = 0
    position = start
    while position <= stop
        if iszero(@inbounds group_weights[position])
            position += 1
            continue
        end
        # Consecutive nonzero keys with the same upper unknown and flux axis
        # form one dual connection; zero keys are skipped.
        plus = @inbounds group_plus[position]
        axis = @inbounds faces[group_flux[position]].axis
        connection_entries = entries
        refinement_interface = group > interior_count || plus > interior_count
        while position <= stop && (@inbounds group_plus[position]) == plus &&
                (@inbounds faces[group_flux[position]].axis) == axis
            weight = @inbounds group_weights[position]
            if !iszero(weight)
                if outputs !== nothing
                    @inbounds flux_indices[first_entry + entries] = group_flux[position]
                    @inbounds flux_weights[first_entry + entries] = weight
                end
                refinement_interface |= abs(weight) != T(0.5)
                entries += 1
            end
            position += 1
        end
        weight_count = entries - connection_entries
        refinement_interface |= weight_count != 2
        if group <= interior_count && plus <= interior_count
            refinement_interface |=
                (@inbounds dual_sizes[group]) != (@inbounds dual_sizes[plus])
        end
        if outputs !== nothing
            @inbounds dual_connections[first_connection + connections] = RawAMRDualFace(
                group, plus, first_entry + connection_entries, weight_count, axis,
                refinement_interface,
            )
        end
        connections += 1
    end
    return entries, connections
end

"""
Assemble the AMR dual schedule of component `Axis` from its flux weights:
one dual connection per run of nonzero keys `(minus, plus, flux)` with equal
`minus`, `plus` and flux axis, in key order. The connections of each lower
unknown are counted, placed by a prefix sum and filled, threaded over fixed
chunks of unknowns, so the schedule is the same for every thread count.
"""
function _assemble_amr_dual_schedule(
        mesh::BlockTreeMesh, ::Val{Axis}, ::Type{T}, weights::_AMRFluxWeights{T}, volumes;
        build_three_point_stencils::Bool = true,
        interior_count::Int = length(volumes),
    ) where {Axis, T}

    # Twice each component-face dual volume, in finest-cell area units.
    # Comparing these integers identifies refinement interfaces independently
    # of physical extent and roundoff in the metric volumes.
    lattice = _finest_lattice_size(mesh)
    faces = mesh.faces
    face_offset = _axis_face_offset(faces, Axis, interior_count)
    dual_sizes = Vector{Int}(undef, interior_count)
    _parallel_for(interior_count) do local_index
        face = faces[face_offset + local_index]
        _, lo, hi = _face_lattice_geometry(mesh, face, lattice)
        _, minus_scale = _cell_lattice_bounds(mesh, mesh.blocks, face.minus, lattice)
        _, plus_scale = _cell_lattice_bounds(mesh, mesh.blocks, face.plus, lattice)
        dual_sizes[local_index] = (hi - lo) * (minus_scale[Axis] + plus_scale[Axis])
    end

    number_of_groups = length(weights.counts)
    group_entries = Vector{Int}(undef, number_of_groups)
    group_connections = Vector{Int}(undef, number_of_groups)
    chunks = cld(number_of_groups, _FLUX_WEIGHT_GROUP_CHUNK)
    @_parallel work = _FLUX_WEIGHT_GROUP_CHUNK for chunk in 1:chunks
        last_group = min(number_of_groups, chunk * _FLUX_WEIGHT_GROUP_CHUNK)
        for group in ((chunk - 1) * _FLUX_WEIGHT_GROUP_CHUNK + 1):last_group
            entries, connections = _dual_group_connections!(
                nothing, group, weights, faces, dual_sizes, interior_count,
            )
            @inbounds group_entries[group] = entries
            @inbounds group_connections[group] = connections
        end
    end
    entry_offsets = _csr_offsets(group_entries)
    connection_offsets = _csr_offsets(group_connections)
    flux_indices = Vector{Int}(undef, entry_offsets[end] - 1)
    flux_weights = Vector{T}(undef, length(flux_indices))
    dual_connections = Vector{RawAMRDualFace}(undef, connection_offsets[end] - 1)
    @_parallel work = _FLUX_WEIGHT_GROUP_CHUNK for chunk in 1:chunks
        last_group = min(number_of_groups, chunk * _FLUX_WEIGHT_GROUP_CHUNK)
        outputs = (
            flux_indices, flux_weights, dual_connections, entry_offsets,
            connection_offsets,
        )
        for group in ((chunk - 1) * _FLUX_WEIGHT_GROUP_CHUNK + 1):last_group
            _dual_group_connections!(
                outputs, group, weights, faces, dual_sizes, interior_count,
            )
        end
    end

    stencils = if build_three_point_stencils
        centres = _component_face_centres(mesh, Axis, T)
        if interior_count < length(volumes)
            append!(
                centres,
                (face.center for face in mesh.boundary_faces if face.axis == Axis),
            )
        end
        _three_point_stencils(mesh, centres, dual_connections)
    else
        _empty_three_point_stencils(T, length(dual_connections))
    end
    dual_faces = Vector{AMRDualFace{T}}(undef, length(dual_connections))
    for (index, connection) in pairs(dual_connections)
        minus_stencil, plus_stencil = stencils[index]
        dual_faces[index] = AMRDualFace(
            connection.minus, connection.plus, connection.first_weight,
            connection.weight_count, connection.axis,
            connection.refinement_interface, minus_stencil, plus_stencil
        )
    end
    return AMRDualAdvectionSchedule(
        dual_faces, flux_indices, flux_weights, volumes
    )
end

function _dual_advection_schedule(
        mesh::BlockTreeMesh, ::Val{Axis}, ::Type{T},
        build_three_point_stencils::Bool,
    ) where {Axis, T}
    first_level = first(mesh.blocks).key.level
    uniform = all(block -> block.key.level == first_level, mesh.blocks)
    return uniform && all(mesh.periodic) ?
        _uniform_dual_schedule(
            mesh, Val(Axis), T; build_three_point_stencils
        ) :
        _amr_dual_schedule(
            mesh, Val(Axis), T; build_three_point_stencils
        )
end

function _staggered_advection_workspace(
        field::FaceField{Axis, T, M}, build_three_point_stencils::Bool
    ) where {Axis, T, M}
    mesh = field.mesh
    _, component_count = _component_face_indices(mesh, Axis)
    component_count == length(field.interior) ||
        throw(ArgumentError("face field does not match the mesh schedule"))
    schedule = _dual_advection_schedule(
        mesh, Val(Axis), T, build_three_point_stencils,
    )
    tendency = FaceField{Axis}(mesh, T)
    rates = FaceField{Axis}(mesh, T)
    return StaggeredAdvectionWorkspace(tendency, rates, schedule, Ref(true))
end

StaggeredAdvectionWorkspace(field::FaceField) =
    _staggered_advection_workspace(field, true)

function StaggeredVelocityAdvectionWorkspace(velocity::FaceVelocity{T, M}) where {T, M}
    x = FaceField{1}(velocity)
    y = FaceField{2}(velocity)
    return StaggeredVelocityAdvectionWorkspace(
        x, y, StaggeredAdvectionWorkspace(x), StaggeredAdvectionWorkspace(y),
        Ref(true)
    )
end

StaggeredAdvectionWorkspace(velocity::FaceVelocity) =
    StaggeredVelocityAdvectionWorkspace(velocity)

@inline _dual_state(
    qminus, qplus, flux, divergence, dt, face, ::Upwind,
    values
) =
    ifelse(flux >= zero(flux), qminus, qplus)

@inline _dual_state(
    qminus, qplus, flux, divergence, dt, face, ::Centred,
    values
) =
    (qminus + qplus) / 2

@inline function _dual_state(
        qminus, qplus, flux, divergence, dt,
        face, ::LaxWendroff, values
    )
    normal_speed = flux / face.area
    normal_courant = abs(dt * normal_speed / face.distance)
    upwind, downwind = flux >= zero(flux) ? (qminus, qplus) : (qplus, qminus)
    return (
        (one(normal_courant) - normal_courant) * downwind +
            (one(normal_courant) + normal_courant) * upwind
    ) / 2 -
        dt * (
        qminus * divergence[face.minus] +
            qplus * divergence[face.plus]
    ) / 4
end

@inline _flux_limiter(r, ::Fromm) = (one(r) + r) / 2
@inline _flux_limiter(r, ::Minmod) = max(zero(r), min(one(r), r))
@inline _flux_limiter(r, ::MonotonizedCentral) =
    max(zero(r), min(2r, (one(r) + r) / 2, oftype(r, 2)))
@inline function _flux_limiter(r, ::VanLeer)
    r <= zero(r) && return zero(r)
    isinf(r) && return oftype(r, 2)
    return 2r / (one(r) + r)
end
@inline _flux_limiter(r, ::Superbee) =
    max(zero(r), min(2r, one(r)), min(r, oftype(r, 2)))

@inline function _three_point_state(
        upwind, downwind, upstream,
        ratio_scale, high_order,
        scheme::AbstractThreePointFluxScheme
    )
    downstream_difference = downwind - upwind
    upstream_difference = upwind - upstream
    scale = max(abs(upstream), abs(upwind), abs(downwind), one(upwind))
    tolerance = 16eps(typeof(float(scale))) * scale
    smoothness = if abs(downstream_difference) <= tolerance
        abs(upstream_difference) <= tolerance ? one(ratio_scale) : zero(ratio_scale)
    else
        ratio_scale * upstream_difference / downstream_difference
    end
    return upwind + _flux_limiter(smoothness, scheme) *
        (high_order - upwind)
end

@inline function _dual_state(
        qminus, qplus, flux, divergence, dt, face,
        scheme::AbstractThreePointFluxScheme, values
    )
    positive_flux = flux >= zero(flux)
    upwind, downwind = positive_flux ? (qminus, qplus) : (qplus, qminus)
    stencil = positive_flux ? face.minus_stencil : face.plus_stencil
    iszero(stencil.upstream) && return upwind
    high_order = _dual_state(
        qminus, qplus, flux, divergence, dt, face, LaxWendroff(), values
    )
    upstream = @inbounds values[stencil.upstream]
    return _three_point_state(
        upwind, downwind, upstream,
        stencil.ratio_scale, high_order, scheme
    )
end

struct ThreePointStencil{T}
    upstream::Int
    ratio_scale::T
end

struct DualFace{T}
    minus::Int
    plus::Int
    flux_a::Int
    flux_b::Int
    axis::Int
    area::T
    distance::T
    minus_stencil::ThreePointStencil{T}
    plus_stencil::ThreePointStencil{T}
end

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

abstract type AbstractDualAdvectionSchedule{T} end

struct UniformDualAdvectionSchedule{T} <: AbstractDualAdvectionSchedule{T}
    faces::Vector{DualFace{T}}
    volume::T
end

struct AMRDualFace{T}
    minus::Int
    plus::Int
    first_weight::Int
    weight_count::Int
    axis::Int
    refinement_interface::Bool
    minus_stencil::ThreePointStencil{T}
    plus_stencil::ThreePointStencil{T}
end

struct AMRDualAdvectionSchedule{T} <: AbstractDualAdvectionSchedule{T}
    faces::Vector{AMRDualFace{T}}
    flux_indices::Vector{Int}
    flux_weights::Vector{T}
    volumes::Vector{T}
end

struct ComponentFaceIncidence{T}
    local_index::Int
    global_index::Int
    side::Int
    fraction::T
end

struct BoundaryDualFace{T}
    degree_of_freedom::Int
    boundary_index::Int
    boundary_local_index::Int
    opposite_flux::Int
    wall_axis::Int
    side::Int
    measure::T
    normal_component::Bool
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
    boundary_conditions::Base.RefValue{Any}
end

"""Workspace for simultaneous transport of both velocity components."""
struct StaggeredVelocityAdvectionWorkspace{T, M <: BlockTreeMesh, WX, WY}
    x::FaceField{1, T, M}
    y::FaceField{2, T, M}
    xworkspace::WX
    yworkspace::WY
    sweep_forward::Base.RefValue{Bool}
end

"""Mesh-only phase transport graph; numerical fluxes/BCs are not shared."""
struct PhaseTransportConnectivity{Axis, T, S <: AbstractDualAdvectionSchedule{T}}
    generation::UInt
    schedule::S
    dual_offsets::Vector{Int}
    dual_faces::Vector{Int}
    dual_signs::Vector{Int8}
    boundary_faces::Vector{BoundaryDualFace{T}}
    interior_indices::Vector{Int}
    boundary_indices::Vector{Int}
    centers::Vector{NTuple{2, T}}
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
        schedule::S, sweep_forward::Base.RefValue{Bool},
    ) where {Axis, T, M, S <: AbstractDualAdvectionSchedule{T}}
    number_of_outputs = schedule isa AMRDualAdvectionSchedule ?
        length(schedule.volumes) : length(tendency.interior)
    offsets, faces, signs = _dual_adjacency(schedule, number_of_outputs)
    return StaggeredAdvectionWorkspace(
        tendency, rates, schedule, zeros(T, length(schedule.faces)),
        offsets, faces, signs,
        sweep_forward, _boundary_dual_faces(tendency.mesh, Val(Axis), T),
        Ref{Any}(nothing),
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
    return StaggeredAdvectionWorkspace(tendency, rates, graph.schedule, flux,
        graph.dual_offsets, graph.dual_faces, graph.dual_signs, sweep_forward,
        graph.boundary_faces, Ref{Any}(nothing))
end

function _phase_transport_connectivity(mesh, ::Val{Axis}, ::Type{T}) where {Axis, T}
    cache = mesh.topology_cache
    graph = cache.phase_transport[Axis]
    graph isa PhaseTransportConnectivity{Axis, T} && return graph
    # Select the actual policy first. In a bounded domain the general AMR
    # schedule's three-point stencils would be discarded and are never used.
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
        _boundary_dual_faces(mesh, Val(Axis), T), interior, boundary, centers)
    cache.phase_transport[Axis] = graph
    return graph
end

function _boundary_dual_faces(
        mesh::BlockTreeMesh, ::Val{Axis}, ::Type{T},
    ) where {Axis, T}
    nx, ny = mesh.block_size
    incidences, global_to_local, _ =
        _component_incidences(mesh, Val(Axis), T)
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
    return connections
end

@inline function _cell_index(ref::CellRef, nx::Int, ny::Int)
    return (ref.block - 1) * nx * ny + (ref.j - 1) * nx + ref.i
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
    adjacency_offsets = Vector{Int}(undef, length(centres) + 1)
    adjacency_offsets[1] = 1
    for index in eachindex(centres)
        adjacency_offsets[index + 1] =
            adjacency_offsets[index] + adjacency_cursor[index]
        adjacency_cursor[index] = adjacency_offsets[index]
    end
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

    result = Vector{NTuple{2, ThreePointStencil{T}}}(undef, length(connections))
    for (index, connection) in pairs(connections)
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

@inline function _add_amr_flux_weight!(
        coefficients, first::Int, second::Int,
        flux::Int, coefficient
    )
    first == second && throw(ArgumentError("degenerate AMR dual connection"))
    if first < second
        key = (first, second, flux)
        coefficients[key] = get(coefficients, key, zero(coefficient)) + coefficient
    else
        key = (second, first, flux)
        coefficients[key] = get(coefficients, key, zero(coefficient)) - coefficient
    end
    return coefficients
end

function _component_incidences(mesh::BlockTreeMesh, ::Val{Axis}, ::Type{T}) where {Axis, T}
    nx, ny = mesh.block_size
    global_to_local, component_count = _component_face_indices(mesh, Axis)
    incidences = [ComponentFaceIncidence{T}[] for _ in 1:ncells(mesh)]
    volumes = Vector{T}(undef, component_count)
    tangent_axis = 3 - Axis
    for (global_index, face) in pairs(mesh.faces)
        face.axis == Axis || continue
        local_index = global_to_local[global_index]
        for (cell, side) in ((face.minus, 1), (face.plus, -1))
            cell_index = _cell_index(cell, nx, ny)
            block = mesh.blocks[cell.block]
            fraction = T(face.area / block.spacing[tangent_axis])
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
    nx, ny = mesh.block_size
    incidences, global_to_local, volumes =
        _component_incidences(mesh, Val(Axis), T)
    coefficients = Dict{NTuple{3, Int}, T}()

    for (flux_index, flux_face) in pairs(mesh.faces)
        if flux_face.axis == Axis
            # A normal primal flux is routed through the two cells adjacent to
            # it. Stars centred on the transported face reproduce the regular
            # two-point average and remain valid when either cell side splits.
            centre = global_to_local[flux_index]
            for (cell, sign) in (
                    (flux_face.minus, one(T)),
                    (flux_face.plus, -one(T)),
                )
                cell_index = _cell_index(cell, nx, ny)
                for component_face in incidences[cell_index]
                    component_face.global_index == flux_index && continue
                    coefficient = sign * component_face.fraction / 2
                    _add_amr_flux_weight!(
                        coefficients, component_face.local_index, centre,
                        flux_index, coefficient
                    )
                end
            end
        else
            # Across a transverse primal face, couple the same normal side of
            # both cells. The product weights are a conservative transport
            # plan between possibly split side partitions; their marginals are
            # exactly one half of the incidence fractions.
            minus_index = _cell_index(flux_face.minus, nx, ny)
            plus_index = _cell_index(flux_face.plus, nx, ny)
            for side in (-1, 1)
                for minus_face in incidences[minus_index]
                    minus_face.side == side || continue
                    for plus_face in incidences[plus_index]
                        plus_face.side == side || continue
                        coefficient = minus_face.fraction * plus_face.fraction / 2
                        _add_amr_flux_weight!(
                            coefficients, minus_face.local_index,
                            plus_face.local_index,
                            flux_index, coefficient
                        )
                    end
                end
            end
        end
    end

    keys_in_order = sort!(
        [
            key for key in keys(coefficients) if !iszero(coefficients[key])
        ]; by = key -> (key[1], key[2], mesh.faces[key[3]].axis, key[3])
    )
    dual_connections = RawAMRDualFace[]
    flux_indices = Int[]
    flux_weights = T[]
    cursor = 1
    while cursor <= length(keys_in_order)
        minus, plus, first_flux = keys_in_order[cursor]
        axis = mesh.faces[first_flux].axis
        first_weight = length(flux_indices) + 1
        while cursor <= length(keys_in_order) &&
                keys_in_order[cursor][1] == minus &&
                keys_in_order[cursor][2] == plus &&
                mesh.faces[keys_in_order[cursor][3]].axis == axis
            key = keys_in_order[cursor]
            push!(flux_indices, key[3])
            push!(flux_weights, coefficients[key])
            cursor += 1
        end
        weight_count = length(flux_indices) - first_weight + 1
        refinement_interface = weight_count != 2
        for index in first_weight:length(flux_indices)
            refinement_interface |= !isapprox(
                abs(flux_weights[index]), T(0.5); atol = 64eps(T), rtol = 64eps(T)
            )
        end
        refinement_interface |= !isapprox(
            volumes[minus], volumes[plus]; atol = 64eps(T), rtol = 64eps(T)
        )
        push!(
            dual_connections, RawAMRDualFace(
                minus, plus, first_weight, weight_count, axis,
                refinement_interface
            )
        )
    end

    stencils = if build_three_point_stencils
        centres = _component_face_centres(mesh, Axis, T)
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

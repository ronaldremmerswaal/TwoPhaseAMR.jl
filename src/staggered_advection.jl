abstract type AbstractStaggeredAdvectionScheme end

"""First-order upwind interpolation on staggered control-volume faces."""
struct Upwind <: AbstractStaggeredAdvectionScheme end

"""Equal-weight interpolation on staggered control-volume faces."""
struct Centred <: AbstractStaggeredAdvectionScheme end

"""Second-order directionally split Lax--Wendroff transport."""
struct LaxWendroff <: AbstractStaggeredAdvectionScheme end

abstract type AbstractThreePointFluxScheme <:
              AbstractStaggeredAdvectionScheme end

"""Unrestricted three-point Fromm flux."""
struct Fromm <: AbstractThreePointFluxScheme end

"""Three-point flux with the minmod limiter."""
struct Minmod <: AbstractThreePointFluxScheme end

"""Three-point flux with the monotonized-central (MC) limiter."""
struct MonotonizedCentral <: AbstractThreePointFluxScheme end

"""Three-point flux with the van Leer limiter."""
struct VanLeer <: AbstractThreePointFluxScheme end

"""Three-point flux with the superbee limiter."""
struct Superbee <: AbstractThreePointFluxScheme end

const Centered = Centred
const LW = LaxWendroff
const MC = MonotonizedCentral

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
        Axis,T,M<:BlockTreeMesh,S<:AbstractDualAdvectionSchedule{T}}
    tendency::FaceField{Axis,T,M}
    rates::FaceField{Axis,T,M}
    schedule::S
    sweep_forward::Base.RefValue{Bool}
end

"""Workspace for simultaneous transport of both velocity components."""
struct StaggeredVelocityAdvectionWorkspace{T,M<:BlockTreeMesh,WX,WY}
    x::FaceField{1,T,M}
    y::FaceField{2,T,M}
    xworkspace::WX
    yworkspace::WY
    sweep_forward::Base.RefValue{Bool}
end

@inline function _cell_index(ref::CellRef, nx::Int, ny::Int)
    return (ref.block - 1) * nx * ny + (ref.j - 1) * nx + ref.i
end

function _uniform_periodic_cell_faces(mesh::BlockTreeMesh)
    all(mesh.periodic) || throw(ArgumentError(
        "staggered advection currently requires periodic boundaries in both directions"))
    level = first(mesh.blocks).key.level
    all(block -> block.key.level == level, mesh.blocks) || throw(ArgumentError(
        "staggered advection currently requires a uniform tree level"))

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
    centres = NTuple{2,T}[]
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

function _three_point_stencils(mesh::BlockTreeMesh, centres,
                               connections)
    T = typeof(first(centres)[1])
    adjacency = [Int[] for _ in centres]
    for (connection_index, connection) in pairs(connections)
        push!(adjacency[connection.minus], connection_index)
        push!(adjacency[connection.plus], connection_index)
    end

    function outer_stencil(connection_index, endpoint::Int, neighbour::Int,
                           direction)
        connection = connections[connection_index]
        axis = connection.axis
        tangent_axis = 3 - axis
        displacement = _dual_displacement(
            mesh, centres[endpoint], centres[neighbour])
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
        for candidate_connection_index in adjacency[endpoint]
            candidate_connection_index == connection_index && continue
            candidate_connection = connections[candidate_connection_index]
            candidate_connection.axis == axis || continue
            candidate = candidate_connection.minus == endpoint ?
                        candidate_connection.plus : candidate_connection.minus
            candidate_displacement = _dual_displacement(
                mesh, centres[endpoint], centres[candidate])
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
            best_neighbour, T(axial_distance / best_axial_distance))
    end

    result = Vector{NTuple{2,ThreePointStencil{T}}}(undef, length(connections))
    for (index, connection) in pairs(connections)
        displacement = _dual_displacement(
            mesh, centres[connection.minus], centres[connection.plus])
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

function _uniform_dual_schedule(mesh::BlockTreeMesh, ::Val{Axis}, ::Type{T}) where {Axis,T}
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
                xminus[cell], xplus[cell], 1, T(dy), T(dx))
            # Transverse dual face through the cell's upper-right vertex.
            raw_faces[n + cell] = RawDualFace(
                global_to_local[xplus[cell]], global_to_local[xplus[above]],
                yplus[cell], yplus[right], 2, T(dx), T(dy))
        else
            # Normal dual face through the cell centre.
            raw_faces[cell] = RawDualFace(
                global_to_local[yminus[cell]], global_to_local[yplus[cell]],
                yminus[cell], yplus[cell], 2, T(dx), T(dy))
            # Transverse dual face through the cell's upper-right vertex.
            raw_faces[n + cell] = RawDualFace(
                global_to_local[yplus[cell]], global_to_local[yplus[right]],
                xplus[cell], xplus[above], 1, T(dy), T(dx))
        end
    end

    centres = _component_face_centres(mesh, Axis, T)
    connections = [(minus=face.minus, plus=face.plus, axis=face.axis)
                   for face in raw_faces]
    stencils = _three_point_stencils(mesh, centres, connections)
    dual_faces = Vector{DualFace{T}}(undef, length(raw_faces))
    for (index, face) in pairs(raw_faces)
        minus_stencil, plus_stencil = stencils[index]
        dual_faces[index] = DualFace(
            face.minus, face.plus, face.flux_a, face.flux_b,
            face.axis, face.area, face.distance,
            minus_stencil, plus_stencil)
    end
    any(face -> iszero(face.minus) || iszero(face.plus), dual_faces) &&
        throw(ArgumentError("invalid directional face mapping"))
    return UniformDualAdvectionSchedule(dual_faces, T(dx * dy))
end

@inline function _add_amr_flux_weight!(coefficients, first::Int, second::Int,
                                       flux::Int, coefficient)
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

function _component_incidences(mesh::BlockTreeMesh, ::Val{Axis}, ::Type{T}) where {Axis,T}
    nx, ny = mesh.block_size
    global_to_local, component_count = _component_face_indices(mesh, Axis)
    incidences = [ComponentFaceIncidence{T}[] for _ in 1:ncells(mesh)]
    volumes = zeros(T, component_count)
    tangent_axis = 3 - Axis
    for (global_index, face) in pairs(mesh.faces)
        face.axis == Axis || continue
        local_index = global_to_local[global_index]
        for (cell, side) in ((face.minus, 1), (face.plus, -1))
            cell_index = _cell_index(cell, nx, ny)
            block = mesh.blocks[cell.block]
            fraction = T(face.area / block.spacing[tangent_axis])
            push!(incidences[cell_index], ComponentFaceIncidence(
                local_index, global_index, side, fraction))
            volumes[local_index] += T(0.5) * fraction * T(block.volume)
        end
    end

    tolerance = 512eps(T)
    for cell_faces in incidences, side in (-1, 1)
        fraction_sum = sum(face.fraction for face in cell_faces if face.side == side;
                           init=zero(T))
        # A zero sum is the physical wall normal to the transported component;
        # its prescribed face value is not an evolved dual-volume unknown.
        (isapprox(fraction_sum, one(T); atol=tolerance, rtol=tolerance) ||
         isapprox(fraction_sum, zero(T); atol=tolerance, rtol=tolerance)) ||
            throw(ArgumentError("nonconforming AMR component-face partition"))
    end
    all(>(zero(T)), volumes) ||
        throw(ArgumentError("AMR staggered control volume has non-positive size"))
    return incidences, global_to_local, volumes
end

function _amr_dual_schedule(mesh::BlockTreeMesh, ::Val{Axis}, ::Type{T}) where {Axis,T}
    nx, ny = mesh.block_size
    incidences, global_to_local, volumes =
        _component_incidences(mesh, Val(Axis), T)
    coefficients = Dict{NTuple{3,Int},T}()

    for (flux_index, flux_face) in pairs(mesh.faces)
        if flux_face.axis == Axis
            # A normal primal flux is routed through the two cells adjacent to
            # it. Stars centred on the transported face reproduce the regular
            # two-point average and remain valid when either cell side splits.
            centre = global_to_local[flux_index]
            for (cell, sign) in ((flux_face.minus, one(T)),
                                 (flux_face.plus, -one(T)))
                cell_index = _cell_index(cell, nx, ny)
                for component_face in incidences[cell_index]
                    component_face.global_index == flux_index && continue
                    coefficient = sign * component_face.fraction / 2
                    _add_amr_flux_weight!(
                        coefficients, component_face.local_index, centre,
                        flux_index, coefficient)
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
                            flux_index, coefficient)
                    end
                end
            end
        end
    end

    keys_in_order = sort!([
        key for key in keys(coefficients) if !iszero(coefficients[key])
    ]; by=key -> (key[1], key[2], mesh.faces[key[3]].axis, key[3]))
    dual_connections = NamedTuple[]
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
                abs(flux_weights[index]), T(0.5); atol=64eps(T), rtol=64eps(T))
        end
        refinement_interface |= !isapprox(
            volumes[minus], volumes[plus]; atol=64eps(T), rtol=64eps(T))
        push!(dual_connections, (;
            minus, plus, first_weight, weight_count, axis,
            refinement_interface))
    end

    centres = _component_face_centres(mesh, Axis, T)
    stencils = _three_point_stencils(mesh, centres, dual_connections)
    dual_faces = Vector{AMRDualFace{T}}(undef, length(dual_connections))
    for (index, connection) in pairs(dual_connections)
        minus_stencil, plus_stencil = stencils[index]
        dual_faces[index] = AMRDualFace(
            connection.minus, connection.plus, connection.first_weight,
            connection.weight_count, connection.axis,
            connection.refinement_interface, minus_stencil, plus_stencil)
    end
    return AMRDualAdvectionSchedule(
        dual_faces, flux_indices, flux_weights, volumes)
end

function StaggeredAdvectionWorkspace(field::FaceField{Axis,T,M}) where {Axis,T,M}
    mesh = field.mesh
    _, component_count = _component_face_indices(mesh, Axis)
    component_count == length(field.interior) ||
        throw(ArgumentError("face field does not match the mesh schedule"))
    first_level = first(mesh.blocks).key.level
    uniform = all(block -> block.key.level == first_level, mesh.blocks)
    schedule = uniform && all(mesh.periodic) ?
        _uniform_dual_schedule(mesh, Val(Axis), T) :
        _amr_dual_schedule(mesh, Val(Axis), T)
    tendency = FaceField{Axis}(mesh, T)
    rates = FaceField{Axis}(mesh, T)
    return StaggeredAdvectionWorkspace(tendency, rates, schedule, Ref(true))
end

function StaggeredVelocityAdvectionWorkspace(velocity::FaceVelocity{T,M}) where {T,M}
    x = FaceField{1}(velocity)
    y = FaceField{2}(velocity)
    return StaggeredVelocityAdvectionWorkspace(
        x, y, StaggeredAdvectionWorkspace(x), StaggeredAdvectionWorkspace(y),
        Ref(true))
end

StaggeredAdvectionWorkspace(velocity::FaceVelocity) =
    StaggeredVelocityAdvectionWorkspace(velocity)

@inline _dual_state(qminus, qplus, flux, divergence, dt, face, ::Upwind,
                    values) =
    ifelse(flux >= zero(flux), qminus, qplus)

@inline _dual_state(qminus, qplus, flux, divergence, dt, face, ::Centred,
                    values) =
    (qminus + qplus) / 2

@inline function _dual_state(qminus, qplus, flux, divergence, dt,
                             face, ::LaxWendroff, values)
    normal_speed = flux / face.area
    normal_courant = abs(dt * normal_speed / face.distance)
    upwind, downwind = flux >= zero(flux) ? (qminus, qplus) : (qplus, qminus)
    return ((one(normal_courant) - normal_courant) * downwind +
            (one(normal_courant) + normal_courant) * upwind) / 2 -
           dt * (qminus * divergence[face.minus] +
                 qplus * divergence[face.plus]) / 4
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

@inline function _three_point_state(upwind, downwind, upstream,
                                    ratio_scale, high_order,
                                    scheme::AbstractThreePointFluxScheme)
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

@inline function _dual_state(qminus, qplus, flux, divergence, dt, face,
                             scheme::AbstractThreePointFluxScheme, values)
    positive_flux = flux >= zero(flux)
    upwind, downwind = positive_flux ? (qminus, qplus) : (qplus, qminus)
    stencil = positive_flux ? face.minus_stencil : face.plus_stencil
    iszero(stencil.upstream) && return upwind
    high_order = _dual_state(
        qminus, qplus, flux, divergence, dt, face, LaxWendroff(), values)
    upstream = @inbounds values[stencil.upstream]
    return _three_point_state(upwind, downwind, upstream,
                              stencil.ratio_scale, high_order, scheme)
end

function _check_staggered_arguments(destination::FaceField{Axis},
                                    advected::FaceField{Axis},
                                    velocity::FaceVelocity,
                                    workspace::StaggeredAdvectionWorkspace{Axis}) where {Axis}
    mesh = advected.mesh
    destination.mesh === mesh === velocity.mesh === workspace.tendency.mesh ||
        throw(ArgumentError("all staggered-advection arguments must share a mesh"))
    destination === advected && throw(ArgumentError(
        "convection! destination must not alias its advected field"))
    return nothing
end

@inline function _amr_dual_flux(face::AMRDualFace,
                                schedule::AMRDualAdvectionSchedule,
                                velocity::FaceVelocity)
    flux = zero(promote_type(eltype(schedule.flux_weights), eltype(velocity)))
    stop = face.first_weight + face.weight_count - 1
    for index in face.first_weight:stop
        @inbounds begin
            primal_face = schedule.flux_indices[index]
            flux += schedule.flux_weights[index] *
                    velocity.interior[primal_face] *
                    velocity.mesh.faces[primal_face].area
        end
    end
    return flux
end

@inline _amr_dual_state(qminus, qplus, flux, divergence, dt, face,
                        minus_volume, plus_volume,
                        ::Upwind, values) =
    ifelse(flux >= zero(flux), qminus, qplus)

@inline _amr_dual_state(qminus, qplus, flux, divergence, dt, face,
                        minus_volume, plus_volume,
                        ::Centred, values) = (qminus + qplus) / 2

@inline function _amr_dual_state(qminus, qplus, flux, divergence, dt, face,
                                 minus_volume, plus_volume, ::LaxWendroff,
                                 values)
    # On a nonconforming schedule there is no unique transverse donor. Use the
    # one-dimensional LW state on each algebraic connection. The smaller
    # adjacent dual volume gives a conservative local length scale and is
    # covered by `stable_timestep`. `Upwind` remains the robust AMR option.
    courant = abs(dt * flux) / min(minus_volume, plus_volume)
    upwind, downwind = flux >= zero(flux) ? (qminus, qplus) : (qplus, qminus)
    return ((one(courant) - courant) * downwind +
            (one(courant) + courant) * upwind) / 2 -
           dt * (qminus * divergence[face.minus] +
                 qplus * divergence[face.plus]) / 4
end

@inline function _amr_dual_state(qminus, qplus, flux, divergence, dt, face,
                                 minus_volume, plus_volume,
                                 scheme::AbstractThreePointFluxScheme, values)
    positive_flux = flux >= zero(flux)
    upwind, downwind = positive_flux ? (qminus, qplus) : (qplus, qminus)
    stencil = positive_flux ? face.minus_stencil : face.plus_stencil
    iszero(stencil.upstream) && return upwind
    high_order = _amr_dual_state(
        qminus, qplus, flux, divergence, dt, face,
        minus_volume, plus_volume, LaxWendroff(), values)
    upstream = @inbounds values[stencil.upstream]
    return _three_point_state(upwind, downwind, upstream,
                              stencil.ratio_scale, high_order, scheme)
end

function _directional_flux_divergence!(divergence, velocity,
                                       schedule::UniformDualAdvectionSchedule,
                                       sweep_axis::Int)
    fill!(divergence, zero(eltype(divergence)))
    inverse_volume = inv(schedule.volume)
    for face in schedule.faces
        (iszero(sweep_axis) || face.axis == sweep_axis) || continue
        @inbounds begin
            flux = (velocity.interior[face.flux_a] +
                    velocity.interior[face.flux_b]) * face.area / 2
            rate = flux * inverse_volume
            divergence[face.minus] += rate
            divergence[face.plus] -= rate
        end
    end
    return divergence
end

function _directional_flux_divergence!(divergence, velocity,
                                       schedule::AMRDualAdvectionSchedule,
                                       sweep_axis::Int)
    fill!(divergence, zero(eltype(divergence)))
    for face in schedule.faces
        (iszero(sweep_axis) || face.axis == sweep_axis) || continue
        @inbounds begin
            flux = _amr_dual_flux(face, schedule, velocity)
            divergence[face.minus] += flux / schedule.volumes[face.minus]
            divergence[face.plus] -= flux / schedule.volumes[face.plus]
        end
    end
    return divergence
end

function _convection_schedule!(destination, advected, velocity, dt,
                               schedule::UniformDualAdvectionSchedule,
                               directional_divergence, scheme,
                               amr_interface_scheme,
                               sweep_axis::Int=0)
    inverse_volume = inv(schedule.volume)
    for face in schedule.faces
        (iszero(sweep_axis) || face.axis == sweep_axis) || continue
        @inbounds begin
            flux = (velocity.interior[face.flux_a] +
                    velocity.interior[face.flux_b]) * face.area / 2
            qminus = advected.interior[face.minus]
            qplus = advected.interior[face.plus]
            state = _dual_state(qminus, qplus, flux,
                                directional_divergence, dt, face, scheme,
                                advected.interior)
            rate = flux * state * inverse_volume
            destination.interior[face.minus] += rate
            destination.interior[face.plus] -= rate
        end
    end
    return destination
end

function _convection_schedule!(destination, advected, velocity, dt,
                               schedule::AMRDualAdvectionSchedule,
                               directional_divergence, scheme,
                               amr_interface_scheme,
                               sweep_axis::Int=0)
    for face in schedule.faces
        (iszero(sweep_axis) || face.axis == sweep_axis) || continue
        @inbounds begin
            flux = _amr_dual_flux(face, schedule, velocity)
            qminus = advected.interior[face.minus]
            qplus = advected.interior[face.plus]
            minus_volume = schedule.volumes[face.minus]
            plus_volume = schedule.volumes[face.plus]
            state = if face.refinement_interface
                _amr_dual_state(qminus, qplus, flux,
                                directional_divergence, dt, face,
                                minus_volume, plus_volume,
                                amr_interface_scheme, advected.interior)
            else
                _amr_dual_state(qminus, qplus, flux,
                                directional_divergence, dt, face,
                                minus_volume, plus_volume, scheme,
                                advected.interior)
            end
            transported = flux * state
            destination.interior[face.minus] += transported / minus_volume
            destination.interior[face.plus] -= transported / plus_volume
        end
    end
    return destination
end

function _convection!(destination::FaceField{Axis,T},
                      advected::FaceField{Axis}, velocity::FaceVelocity, dt,
                      workspace::StaggeredAdvectionWorkspace{Axis},
                      scheme, amr_interface_scheme,
                      sweep_axis::Int=0) where {Axis,T}
    _check_staggered_arguments(destination, advected, velocity, workspace)
    fill!(destination.interior, zero(T))
    fill!(destination.boundary, zero(T))
    if !iszero(dt) && _directionally_split_scheme(
            scheme, amr_interface_scheme, workspace.schedule)
        _directional_flux_divergence!(
            workspace.rates.interior, velocity, workspace.schedule, sweep_axis)
    else
        fill!(workspace.rates.interior, zero(T))
    end
    return _convection_schedule!(
        destination, advected, velocity, dt, workspace.schedule,
        workspace.rates.interior, scheme, amr_interface_scheme, sweep_axis)
end

function _dispatch_interface_convection!(destination, advected, velocity, dt,
                                         workspace, scheme,
                                         amr_interface_scheme,
                                         sweep_axis::Int=0)
    if amr_interface_scheme isa Upwind || amr_interface_scheme === :upwind
        return _convection!(destination, advected, velocity, dt, workspace,
                            scheme, Upwind(), sweep_axis)
    elseif amr_interface_scheme isa Centred ||
           amr_interface_scheme === :centred ||
           amr_interface_scheme === :centered ||
           amr_interface_scheme === :central
        return _convection!(destination, advected, velocity, dt, workspace,
                            scheme, Centred(), sweep_axis)
    elseif amr_interface_scheme isa LaxWendroff ||
           amr_interface_scheme === :lw ||
           amr_interface_scheme === :lax_wendroff
        return _convection!(destination, advected, velocity, dt, workspace,
                            scheme, LaxWendroff(), sweep_axis)
    elseif amr_interface_scheme isa Fromm ||
           amr_interface_scheme === :fromm
        return _convection!(destination, advected, velocity, dt, workspace,
                            scheme, Fromm(), sweep_axis)
    elseif amr_interface_scheme isa Minmod ||
           amr_interface_scheme === :minmod
        return _convection!(destination, advected, velocity, dt, workspace,
                            scheme, Minmod(), sweep_axis)
    elseif amr_interface_scheme isa MonotonizedCentral ||
           amr_interface_scheme === :mc ||
           amr_interface_scheme === :monotonized_central ||
           amr_interface_scheme === :monotonised_central
        return _convection!(destination, advected, velocity, dt, workspace,
                            scheme, MonotonizedCentral(), sweep_axis)
    elseif amr_interface_scheme isa VanLeer ||
           amr_interface_scheme === :van_leer ||
           amr_interface_scheme === :vanleer
        return _convection!(destination, advected, velocity, dt, workspace,
                            scheme, VanLeer(), sweep_axis)
    elseif amr_interface_scheme isa Superbee ||
           amr_interface_scheme === :superbee
        return _convection!(destination, advected, velocity, dt, workspace,
                            scheme, Superbee(), sweep_axis)
    end
    throw(ArgumentError(
        "unsupported AMR-interface advection scheme: $amr_interface_scheme"))
end

function _dispatch_convection!(destination, advected, velocity, dt, workspace,
                               scheme, amr_interface_scheme,
                               sweep_axis::Int=0)
    if scheme isa Upwind || scheme === :upwind
        return _dispatch_interface_convection!(
            destination, advected, velocity, dt, workspace,
            Upwind(), amr_interface_scheme, sweep_axis)
    elseif scheme isa Centred || scheme === :centred ||
           scheme === :centered || scheme === :central
        return _dispatch_interface_convection!(
            destination, advected, velocity, dt, workspace,
            Centred(), amr_interface_scheme, sweep_axis)
    elseif scheme isa LaxWendroff || scheme === :lw || scheme === :lax_wendroff
        return _dispatch_interface_convection!(
            destination, advected, velocity, dt, workspace,
            LaxWendroff(), amr_interface_scheme, sweep_axis)
    elseif scheme isa Fromm || scheme === :fromm
        return _dispatch_interface_convection!(
            destination, advected, velocity, dt, workspace,
            Fromm(), amr_interface_scheme, sweep_axis)
    elseif scheme isa Minmod || scheme === :minmod
        return _dispatch_interface_convection!(
            destination, advected, velocity, dt, workspace,
            Minmod(), amr_interface_scheme, sweep_axis)
    elseif scheme isa MonotonizedCentral || scheme === :mc ||
           scheme === :monotonized_central ||
           scheme === :monotonised_central
        return _dispatch_interface_convection!(
            destination, advected, velocity, dt, workspace,
            MonotonizedCentral(), amr_interface_scheme, sweep_axis)
    elseif scheme isa VanLeer || scheme === :van_leer || scheme === :vanleer
        return _dispatch_interface_convection!(
            destination, advected, velocity, dt, workspace,
            VanLeer(), amr_interface_scheme, sweep_axis)
    elseif scheme isa Superbee || scheme === :superbee
        return _dispatch_interface_convection!(
            destination, advected, velocity, dt, workspace,
            Superbee(), amr_interface_scheme, sweep_axis)
    end
    throw(ArgumentError("unsupported staggered advection scheme: $scheme"))
end

@inline _directionally_split_scheme(::LaxWendroff) = true
@inline _directionally_split_scheme(::AbstractThreePointFluxScheme) = true
@inline _directionally_split_scheme(scheme::Symbol) =
    scheme === :lw || scheme === :lax_wendroff || scheme === :fromm ||
    scheme === :minmod || scheme === :mc ||
    scheme === :monotonized_central || scheme === :monotonised_central ||
    scheme === :van_leer || scheme === :vanleer || scheme === :superbee
@inline _directionally_split_scheme(scheme) = false
@inline _directionally_split_scheme(scheme, amr_interface_scheme) =
    _directionally_split_scheme(scheme) ||
    _directionally_split_scheme(amr_interface_scheme)
@inline _directionally_split_scheme(scheme, amr_interface_scheme,
                                    ::UniformDualAdvectionSchedule) =
    _directionally_split_scheme(scheme)
@inline _directionally_split_scheme(scheme, amr_interface_scheme,
                                    ::AMRDualAdvectionSchedule) =
    _directionally_split_scheme(scheme, amr_interface_scheme)

function _split_convection_rate!(destination::FaceField{Axis,T},
                                 advected::FaceField{Axis},
                                 velocity::FaceVelocity, dt,
                                 workspace::StaggeredAdvectionWorkspace{Axis},
                                 scheme, amr_interface_scheme) where {Axis,T}
    destination === advected && throw(ArgumentError(
        "convection! destination must not alias its advected field"))
    (destination === workspace.tendency || destination === workspace.rates) &&
        throw(ArgumentError(
            "split convection destination must be separate from its workspace"))
    copyto!(destination.interior, advected.interior)
    copyto!(destination.boundary, advected.boundary)
    scratch = destination === workspace.tendency ?
              workspace.rates : workspace.tendency
    sweep_axes = workspace.sweep_forward[] ? (1, 2) : (2, 1)
    for sweep_axis in sweep_axes
        sweep_dt = dt
        _dispatch_convection!(scratch, destination, velocity, sweep_dt, workspace,
                              scheme, amr_interface_scheme, sweep_axis)
        for index in eachindex(destination.interior)
            @inbounds destination.interior[index] -=
                sweep_dt * scratch.interior[index]
        end
    end
    for index in eachindex(destination.interior)
        @inbounds destination.interior[index] =
            (advected.interior[index] - destination.interior[index]) / dt
    end
    fill!(destination.boundary, zero(T))
    return destination
end

"""
    convection!(destination, advected, velocity, dt, workspace;
                scheme=LW(), amr_interface_scheme=scheme)

Evaluate the conservative staggered convection operator `div(I(velocity) q)`
on owned dual control volumes. `scheme` may be `Upwind()`, `Centred()`,
`LaxWendroff()`, `Fromm()`, `Minmod()`, `MC()`, `VanLeer()`, or `Superbee()`
(the corresponding symbols are also accepted) and defaults to `LW()`.

For LW and positive `dt`, the returned rate is the effective rate of the next
directionally split step. Each step has two full sweeps; their order alternates
between x/y and y/x on successive `advect!` calls. Every sweep consumes the
preceding intermediate field. Thus `destination == (advected -
advect(advected, dt)) / dt` without advancing the workspace's order flag; x
and y fluxes are not accumulated from one old state. At `dt == 0`, the
limiting spatial operator is evaluated directly.

For an all-centred operator (`scheme=Centred()` and
`amr_interface_scheme=Centred()`), the face value is the equal-weight average.
On every supported periodic mesh this makes twice the convection-matrix
diagonal equal the volume-weighted interpolation of the primary cell
divergence. The operator is therefore volume-weighted skew-symmetric for a
discretely divergence-free velocity, including across refinement interfaces.

On a mixed-level mesh, `amr_interface_scheme` selects the state used only on
non-regular factored connections. It defaults to the value of `scheme`, so the
default call applies `LW()` throughout the mesh. Pass `Upwind()` explicitly
for a first-order interface closure, or one of the three-point schemes for an
upstream reconstruction. The four limited variants revert to upwind at a local
extremum. A hanging connection without an unambiguous upstream direction also
falls back to upwind. `Centred()` retains the diagonal and skew identities
throughout the mesh. The keyword is accepted but has no effect on a uniform
mesh.
"""
function convection!(destination::FaceField{Axis}, advected::FaceField{Axis},
                     velocity::FaceVelocity, dt,
                     workspace::StaggeredAdvectionWorkspace{Axis};
                     scheme=LW(),
                     amr_interface_scheme=scheme) where {Axis}
    dt >= zero(dt) || throw(ArgumentError("dt must be nonnegative"))
    if !iszero(dt) && _directionally_split_scheme(
            scheme, amr_interface_scheme, workspace.schedule)
        return _split_convection_rate!(
            destination, advected, velocity, dt, workspace,
            scheme, amr_interface_scheme)
    end
    return _dispatch_convection!(destination, advected, velocity, dt,
                                 workspace, scheme, amr_interface_scheme)
end

"""
    advect!(field, velocity, dt, workspace;
            scheme=LW(), amr_interface_scheme=scheme)

Advance a directional face field using conservative fluxes through its
staggered control volumes. Lax--Wendroff uses two full directional sweeps,
publishing the first intermediate before evaluating the second and alternating
x/y with y/x order on successive calls. The variable-coefficient LW predictor
includes the directional dual divergence of the transporting velocity; the
two directional divergence terms cancel for a divergence-free velocity only
after both sweeps. On a uniform constant-velocity line it reduces to classical
one-dimensional LW. By default `amr_interface_scheme` inherits `scheme`,
giving global `LW()`; pass `amr_interface_scheme=Upwind()` for the hybrid
closure or, for example, `amr_interface_scheme=MC()` for a three-point
interface-limited closure.
"""
function advect!(field::FaceField{Axis,T}, velocity::FaceVelocity, dt,
                 workspace::StaggeredAdvectionWorkspace{Axis,T};
                 scheme=LW(),
                 amr_interface_scheme=scheme) where {Axis,T}
    field === workspace.tendency && throw(ArgumentError(
        "the transported field must not be the workspace tendency"))
    dt >= zero(dt) || throw(ArgumentError("dt must be nonnegative"))
    split = _directionally_split_scheme(
        scheme, amr_interface_scheme, workspace.schedule)
    stages = split ?
             (workspace.sweep_forward[] ? ((1, dt), (2, dt)) :
                                            ((2, dt), (1, dt))) :
             ((0, dt),)
    for (sweep_axis, sweep_dt) in stages
        _dispatch_convection!(workspace.tendency, field, velocity, sweep_dt,
                              workspace,
                              scheme, amr_interface_scheme, sweep_axis)
        for index in eachindex(field.interior)
            @inbounds field.interior[index] -=
                sweep_dt * workspace.tendency.interior[index]
        end
    end
    split && !iszero(dt) && (workspace.sweep_forward[] = !workspace.sweep_forward[])
    return field
end

function _velocity_convection!(destination::FaceVelocity,
                               advected::FaceVelocity,
                               advecting::FaceVelocity, dt,
                               workspace::StaggeredVelocityAdvectionWorkspace,
                               scheme, amr_interface_scheme)
    destination.mesh === advected.mesh === advecting.mesh === workspace.x.mesh ||
        throw(ArgumentError("all staggered-advection arguments must share a mesh"))
    destination === advected && throw(ArgumentError(
        "convection! destination must not alias its advected field"))
    copyto!(workspace.x, advected)
    copyto!(workspace.y, advected)
    _dispatch_convection!(workspace.xworkspace.tendency, workspace.x,
                          advecting, dt, workspace.xworkspace,
                          scheme, amr_interface_scheme)
    _dispatch_convection!(workspace.yworkspace.tendency, workspace.y,
                          advecting, dt, workspace.yworkspace,
                          scheme, amr_interface_scheme)
    fill!(destination.interior, zero(eltype(destination.interior)))
    fill!(destination.boundary, zero(eltype(destination.boundary)))
    copyto!(destination, workspace.xworkspace.tendency)
    copyto!(destination, workspace.yworkspace.tendency)
    return destination
end

function _split_velocity_convection!(destination::FaceVelocity,
                                     advected::FaceVelocity,
                                     advecting::FaceVelocity, dt,
                                     workspace::StaggeredVelocityAdvectionWorkspace,
                                     scheme, amr_interface_scheme)
    destination.mesh === advected.mesh === advecting.mesh === workspace.x.mesh ||
        throw(ArgumentError("all staggered-advection arguments must share a mesh"))
    destination === advected && throw(ArgumentError(
        "convection! destination must not alias its advected field"))
    destination === advecting && throw(ArgumentError(
        "convection! destination must not alias its advecting field"))
    copyto!(destination.interior, advected.interior)
    copyto!(destination.boundary, advected.boundary)
    copyto!(workspace.x, advected)
    copyto!(workspace.y, advected)
    self_advected = advected === advecting
    sweep_axes = workspace.sweep_forward[] ? (1, 2) : (2, 1)
    for sweep_axis in sweep_axes
        sweep_dt = dt
        current_advecting = self_advected ? destination : advecting
        _dispatch_convection!(workspace.xworkspace.tendency, workspace.x,
                              current_advecting, sweep_dt, workspace.xworkspace,
                              scheme, amr_interface_scheme, sweep_axis)
        _dispatch_convection!(workspace.yworkspace.tendency, workspace.y,
                              current_advecting, sweep_dt, workspace.yworkspace,
                              scheme, amr_interface_scheme, sweep_axis)
        for index in eachindex(workspace.x.interior)
            @inbounds workspace.x.interior[index] -=
                sweep_dt * workspace.xworkspace.tendency.interior[index]
        end
        for index in eachindex(workspace.y.interior)
            @inbounds workspace.y.interior[index] -=
                sweep_dt * workspace.yworkspace.tendency.interior[index]
        end
        copyto!(destination, workspace.x)
        copyto!(destination, workspace.y)
    end
    for index in eachindex(destination.interior)
        @inbounds destination.interior[index] =
            (advected.interior[index] - destination.interior[index]) / dt
    end
    fill!(destination.boundary, zero(eltype(destination.boundary)))
    return destination
end

"""Evaluate convection for both components of an owned staggered field."""
function convection!(destination::FaceVelocity, advected::FaceVelocity,
                     advecting::FaceVelocity, dt,
                     workspace::StaggeredVelocityAdvectionWorkspace;
                     scheme=LW(),
                     amr_interface_scheme=scheme)
    dt >= zero(dt) || throw(ArgumentError("dt must be nonnegative"))
    if !iszero(dt) && _directionally_split_scheme(
            scheme, amr_interface_scheme, workspace.xworkspace.schedule)
        return _split_velocity_convection!(
            destination, advected, advecting, dt, workspace,
            scheme, amr_interface_scheme)
    end
    return _velocity_convection!(
        destination, advected, advecting, dt, workspace,
        scheme, amr_interface_scheme)
end

"""
    advect!(field, advecting, dt, workspace;
            scheme=LW(), amr_interface_scheme=scheme)
    advect!(velocity, dt, workspace;
            scheme=LW(), amr_interface_scheme=scheme)

Advance both components of a staggered vector field. Within one coordinate
sweep, both component tendencies are evaluated from the same input and then
both components are published. The next coordinate sweep therefore consumes
one consistent intermediate vector rather than updating x and y velocity
components separately. The one-velocity shorthand also uses each published
intermediate as its advecting field. By default `amr_interface_scheme`
inherits `scheme`, so the no-keyword call uses `LW()` globally.
"""
function advect!(field::FaceVelocity, advecting::FaceVelocity, dt,
                 workspace::StaggeredVelocityAdvectionWorkspace;
                 scheme=LW(),
                 amr_interface_scheme=scheme)
    field.mesh === advecting.mesh === workspace.x.mesh ||
        throw(ArgumentError("all staggered-advection arguments must share a mesh"))
    dt >= zero(dt) || throw(ArgumentError("dt must be nonnegative"))
    copyto!(workspace.x, field)
    copyto!(workspace.y, field)
    self_advected = field === advecting
    split = _directionally_split_scheme(
        scheme, amr_interface_scheme, workspace.xworkspace.schedule)
    stages = split ?
             (workspace.sweep_forward[] ? ((1, dt), (2, dt)) :
                                            ((2, dt), (1, dt))) :
             ((0, dt),)
    for (sweep_axis, sweep_dt) in stages
        current_advecting = self_advected ? field : advecting
        _dispatch_convection!(workspace.xworkspace.tendency, workspace.x,
                              current_advecting, sweep_dt, workspace.xworkspace,
                              scheme, amr_interface_scheme, sweep_axis)
        _dispatch_convection!(workspace.yworkspace.tendency, workspace.y,
                              current_advecting, sweep_dt, workspace.yworkspace,
                              scheme, amr_interface_scheme, sweep_axis)
        for index in eachindex(workspace.x.interior)
            @inbounds workspace.x.interior[index] -=
                sweep_dt * workspace.xworkspace.tendency.interior[index]
        end
        for index in eachindex(workspace.y.interior)
            @inbounds workspace.y.interior[index] -=
                sweep_dt * workspace.yworkspace.tendency.interior[index]
        end
        if self_advected
            # Both components see the same intermediate state. Publishing it
            # only after both tendencies have been evaluated makes this a
            # coordinate sweep, not a component-wise split.
            copyto!(field, workspace.x)
            copyto!(field, workspace.y)
        end
    end
    copyto!(field, workspace.x)
    copyto!(field, workspace.y)
    split && !iszero(dt) && (workspace.sweep_forward[] = !workspace.sweep_forward[])
    return field
end

advect!(velocity::FaceVelocity, dt,
        workspace::StaggeredVelocityAdvectionWorkspace; kwargs...) =
    advect!(velocity, velocity, dt, workspace; kwargs...)

@inline _dual_volumes(schedule::UniformDualAdvectionSchedule, index) = schedule.volume
@inline _dual_volumes(schedule::AMRDualAdvectionSchedule, index) =
    @inbounds schedule.volumes[index]

"""Return the dual-volume-weighted integral of a periodic face field."""
function integrate(field::FaceField{Axis},
                   workspace::StaggeredAdvectionWorkspace{Axis}) where {Axis}
    field.mesh === workspace.tendency.mesh ||
        throw(ArgumentError("field and workspace must share a mesh"))
    result = zero(eltype(field))
    for index in eachindex(field.interior)
        @inbounds result += field.interior[index] *
                            _dual_volumes(workspace.schedule, index)
    end
    return result
end

"""Return the volume-weighted inner product of two staggered components."""
function staggered_inner_product(a::FaceField{Axis}, b::FaceField{Axis},
                                 workspace::StaggeredAdvectionWorkspace{Axis}) where {Axis}
    a.mesh === b.mesh === workspace.tendency.mesh ||
        throw(ArgumentError("fields and workspace must share a mesh"))
    result = zero(promote_type(eltype(a), eltype(b)))
    for index in eachindex(a.interior, b.interior)
        @inbounds result += a.interior[index] * b.interior[index] *
                            _dual_volumes(workspace.schedule, index)
    end
    return result
end

function _accumulate_staggered_rates!(rates, velocity,
                                      schedule::UniformDualAdvectionSchedule)
    inverse_volume = inv(schedule.volume)
    for face in schedule.faces
        @inbounds flux = (velocity.interior[face.flux_a] +
                          velocity.interior[face.flux_b]) * face.area / 2
        rate = abs(flux) * inverse_volume
        @inbounds begin
            rates[face.minus] += rate
            rates[face.plus] += rate
        end
    end
    return rates
end

function _accumulate_staggered_rates!(rates, velocity,
                                      schedule::AMRDualAdvectionSchedule)
    for face in schedule.faces
        @inbounds begin
            flux = abs(_amr_dual_flux(face, schedule, velocity))
            rates[face.minus] += flux / schedule.volumes[face.minus]
            rates[face.plus] += flux / schedule.volumes[face.plus]
        end
    end
    return rates
end

"""Return a conservative CFL bound for one staggered component."""
function stable_timestep(velocity::FaceVelocity{TV},
                         workspace::StaggeredAdvectionWorkspace{Axis,TW};
                         cfl=0.9) where {TV,Axis,TW}
    velocity.mesh === workspace.rates.mesh ||
        throw(ArgumentError("velocity and workspace must share a mesh"))
    T = promote_type(TV, TW, typeof(cfl))
    zero(T) < cfl <= one(T) || throw(ArgumentError("cfl must lie in (0, 1]"))
    fill!(workspace.rates.interior, zero(TW))
    _accumulate_staggered_rates!(
        workspace.rates.interior, velocity, workspace.schedule)
    maxrate = zero(T)
    for rate in workspace.rates.interior
        maxrate = max(maxrate, rate)
    end
    return iszero(maxrate) ? T(Inf) : T(cfl) / maxrate
end


function stable_timestep(velocity::FaceVelocity,
                         workspace::StaggeredVelocityAdvectionWorkspace;
                         cfl=0.9)
    return min(stable_timestep(velocity, workspace.xworkspace; cfl),
               stable_timestep(velocity, workspace.yworkspace; cfl))
end

"""Interface for tangential viscous wall closures."""
abstract type AbstractWallBoundaryCondition end

struct StationaryWall end
@inline (::StationaryWall)(x, y, time, axis) = zero(
    promote_type(
        typeof(x), typeof(y), typeof(time)
    )
)

"""
    NoSlip([velocity])

No-slip wall condition for staggered diffusion. `velocity`, when supplied, is
called as `velocity(x, y, time, axis)` and prescribes the tangential wall
velocity. The wall-normal velocity is always zero (no permeation).
"""
struct NoSlip{F} <: AbstractWallBoundaryCondition
    velocity::F
end

NoSlip() = NoSlip(StationaryWall())

"""Free-slip wall condition: zero normal velocity and zero tangential shear."""
struct FreeSlip <: AbstractWallBoundaryCondition end

"""Prescribed velocity at an inflow side; `velocity(x,y,time,axis)` is sampled."""
struct Inflow{F}
    velocity::F
end

"""Zero-normal-gradient velocity with a flux-balanced extrapolated normal value."""
struct Outflow end

"""Zero normal derivative for every velocity component without flux correction."""
struct ZeroGradientVelocity end

struct BoundaryExtrapolationSchedule
    topology_identity::Base.RefValue{Nothing}
    generation::UInt
    sources::Vector{Vector{Int}}
end

"""
    VelocityBoundaryConditions(; left, right, bottom, top)

Side-specific velocity conditions. Each side accepts `NoSlip()`, `FreeSlip()`,
`Inflow(f)`, `Outflow()`, or `ZeroGradientVelocity()`. This object can be
passed through the existing `boundary` keyword. The default remains four
stationary no-slip walls.
"""
struct VelocityBoundaryConditions{L, R, B, U} <: AbstractWallBoundaryCondition
    left::L
    right::R
    bottom::B
    top::U
    cache::Base.RefValue{Union{Nothing, BoundaryExtrapolationSchedule}}
end

VelocityBoundaryConditions(left, right, bottom, top) =
    VelocityBoundaryConditions(
    left, right, bottom, top,
    Ref{Union{Nothing, BoundaryExtrapolationSchedule}}(nothing)
)

function VelocityBoundaryConditions(;
        left = NoSlip(), right = NoSlip(),
        bottom = NoSlip(), top = NoSlip(),
    )
    conditions = (left, right, bottom, top)
    all(
        condition -> condition isa Union{
            NoSlip, FreeSlip, Inflow, Outflow, ZeroGradientVelocity,
        },
        conditions,
    ) || throw(
        ArgumentError(
            "velocity sides must be NoSlip(), FreeSlip(), Inflow(f), " *
                "Outflow(), or ZeroGradientVelocity()",
        )
    )
    return VelocityBoundaryConditions(left, right, bottom, top)
end

@inline function _side_boundary(boundary::VelocityBoundaryConditions, axis, side)
    return axis == 1 ? (side < 0 ? boundary.left : boundary.right) :
        (side < 0 ? boundary.bottom : boundary.top)
end

function _boundary_extrapolation_schedule!(
        boundary::VelocityBoundaryConditions, mesh::BlockTreeMesh,
    )
    cached = boundary.cache[]
    topology = mesh.topology_cache
    if !isnothing(cached) &&
            cached.topology_identity === topology.identity &&
            cached.generation == topology.generation
        return cached.sources
    end
    sources = [Int[] for _ in mesh.boundary_faces]
    cells = Dict{CellRef, Vector{Int}}()
    empty_indices = Int[]
    for (index, face) in pairs(mesh.boundary_faces)
        condition = _side_boundary(boundary, face.axis, face.side)
        condition isa Union{Outflow, ZeroGradientVelocity} || continue
        push!(get!(cells, face.cell, Int[]), index)
    end
    for (face_index, face) in pairs(mesh.faces)
        for (cell, incidence_side) in ((face.minus, 1), (face.plus, -1))
            for boundary_index in get(cells, cell, empty_indices)
                boundary_face = mesh.boundary_faces[boundary_index]
                face.axis == boundary_face.axis || continue
                incidence_side == -boundary_face.side || continue
                push!(sources[boundary_index], face_index)
            end
        end
    end
    boundary.cache[] = BoundaryExtrapolationSchedule(
        topology.identity, topology.generation, sources
    )
    return sources
end

"""
    enforce_velocity_boundary!(velocity, boundary; time=0, geometry=nothing)

Apply side-specific physical velocity data. `Outflow()` copies the closest
parallel interior face and then receives one uniform outward correction so
the total open-domain flux is exactly zero, as required by the Neumann
pressure projection.
"""
function enforce_velocity_boundary!(
        velocity::FaceVelocity{T},
        boundary::VelocityBoundaryConditions;
        time = zero(T), geometry = nothing, allow_absent = false,
    ) where {T}
    mesh = velocity.mesh
    isnothing(geometry) || geometry.mesh === mesh || throw(
        ArgumentError("velocity and geometry must share a mesh"),
    )
    sources = _boundary_extrapolation_schedule!(boundary, mesh)
    for (index, face) in pairs(mesh.boundary_faces)
        allow_absent && isnan(velocity.boundary[index]) && continue
        condition = _side_boundary(boundary, face.axis, face.side)
        if condition isa Inflow
            @inbounds velocity.boundary[index] = T(
                condition.velocity(
                    face.center[1], face.center[2], time, face.axis,
                ),
            )
        elseif condition isa Union{Outflow, ZeroGradientVelocity}
            numerator = zero(T)
            denominator = zero(T)
            for source_index in sources[index]
                area = mesh.faces[source_index].area
                open = isnothing(geometry) ? one(T) :
                    geometry.aperture[source_index]
                weight = area * open
                iszero(weight) && continue
                allow_absent && isnan(velocity.interior[source_index]) && continue
                numerator += weight * velocity.interior[source_index]
                denominator += weight
            end
            @inbounds velocity.boundary[index] = denominator > zero(T) ?
                numerator / denominator : zero(T)
        else
            @inbounds velocity.boundary[index] = zero(T)
        end
    end
    net_flux = zero(T)
    outflow_area = zero(T)
    for (index, face) in pairs(mesh.boundary_faces)
        allow_absent && isnan(velocity.boundary[index]) && continue
        aperture = isnothing(geometry) ? one(T) :
            geometry.boundary_aperture[index]
        area = T(face.area) * aperture
        @inbounds net_flux += face.side * velocity.boundary[index] * area
        _side_boundary(boundary, face.axis, face.side) isa Outflow &&
            (outflow_area += area)
    end
    if outflow_area > zero(T)
        correction = -net_flux / outflow_area
        for (index, face) in pairs(mesh.boundary_faces)
            allow_absent && isnan(velocity.boundary[index]) && continue
            _side_boundary(boundary, face.axis, face.side) isa Outflow ||
                continue
            @inbounds velocity.boundary[index] += face.side * correction
        end
    elseif abs(net_flux) > T(256) * eps(T)
        throw(
            ArgumentError(
                "prescribed boundary velocities have nonzero net flux but no open outflow",
            )
        )
    end
    return velocity
end

enforce_velocity_boundary!(
    velocity::FaceVelocity, ::Union{NoSlip, FreeSlip}; kwargs...
) = enforce_no_permeation!(velocity)

struct DualDiffusionConnection{T}
    minus::Int
    plus::Int
    axis::Int
    measure::T
    distance::T
    refinement_interface::Bool
end

struct WallDiffusionConnection{T}
    degree_of_freedom::Int
    conductance::T
    point::NTuple{2, T}
    tangential::Bool
    boundary_axis::Int8
    boundary_side::Int8
end

WallDiffusionConnection(degree, conductance, point, tangential) =
    WallDiffusionConnection(
    degree, conductance, point, tangential, Int8(0), Int8(0),
)

struct StaggeredDiffusionComponent{Axis, T}
    stiffness::SparseMatrixCSC{T, Int}
    laplacian::SparseMatrixCSC{T, Int}
    volumes::Vector{T}
    local_to_global::Vector{Int}
    wall_connections::Vector{WallDiffusionConnection{T}}
    input::Vector{T}
    output::Vector{T}
    maximum_row_sum::T
end

"""
    StaggeredDiffusionOperator(mesh; viscosity=1, boundary=NoSlip())

Construct the componentwise staggered operator `viscosity*Δu`. In its stable
core, fluxes are shared between adjacent dual control volumes, and abrupt AMR
stretching is included through the actual face-centre distance. At a split
coarse--fine interface the dual-face measure is inherited from the factored
primary-flux schedule. A two-ring band uses a linear-exact normal-gradient
closure and a wider quadratic Laplacian correction, following the distinction
between stretching and splitting made by Van der Plas. The uncorrected
integrated stiffness is symmetric positive semidefinite; the corrected AMR
operator is generally nonsymmetric but restores consistency and has a
left-half-plane spectrum in the tested 2:1 interface configurations.

Physical walls are impermeable. `NoSlip()` (the default) imposes stationary
tangential Dirichlet data; pass `NoSlip(f)` for a moving wall or `FreeSlip()`
for zero tangential shear. The component matrices and dual volumes are stored
explicitly and are reused by the implicit viscous solve.
"""
struct StaggeredDiffusionOperator{T, M <: BlockTreeMesh, X, Y, B <: AbstractWallBoundaryCondition}
    mesh::M
    x::X
    y::Y
    viscosity::T
    boundary::B
end

function _component_local_to_global(mesh::BlockTreeMesh, axis::Int)
    global_to_local, count = _component_face_indices(mesh, axis)
    local_to_global = Vector{Int}(undef, count)
    for global_index in eachindex(global_to_local)
        local_index = global_to_local[global_index]
        iszero(local_index) || (local_to_global[local_index] = global_index)
    end
    return global_to_local, local_to_global
end

@inline function _component_face_distance(
        mesh::BlockTreeMesh, first,
        second, axis::Int
    )
    delta = second[axis] - first[axis]
    return abs(_wrapped_delta(delta, mesh.extent[axis], mesh.periodic[axis]))
end

function _dual_diffusion_connections(
        mesh::BlockTreeMesh,
        schedule::UniformDualAdvectionSchedule{T},
        local_to_global
    ) where {T}
    connections = Vector{DualDiffusionConnection{T}}(undef, length(schedule.faces))
    for (index, face) in pairs(schedule.faces)
        axis = mesh.faces[face.flux_a].axis
        connections[index] = DualDiffusionConnection(
            face.minus, face.plus, axis, face.area, face.distance, false
        )
    end
    return connections
end

function _dual_diffusion_connections(
        mesh::BlockTreeMesh,
        schedule::AMRDualAdvectionSchedule{T},
        local_to_global
    ) where {T}
    connections = DualDiffusionConnection{T}[]
    cursor = 1
    while cursor <= length(schedule.faces)
        first_face = schedule.faces[cursor]
        minus, plus = first_face.minus, first_face.plus
        signed_measures = zeros(T, 2)
        absolute_measure = zero(T)
        refinement_interface = false
        while cursor <= length(schedule.faces)
            face = schedule.faces[cursor]
            (face.minus == minus && face.plus == plus) || break
            refinement_interface |= face.refinement_interface
            first_weight = face.first_weight
            last_weight = first_weight + face.weight_count - 1
            for index in first_weight:last_weight
                primal_index = schedule.flux_indices[index]
                primal_face = mesh.faces[primal_index]
                contribution = schedule.flux_weights[index] * T(primal_face.area)
                signed_measures[primal_face.axis] += contribution
                absolute_measure += abs(contribution)
            end
            cursor += 1
        end
        minus_face = mesh.faces[local_to_global[minus]]
        plus_face = mesh.faces[local_to_global[plus]]
        connection_count = length(connections)
        for axis in 1:2
            measure = abs(signed_measures[axis])
            measure <= 256eps(T) * absolute_measure && continue
            distance = _component_face_distance(
                mesh, minus_face.center, plus_face.center, axis
            )
            distance > 256eps(T) * mesh.extent[axis] || continue
            push!(
                connections, DualDiffusionConnection(
                    minus, plus, axis, measure, distance,
                    refinement_interface
                )
            )
        end
        if length(connections) == connection_count
            absolute_measure <= eps(T) && continue
            throw(ArgumentError("degenerate AMR dual diffusion connection"))
        end
    end
    return connections
end

function _wall_diffusion_connections(
        mesh::BlockTreeMesh, ::Val{Axis}, ::Type{T},
        incidences, global_to_local
    ) where {Axis, T}
    nx, ny = mesh.block_size
    connections = WallDiffusionConnection{T}[]
    for boundary_face in mesh.boundary_faces
        cell_index = _cell_index(boundary_face.cell, nx, ny)
        block = mesh.blocks[boundary_face.cell.block]
        wall_axis = boundary_face.axis
        tangential = wall_axis != Axis
        for incidence in incidences[cell_index]
            if tangential
                # Each component face owns half of the part of the wall that
                # lies in this primal cell. Split incidences partition it at
                # nonconforming block boundaries.
                measure = T(boundary_face.area) * incidence.fraction / 2
            else
                # The boundary-normal value itself is prescribed. Connect it
                # only to the component face on the opposite side of the cell.
                incidence.side == -boundary_face.side || continue
                measure = T(boundary_face.area) * incidence.fraction
            end
            global_index = global_to_local[incidence.global_index]
            component_face = mesh.faces[incidence.global_index]
            point = if wall_axis == 1
                (T(boundary_face.center[1]), T(component_face.center[2]))
            else
                (T(component_face.center[1]), T(boundary_face.center[2]))
            end
            distance = abs(T(component_face.center[wall_axis]) - point[wall_axis])
            distance > zero(T) || continue
            push!(
                connections, WallDiffusionConnection(
                    global_index, measure / distance, point, tangential,
                    Int8(boundary_face.axis), Int8(boundary_face.side),
                )
            )
        end
    end
    return connections
end

@inline _include_wall_connection(::NoSlip, connection::WallDiffusionConnection) = true
@inline _include_wall_connection(::FreeSlip, connection::WallDiffusionConnection) =
    !connection.tangential
@inline function _include_wall_connection(
        boundary::VelocityBoundaryConditions,
        connection::WallDiffusionConnection,
    )
    iszero(connection.boundary_axis) && return true # embedded stationary wall
    side = _side_boundary(
        boundary, connection.boundary_axis, connection.boundary_side,
    )
    side isa Union{Outflow, ZeroGradientVelocity} && return false
    side isa FreeSlip && return !connection.tangential
    return true
end

function _connection_adjacency(number_of_faces, connections)
    adjacency = [Int[] for _ in 1:number_of_faces]
    for connection in connections
        push!(adjacency[connection.minus], connection.plus)
        push!(adjacency[connection.plus], connection.minus)
    end
    foreach(unique!, adjacency)
    return adjacency
end

function _expand_graph_neighbourhood!(selected, frontier, adjacency, depth, excluded = nothing)
    for _ in 1:depth
        next_frontier = Int[]
        for index in frontier, neighbour in adjacency[index]
            !isnothing(excluded) && neighbour in excluded && continue
            neighbour in selected && continue
            push!(selected, neighbour)
            push!(next_frontier, neighbour)
        end
        frontier = next_frontier
    end
    return selected
end

function _nearby_degrees_of_freedom(connection, adjacency)
    selected = Set((connection.minus, connection.plus))
    frontier = [connection.minus, connection.plus]
    _expand_graph_neighbourhood!(selected, frontier, adjacency, 2)
    return sort!(collect(selected))
end

function _gradient_stencil(
        mesh, connection::DualDiffusionConnection{T},
        local_to_global, adjacency
    ) where {T}
    minus_point = mesh.faces[local_to_global[connection.minus]].center
    plus_point = mesh.faces[local_to_global[connection.plus]].center
    tangent_axis = 3 - connection.axis
    tangent_offset = abs(
        _wrapped_delta(
            plus_point[tangent_axis] - minus_point[tangent_axis],
            mesh.extent[tangent_axis], mesh.periodic[tangent_axis]
        )
    )
    if tangent_offset <= 256eps(T) * mesh.extent[tangent_axis]
        return (
            [connection.minus, connection.plus],
            T[-inv(connection.distance), inv(connection.distance)],
        )
    end

    # A two-point difference across a split AMR interface contains a spurious
    # tangential derivative. Use the immediate two-ring face graph to form the
    # minimum-norm linear-exact normal derivative. Squaring this wider row in
    # the viscous energy retains symmetry and non-negative dissipation.
    indices = _nearby_degrees_of_freedom(connection, adjacency)
    number_of_points = length(indices)
    constraints = Matrix{T}(undef, 3, number_of_points)
    weights = Vector{T}(undef, number_of_points)
    length_scale = max(connection.distance, tangent_offset)
    for (column, local_index) in pairs(indices)
        point = mesh.faces[local_to_global[local_index]].center
        dx = T(
            _wrapped_delta(
                point[1] - minus_point[1],
                mesh.extent[1], mesh.periodic[1]
            )
        )
        dy = T(
            _wrapped_delta(
                point[2] - minus_point[2],
                mesh.extent[2], mesh.periodic[2]
            )
        )
        constraints[1, column] = one(T)
        constraints[2, column] = dx
        constraints[3, column] = dy
        weights[column] = inv(dx^2 + dy^2 + length_scale^2 / 16)
    end
    weighted_transpose = Diagonal(weights) * transpose(constraints)
    moment = constraints * weighted_transpose
    isposdef(Symmetric(moment)) ||
        throw(ArgumentError("degenerate widened AMR diffusion stencil"))
    target = T[
        zero(T), connection.axis == 1 ? one(T) : zero(T),
        connection.axis == 2 ? one(T) : zero(T),
    ]
    coefficients = weighted_transpose * (moment \ target)
    return indices, coefficients
end

function _diffusion_stiffness(
        mesh, local_to_global, connections,
        wall_connections, boundary, ::Type{T}
    ) where {T}
    number_of_faces = length(local_to_global)
    rows = Int[]
    columns = Int[]
    coefficients = T[]
    adjacency = _connection_adjacency(number_of_faces, connections)
    for connection in connections
        indices, stencil = _gradient_stencil(
            mesh, connection, local_to_global, adjacency
        )
        weight = connection.measure * connection.distance
        for (local_column, column) in pairs(indices)
            for (local_row, row) in pairs(indices)
                push!(rows, row)
                push!(columns, column)
                push!(
                    coefficients,
                    weight * stencil[local_row] * stencil[local_column]
                )
            end
        end
    end
    for connection in wall_connections
        _include_wall_connection(boundary, connection) || continue
        push!(rows, connection.degree_of_freedom)
        push!(columns, connection.degree_of_freedom)
        push!(coefficients, connection.conductance)
    end
    stiffness = sparse(
        rows, columns, coefficients,
        number_of_faces, number_of_faces
    )
    stiffness = sparse((stiffness + transpose(stiffness)) / 2)
    dropzeros!(stiffness)
    return stiffness
end

function _graph_neighbourhood(index, adjacency, depth)
    selected = Set((index,))
    frontier = [index]
    _expand_graph_neighbourhood!(selected, frontier, adjacency, depth)
    return sort!(collect(selected))
end

function _quadratic_laplacian_stencil(
        mesh, index, local_to_global,
        adjacency, ::Type{T}
    ) where {T}
    indices = _graph_neighbourhood(index, adjacency, 3)
    target_point = mesh.faces[local_to_global[index]].center
    constraints = Matrix{T}(undef, 6, length(indices))
    offsets = Vector{NTuple{2, T}}(undef, length(indices))
    scale = eps(T)
    for (column, local_index) in pairs(indices)
        point = mesh.faces[local_to_global[local_index]].center
        dx = T(
            _wrapped_delta(
                point[1] - target_point[1],
                mesh.extent[1], mesh.periodic[1]
            )
        )
        dy = T(
            _wrapped_delta(
                point[2] - target_point[2],
                mesh.extent[2], mesh.periodic[2]
            )
        )
        offsets[column] = (dx, dy)
        scale = max(scale, abs(dx), abs(dy))
    end
    distances_squared = Vector{T}(undef, length(indices))
    for column in eachindex(indices)
        dx = offsets[column][1] / scale
        dy = offsets[column][2] / scale
        distances_squared[column] = dx^2 + dy^2
        constraints[:, column] .=
            (one(T), dx, dy, dx^2 / 2, dx * dy, dy^2 / 2)
    end
    weights = @. inv(distances_squared + T(1 // 64))
    weighted_transpose = Diagonal(weights) * transpose(constraints)
    moment = constraints * weighted_transpose
    isposdef(Symmetric(moment)) ||
        throw(ArgumentError("degenerate quadratic AMR diffusion stencil"))
    coefficients = weighted_transpose *
        (moment \ T[0, 0, 0, inv(scale)^2, 0, inv(scale)^2])
    return indices, coefficients
end

function _corrected_laplacian(
        mesh, local_to_global, volumes, stiffness,
        connections, wall_connections, ::Type{T}
    ) where {T}
    inverse_volumes = spdiagm(0 => inv.(volumes))
    base = sparse(-inverse_volumes * stiffness)
    adjacency = _connection_adjacency(length(local_to_global), connections)
    wall_dofs = Set(connection.degree_of_freedom for connection in wall_connections)
    affected = Set{Int}()
    for connection in connections
        connection.refinement_interface || continue
        connection.minus in wall_dofs || push!(affected, connection.minus)
        connection.plus in wall_dofs || push!(affected, connection.plus)
    end
    _expand_graph_neighbourhood!(affected, collect(affected), adjacency, 2, wall_dofs)
    isempty(affected) && return base

    rows = Int[]
    columns = Int[]
    coefficients = T[]
    base_rows = rowvals(base)
    base_values = nonzeros(base)
    for column in axes(base, 2), pointer in nzrange(base, column)
        row = base_rows[pointer]
        row in affected && continue
        push!(rows, row)
        push!(columns, column)
        push!(coefficients, base_values[pointer])
    end
    for row in sort!(collect(affected))
        indices, stencil = _quadratic_laplacian_stencil(
            mesh, row, local_to_global, adjacency, T
        )
        append!(rows, fill(row, length(indices)))
        append!(columns, indices)
        append!(coefficients, stencil)
    end
    laplacian = sparse(
        rows, columns, coefficients,
        length(local_to_global), length(local_to_global)
    )
    dropzeros!(laplacian)
    return laplacian
end

function _maximum_absolute_row_sum(matrix::SparseMatrixCSC{T}) where {T}
    row_sums = zeros(T, size(matrix, 1))
    rows = rowvals(matrix)
    values = nonzeros(matrix)
    for column in axes(matrix, 2), pointer in nzrange(matrix, column)
        @inbounds row_sums[rows[pointer]] += abs(values[pointer])
    end
    return maximum(row_sums; init = zero(T))
end

function _staggered_diffusion_component(
        mesh::BlockTreeMesh, ::Val{Axis},
        boundary, ::Type{T}
    ) where {Axis, T}
    field = FaceField{Axis}(mesh, T)
    advection_workspace = StaggeredAdvectionWorkspace(field)
    schedule = advection_workspace.schedule
    incidences, global_to_local, incidence_volumes =
        _component_incidences(mesh, Val(Axis), T)
    _, local_to_global = _component_local_to_global(mesh, Axis)
    volumes = schedule isa UniformDualAdvectionSchedule ?
        fill(T(schedule.volume), length(local_to_global)) : copy(schedule.volumes)
    isapprox(volumes, incidence_volumes; atol = 256eps(T), rtol = 256eps(T)) ||
        throw(ArgumentError("inconsistent staggered dual volumes"))
    connections = _dual_diffusion_connections(mesh, schedule, local_to_global)
    wall_connections = _wall_diffusion_connections(
        mesh, Val(Axis), T, incidences, global_to_local
    )
    stiffness = _diffusion_stiffness(
        mesh, local_to_global, connections, wall_connections, boundary, T
    )
    laplacian = _corrected_laplacian(
        mesh, local_to_global, volumes, stiffness, connections,
        wall_connections, T
    )
    return StaggeredDiffusionComponent{Axis, T}(
        stiffness, laplacian, volumes, local_to_global, wall_connections,
        zeros(T, length(local_to_global)), zeros(T, length(local_to_global)),
        _maximum_absolute_row_sum(laplacian)
    )
end

function _staggered_diffusion_operator(
        mesh::M; viscosity = 1, boundary = NoSlip(),
        T = eltype(mesh.origin), kwargs...
    ) where {M <: BlockTreeMesh}
    return _staggered_diffusion_operator(mesh, viscosity, boundary, T; kwargs...)
end

function _staggered_diffusion_operator(
        mesh::M, viscosity, boundary, T;
        construction_cache = nothing, reuse = nothing, cache_layer = 1,
        geometry = nothing,
    ) where {M <: BlockTreeMesh}
    isnothing(geometry) || throw(
        ArgumentError("geometry applies only to variable diffusion here"),
    )
    isnothing(construction_cache) && isnothing(reuse) &&
        cache_layer == 1 || throw(
        ArgumentError(
            "construction_cache, reuse, and non-default cache_layer apply only to variable diffusion",
        ),
    )
    _check_real_type(T, "staggered diffusion requires Float32 or Float64")
    viscosity_value = T(viscosity)
    viscosity_value > zero(T) ||
        throw(ArgumentError("constant viscosity must be strictly positive"))
    boundary isa AbstractWallBoundaryCondition ||
        throw(ArgumentError("boundary must be NoSlip() or FreeSlip()"))
    x = _staggered_diffusion_component(mesh, Val(1), boundary, T)
    y = _staggered_diffusion_component(mesh, Val(2), boundary, T)
    return StaggeredDiffusionOperator{T, M, typeof(x), typeof(y), typeof(boundary)}(
        mesh, x, y, viscosity_value, boundary
    )
end

function StaggeredDiffusionOperator(
        mesh::M; viscosity = 1, boundary = NoSlip(),
        T = eltype(mesh.origin),
    ) where {M <: BlockTreeMesh}
    return _staggered_diffusion_operator(mesh; viscosity, boundary, T)
end

@inline function _wall_value(boundary::NoSlip, connection, time, axis, ::Type{T}) where {T}
    connection.tangential || return zero(T)
    return T(
        boundary.velocity(
            connection.point[1], connection.point[2], time, axis
        )
    )
end

@inline _wall_value(::FreeSlip, connection, time, axis, ::Type{T}) where {T} = zero(T)
@inline function _wall_value(
        boundary::VelocityBoundaryConditions, connection, time, axis,
        ::Type{T},
    ) where {T}
    iszero(connection.boundary_axis) && return zero(T)
    side = _side_boundary(
        boundary, connection.boundary_axis, connection.boundary_side,
    )
    side isa Inflow && return T(
        side.velocity(connection.point[1], connection.point[2], time, axis),
    )
    side isa NoSlip && connection.tangential && return T(
        side.velocity(connection.point[1], connection.point[2], time, axis),
    )
    return zero(T)
end

function _apply_staggered_diffusion!(
        destination, source,
        component::StaggeredDiffusionComponent{Axis, T},
        viscosity, boundary, time
    ) where {Axis, T}
    for local_index in eachindex(component.local_to_global)
        global_index = component.local_to_global[local_index]
        @inbounds component.input[local_index] = source[global_index]
    end
    mul!(component.output, component.laplacian, component.input)
    for local_index in eachindex(component.output)
        @inbounds component.output[local_index] =
            viscosity * component.output[local_index]
    end
    for connection in component.wall_connections
        _include_wall_connection(boundary, connection) || continue
        wall_value = _wall_value(boundary, connection, time, Axis, T)
        @inbounds component.output[connection.degree_of_freedom] +=
            viscosity * connection.conductance * wall_value /
            component.volumes[connection.degree_of_freedom]
    end
    for local_index in eachindex(component.local_to_global)
        global_index = component.local_to_global[local_index]
        @inbounds destination[global_index] = component.output[local_index]
    end
    return destination
end

"""
    diffusion!(destination, velocity, operator; time=0)

Apply the strong-form staggered viscous operator. `destination` must not alias
`velocity`; physical boundary rates are zero because their normal velocities
are prescribed rather than evolved.
"""
function diffusion!(
        destination::FaceVelocity{T}, velocity::FaceVelocity{T},
        operator::StaggeredDiffusionOperator{T}; time = 0
    ) where {T}
    destination.mesh === velocity.mesh === operator.mesh ||
        throw(ArgumentError("diffusion arguments must share a mesh"))
    destination === velocity &&
        throw(ArgumentError("diffusion! destination must not alias velocity"))
    _apply_staggered_diffusion!(
        destination.interior, velocity.interior,
        operator.x, operator.viscosity,
        operator.boundary, time
    )
    _apply_staggered_diffusion!(
        destination.interior, velocity.interior,
        operator.y, operator.viscosity,
        operator.boundary, time
    )
    fill!(destination.boundary, zero(T))
    return destination
end

"""Absolute-row-sum explicit time-step estimate for the assembled operator."""
function stable_timestep(operator::StaggeredDiffusionOperator{T}; cfl = 0.9) where {T}
    zero(T) < cfl <= one(T) || throw(ArgumentError("cfl must lie in (0, 1]"))
    maximum_rate = zero(T)
    for component in (operator.x, operator.y)
        maximum_rate = max(
            maximum_rate,
            operator.viscosity * component.maximum_row_sum
        )
    end
    return iszero(maximum_rate) ? T(Inf) : T(cfl) / maximum_rate
end

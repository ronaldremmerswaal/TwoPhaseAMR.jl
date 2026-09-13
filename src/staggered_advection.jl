abstract type AbstractStaggeredAdvectionScheme end

"""First-order upwind interpolation on staggered control-volume faces."""
struct Upwind <: AbstractStaggeredAdvectionScheme end

"""Equal-weight interpolation on staggered control-volume faces."""
struct Centred <: AbstractStaggeredAdvectionScheme end

"""Second-order Lax--Wendroff interpolation using the local dual-face CFL."""
struct LaxWendroff <: AbstractStaggeredAdvectionScheme end

const Centered = Centred
const LW = LaxWendroff

struct DualFace{T}
    minus::Int
    plus::Int
    flux_a::Int
    flux_b::Int
    tangent_flux_a::Int
    tangent_flux_b::Int
    minus_tangent_minus::Int
    minus_tangent_plus::Int
    plus_tangent_minus::Int
    plus_tangent_plus::Int
    area::T
    distance::T
    tangent_distance::T
end

struct RawDualFace{T}
    minus::Int
    plus::Int
    flux_a::Int
    flux_b::Int
    tangent_flux_a::Int
    tangent_flux_b::Int
    axis::Int
    area::T
    distance::T
    tangent_distance::T
end

abstract type AbstractDualAdvectionSchedule{T} end

struct UniformDualAdvectionSchedule{T} <: AbstractDualAdvectionSchedule{T}
    faces::Vector{DualFace{T}}
    volume::T
end

struct AMRDualFace
    minus::Int
    plus::Int
    first_weight::Int
    weight_count::Int
    refinement_interface::Bool
end

struct AMRDualAdvectionSchedule{T} <: AbstractDualAdvectionSchedule{T}
    faces::Vector{AMRDualFace}
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
cell divergence. The schedule is built once; staggered transport does not
traverse the tree.
"""
struct StaggeredAdvectionWorkspace{
        Axis,T,M<:BlockTreeMesh,S<:AbstractDualAdvectionSchedule{T}}
    tendency::FaceField{Axis,T,M}
    rates::FaceField{Axis,T,M}
    schedule::S
end

"""Workspace for simultaneous transport of both velocity components."""
struct StaggeredVelocityAdvectionWorkspace{T,M<:BlockTreeMesh,WX,WY}
    x::FaceField{1,T,M}
    y::FaceField{2,T,M}
    xworkspace::WX
    yworkspace::WY
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
                xminus[cell], xplus[cell], yminus[cell], yplus[cell], 1,
                T(dy), T(dx), T(dy))
            # Transverse dual face through the cell's upper-right vertex.
            raw_faces[n + cell] = RawDualFace(
                global_to_local[xplus[cell]], global_to_local[xplus[above]],
                yplus[cell], yplus[right], xplus[cell], xplus[above], 2,
                T(dx), T(dy), T(dx))
        else
            # Normal dual face through the cell centre.
            raw_faces[cell] = RawDualFace(
                global_to_local[yminus[cell]], global_to_local[yplus[cell]],
                yminus[cell], yplus[cell], xminus[cell], xplus[cell], 2,
                T(dx), T(dy), T(dx))
            # Transverse dual face through the cell's upper-right vertex.
            raw_faces[n + cell] = RawDualFace(
                global_to_local[yplus[cell]], global_to_local[yplus[right]],
                xplus[cell], xplus[above], yplus[cell], yplus[right], 1,
                T(dy), T(dx), T(dy))
        end
    end

    # Map the regular staggered lattice in both coordinate directions. These
    # neighbours are needed only by the multidimensional LW interpolant.
    negative = zeros(Int, 2, component_count)
    positive = zeros(Int, 2, component_count)
    for face in raw_faces
        positive[face.axis, face.minus] = face.plus
        negative[face.axis, face.plus] = face.minus
    end
    all(>(0), negative) && all(>(0), positive) ||
        throw(ArgumentError("incomplete periodic dual-face schedule"))
    dual_faces = Vector{DualFace{T}}(undef, length(raw_faces))
    for (index, face) in pairs(raw_faces)
        tangent_axis = 3 - face.axis
        dual_faces[index] = DualFace(
            face.minus, face.plus, face.flux_a, face.flux_b,
            face.tangent_flux_a, face.tangent_flux_b,
            negative[tangent_axis, face.minus], positive[tangent_axis, face.minus],
            negative[tangent_axis, face.plus], positive[tangent_axis, face.plus],
            face.area, face.distance, face.tangent_distance)
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
        isapprox(fraction_sum, one(T); atol=tolerance, rtol=tolerance) ||
            throw(ArgumentError("nonconforming AMR component-face partition"))
    end
    all(>(zero(T)), volumes) ||
        throw(ArgumentError("AMR staggered control volume has non-positive size"))
    return incidences, global_to_local, volumes
end

function _amr_dual_schedule(mesh::BlockTreeMesh, ::Val{Axis}, ::Type{T}) where {Axis,T}
    all(mesh.periodic) || throw(ArgumentError(
        "staggered advection currently requires periodic boundaries in both directions"))
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
    ])
    dual_faces = AMRDualFace[]
    flux_indices = Int[]
    flux_weights = T[]
    cursor = 1
    while cursor <= length(keys_in_order)
        minus, plus, _ = keys_in_order[cursor]
        first_weight = length(flux_indices) + 1
        while cursor <= length(keys_in_order) &&
              keys_in_order[cursor][1] == minus &&
              keys_in_order[cursor][2] == plus
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
        push!(dual_faces, AMRDualFace(
            minus, plus, first_weight, weight_count, refinement_interface))
    end
    return AMRDualAdvectionSchedule(
        dual_faces, flux_indices, flux_weights, volumes)
end

function StaggeredAdvectionWorkspace(field::FaceField{Axis,T,M}) where {Axis,T,M}
    mesh = field.mesh
    all(mesh.periodic) || throw(ArgumentError(
        "staggered advection currently requires periodic boundaries in both directions"))
    _, component_count = _component_face_indices(mesh, Axis)
    component_count == length(field.interior) ||
        throw(ArgumentError("face field does not match the mesh schedule"))
    first_level = first(mesh.blocks).key.level
    uniform = all(block -> block.key.level == first_level, mesh.blocks)
    schedule = uniform ? _uniform_dual_schedule(mesh, Val(Axis), T) :
                         _amr_dual_schedule(mesh, Val(Axis), T)
    tendency = FaceField{Axis}(mesh, T)
    rates = FaceField{Axis}(mesh, T)
    return StaggeredAdvectionWorkspace(tendency, rates, schedule)
end

function StaggeredVelocityAdvectionWorkspace(velocity::FaceVelocity{T,M}) where {T,M}
    x = FaceField{1}(velocity)
    y = FaceField{2}(velocity)
    return StaggeredVelocityAdvectionWorkspace(
        x, y, StaggeredAdvectionWorkspace(x), StaggeredAdvectionWorkspace(y))
end

StaggeredAdvectionWorkspace(velocity::FaceVelocity) =
    StaggeredVelocityAdvectionWorkspace(velocity)

@inline _dual_state(qminus, qplus, flux, values, velocity, dt, face, ::Upwind) =
    ifelse(flux >= zero(flux), qminus, qplus)

@inline _dual_state(qminus, qplus, flux, values, velocity, dt, face, ::Centred) =
    (qminus + qplus) / 2

@inline function _dual_state(qminus, qplus, flux, values, velocity, dt,
                             face, ::LaxWendroff)
    normal_speed = flux / face.area
    normal_courant = abs(dt * normal_speed / face.distance)
    @inbounds tangent_speed =
        (velocity.interior[face.tangent_flux_a] +
         velocity.interior[face.tangent_flux_b]) / 2

    if flux >= zero(flux)
        upwind = qminus
        downwind = qplus
        tangent_index = tangent_speed >= zero(tangent_speed) ?
            face.minus_tangent_minus : face.minus_tangent_plus
    else
        upwind = qplus
        downwind = qminus
        tangent_index = tangent_speed >= zero(tangent_speed) ?
            face.plus_tangent_minus : face.plus_tangent_plus
    end
    tangent_fraction = abs(dt * tangent_speed) /
                       ((one(normal_courant) + normal_courant) *
                        face.tangent_distance)
    @inbounds tangent_upwind = values[tangent_index]
    interpolated_upwind = upwind + tangent_fraction * (tangent_upwind - upwind)
    return ((one(normal_courant) - normal_courant) * downwind +
            (one(normal_courant) + normal_courant) * interpolated_upwind) / 2
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

@inline _amr_dual_state(qminus, qplus, flux, dt, minus_volume, plus_volume,
                        ::Upwind) =
    ifelse(flux >= zero(flux), qminus, qplus)

@inline _amr_dual_state(qminus, qplus, flux, dt, minus_volume, plus_volume,
                        ::Centred) = (qminus + qplus) / 2

@inline function _amr_dual_state(qminus, qplus, flux, dt,
                                 minus_volume, plus_volume, ::LaxWendroff)
    # On a nonconforming schedule there is no unique transverse donor. Use the
    # one-dimensional LW state on each algebraic connection. The smaller
    # adjacent dual volume gives a conservative local length scale and is
    # covered by `stable_timestep`. `Upwind` remains the robust AMR option.
    courant = abs(dt * flux) / min(minus_volume, plus_volume)
    upwind, downwind = flux >= zero(flux) ? (qminus, qplus) : (qplus, qminus)
    return ((one(courant) - courant) * downwind +
            (one(courant) + courant) * upwind) / 2
end

function _convection_schedule!(destination, advected, velocity, dt,
                               schedule::UniformDualAdvectionSchedule,
                               scheme, amr_interface_scheme)
    inverse_volume = inv(schedule.volume)
    for face in schedule.faces
        @inbounds begin
            flux = (velocity.interior[face.flux_a] +
                    velocity.interior[face.flux_b]) * face.area / 2
            qminus = advected.interior[face.minus]
            qplus = advected.interior[face.plus]
            state = _dual_state(qminus, qplus, flux, advected.interior,
                                velocity, dt, face, scheme)
            rate = flux * state * inverse_volume
            destination.interior[face.minus] += rate
            destination.interior[face.plus] -= rate
        end
    end
    return destination
end

function _convection_schedule!(destination, advected, velocity, dt,
                               schedule::AMRDualAdvectionSchedule,
                               scheme, amr_interface_scheme)
    for face in schedule.faces
        @inbounds begin
            flux = _amr_dual_flux(face, schedule, velocity)
            qminus = advected.interior[face.minus]
            qplus = advected.interior[face.plus]
            minus_volume = schedule.volumes[face.minus]
            plus_volume = schedule.volumes[face.plus]
            state = if face.refinement_interface
                _amr_dual_state(qminus, qplus, flux, dt,
                                minus_volume, plus_volume,
                                amr_interface_scheme)
            else
                _amr_dual_state(qminus, qplus, flux, dt,
                                minus_volume, plus_volume, scheme)
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
                      scheme, amr_interface_scheme) where {Axis,T}
    _check_staggered_arguments(destination, advected, velocity, workspace)
    fill!(destination.interior, zero(T))
    fill!(destination.boundary, zero(T))
    return _convection_schedule!(
        destination, advected, velocity, dt, workspace.schedule,
        scheme, amr_interface_scheme)
end

function _dispatch_interface_convection!(destination, advected, velocity, dt,
                                         workspace, scheme,
                                         amr_interface_scheme)
    if amr_interface_scheme isa Upwind || amr_interface_scheme === :upwind
        return _convection!(destination, advected, velocity, dt, workspace,
                            scheme, Upwind())
    elseif amr_interface_scheme isa Centred ||
           amr_interface_scheme === :centred ||
           amr_interface_scheme === :centered ||
           amr_interface_scheme === :central
        return _convection!(destination, advected, velocity, dt, workspace,
                            scheme, Centred())
    elseif amr_interface_scheme isa LaxWendroff ||
           amr_interface_scheme === :lw ||
           amr_interface_scheme === :lax_wendroff
        return _convection!(destination, advected, velocity, dt, workspace,
                            scheme, LaxWendroff())
    end
    throw(ArgumentError(
        "unsupported AMR-interface advection scheme: $amr_interface_scheme"))
end

function _dispatch_convection!(destination, advected, velocity, dt, workspace,
                               scheme, amr_interface_scheme)
    if scheme isa Upwind || scheme === :upwind
        return _dispatch_interface_convection!(
            destination, advected, velocity, dt, workspace,
            Upwind(), amr_interface_scheme)
    elseif scheme isa Centred || scheme === :centred ||
           scheme === :centered || scheme === :central
        return _dispatch_interface_convection!(
            destination, advected, velocity, dt, workspace,
            Centred(), amr_interface_scheme)
    elseif scheme isa LaxWendroff || scheme === :lw || scheme === :lax_wendroff
        return _dispatch_interface_convection!(
            destination, advected, velocity, dt, workspace,
            LaxWendroff(), amr_interface_scheme)
    end
    throw(ArgumentError("unsupported staggered advection scheme: $scheme"))
end

"""
    convection!(destination, advected, velocity, dt, workspace;
                scheme=LW(), amr_interface_scheme=scheme)

Evaluate the conservative staggered convection operator `div(I(velocity) q)`
on owned dual control volumes. `scheme` may be `Upwind()`, `Centred()`, or
`LaxWendroff()` (the corresponding symbols are also accepted) and defaults to
`LW()`.

For an all-centred operator (`scheme=Centred()` and
`amr_interface_scheme=Centred()`), the face value is the equal-weight average.
On every supported periodic mesh this makes twice the convection-matrix
diagonal equal the volume-weighted interpolation of the primary cell
divergence. The operator is therefore volume-weighted skew-symmetric for a
discretely divergence-free velocity, including across refinement interfaces.

On a mixed-level mesh, `amr_interface_scheme` selects the state used only on
non-regular factored connections. It defaults to the value of `scheme`, so the
default call applies `LW()` throughout the mesh. Pass `Upwind()` explicitly
for a more dissipative interface closure, or `Centred()` to retain the
diagonal and skew identities throughout the mesh. The keyword is accepted but
has no effect on a uniform mesh.
"""
function convection!(destination::FaceField{Axis}, advected::FaceField{Axis},
                     velocity::FaceVelocity, dt,
                     workspace::StaggeredAdvectionWorkspace{Axis};
                     scheme=LW(),
                     amr_interface_scheme=scheme) where {Axis}
    dt >= zero(dt) || throw(ArgumentError("dt must be nonnegative"))
    return _dispatch_convection!(destination, advected, velocity, dt,
                                 workspace, scheme, amr_interface_scheme)
end

"""
    advect!(field, velocity, dt, workspace;
            scheme=LW(), amr_interface_scheme=scheme)

Advance a directional face field by forward Euler using conservative fluxes
through its staggered control volumes. The Lax--Wendroff flux includes `dt` in
its face interpolation, yielding the usual second-order one-step update on a
uniform grid. By default `amr_interface_scheme` inherits `scheme`, giving
global `LW()`; pass `amr_interface_scheme=Upwind()` to select the robust
hybrid closure.
"""
function advect!(field::FaceField{Axis,T}, velocity::FaceVelocity, dt,
                 workspace::StaggeredAdvectionWorkspace{Axis,T};
                 scheme=LW(),
                 amr_interface_scheme=scheme) where {Axis,T}
    field === workspace.tendency && throw(ArgumentError(
        "the transported field must not be the workspace tendency"))
    dt >= zero(dt) || throw(ArgumentError("dt must be nonnegative"))
    _dispatch_convection!(workspace.tendency, field, velocity, dt, workspace,
                          scheme, amr_interface_scheme)
    for index in eachindex(field.interior)
        @inbounds field.interior[index] -= dt * workspace.tendency.interior[index]
    end
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

"""Evaluate convection for both components of an owned staggered field."""
function convection!(destination::FaceVelocity, advected::FaceVelocity,
                     advecting::FaceVelocity, dt,
                     workspace::StaggeredVelocityAdvectionWorkspace;
                     scheme=LW(),
                     amr_interface_scheme=scheme)
    dt >= zero(dt) || throw(ArgumentError("dt must be nonnegative"))
    return _velocity_convection!(
        destination, advected, advecting, dt, workspace,
        scheme, amr_interface_scheme)
end

"""
    advect!(field, advecting, dt, workspace;
            scheme=LW(), amr_interface_scheme=scheme)
    advect!(velocity, dt, workspace;
            scheme=LW(), amr_interface_scheme=scheme)

Advance both components of a staggered vector field simultaneously. The
one-velocity shorthand uses the old velocity as both advected and advecting
field; both tendencies are evaluated before either component is updated. By
default `amr_interface_scheme` inherits `scheme`, so the no-keyword call uses
`LW()` globally.
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
    _dispatch_convection!(workspace.xworkspace.tendency, workspace.x,
                          advecting, dt, workspace.xworkspace,
                          scheme, amr_interface_scheme)
    _dispatch_convection!(workspace.yworkspace.tendency, workspace.y,
                          advecting, dt, workspace.yworkspace,
                          scheme, amr_interface_scheme)
    for index in eachindex(workspace.x.interior)
        @inbounds workspace.x.interior[index] -=
            dt * workspace.xworkspace.tendency.interior[index]
    end
    for index in eachindex(workspace.y.interior)
        @inbounds workspace.y.interior[index] -=
            dt * workspace.yworkspace.tendency.interior[index]
    end
    copyto!(field, workspace.x)
    copyto!(field, workspace.y)
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

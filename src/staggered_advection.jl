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

"""
    StaggeredAdvectionWorkspace(field)

Build the dual-control-volume schedule for a directional `FaceField`. The
current vertical slice supports fully periodic, uniform-resolution meshes,
including meshes partitioned into multiple tree blocks. The schedule is built
once; staggered transport does not traverse the tree.
"""
struct StaggeredAdvectionWorkspace{Axis,T,M<:BlockTreeMesh}
    tendency::FaceField{Axis,T,M}
    rates::FaceField{Axis,T,M}
    dual_faces::Vector{DualFace{T}}
    dual_volume::T
end

"""Workspace for simultaneous transport of both velocity components."""
struct StaggeredVelocityAdvectionWorkspace{T,M<:BlockTreeMesh}
    x::FaceField{1,T,M}
    y::FaceField{2,T,M}
    xworkspace::StaggeredAdvectionWorkspace{1,T,M}
    yworkspace::StaggeredAdvectionWorkspace{2,T,M}
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

function StaggeredAdvectionWorkspace(field::FaceField{Axis,T,M}) where {Axis,T,M}
    mesh = field.mesh
    xminus, xplus, yminus, yplus = _uniform_periodic_cell_faces(mesh)
    global_to_local, component_count = _component_face_indices(mesh, Axis)
    component_count == length(field.interior) ||
        throw(ArgumentError("face field does not match the mesh schedule"))

    nx, ny = mesh.block_size
    n = ncells(mesh)
    first_block = first(mesh.blocks)
    dx, dy = first_block.spacing
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
    tendency = FaceField{Axis}(mesh, T)
    rates = FaceField{Axis}(mesh, T)
    return StaggeredAdvectionWorkspace(
        tendency, rates, dual_faces, T(dx * dy))
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

function _convection!(destination::FaceField{Axis,T},
                      advected::FaceField{Axis}, velocity::FaceVelocity, dt,
                      workspace::StaggeredAdvectionWorkspace{Axis}, scheme) where {Axis,T}
    _check_staggered_arguments(destination, advected, velocity, workspace)
    fill!(destination.interior, zero(T))
    fill!(destination.boundary, zero(T))
    inverse_volume = inv(workspace.dual_volume)
    for face in workspace.dual_faces
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

function _dispatch_convection!(destination, advected, velocity, dt, workspace, scheme)
    if scheme isa Upwind || scheme === :upwind
        return _convection!(destination, advected, velocity, dt, workspace, Upwind())
    elseif scheme isa Centred || scheme === :centred ||
           scheme === :centered || scheme === :central
        return _convection!(destination, advected, velocity, dt, workspace, Centred())
    elseif scheme isa LaxWendroff || scheme === :lw || scheme === :lax_wendroff
        return _convection!(destination, advected, velocity, dt, workspace, LaxWendroff())
    end
    throw(ArgumentError("unsupported staggered advection scheme: $scheme"))
end

"""
    convection!(destination, advected, velocity, dt, workspace; scheme=Centred())

Evaluate the conservative staggered convection operator `div(I(velocity) q)`
on owned dual control volumes. `scheme` may be `Upwind()`, `Centred()`, or
`LaxWendroff()` (the corresponding symbols are also accepted).

For `Centred()` the face value is the equal-weight average. On a periodic,
uniform mesh this makes the volume-weighted operator skew-symmetric whenever
the cell-centred divergence of `velocity` is zero.
"""
function convection!(destination::FaceField{Axis}, advected::FaceField{Axis},
                     velocity::FaceVelocity, dt,
                     workspace::StaggeredAdvectionWorkspace{Axis};
                     scheme=Centred()) where {Axis}
    dt >= zero(dt) || throw(ArgumentError("dt must be nonnegative"))
    return _dispatch_convection!(destination, advected, velocity, dt, workspace, scheme)
end

"""
    advect!(field, velocity, dt, workspace; scheme=Upwind())

Advance a directional face field by forward Euler using conservative fluxes
through its staggered control volumes. The Lax--Wendroff flux includes `dt` in
its face interpolation, yielding the usual second-order one-step update on a
uniform grid.
"""
function advect!(field::FaceField{Axis,T}, velocity::FaceVelocity, dt,
                 workspace::StaggeredAdvectionWorkspace{Axis,T};
                 scheme=Upwind()) where {Axis,T}
    field === workspace.tendency && throw(ArgumentError(
        "the transported field must not be the workspace tendency"))
    dt >= zero(dt) || throw(ArgumentError("dt must be nonnegative"))
    _dispatch_convection!(workspace.tendency, field, velocity, dt, workspace, scheme)
    for index in eachindex(field.interior)
        @inbounds field.interior[index] -= dt * workspace.tendency.interior[index]
    end
    return field
end

function _velocity_convection!(destination::FaceVelocity,
                               advected::FaceVelocity,
                               advecting::FaceVelocity, dt,
                               workspace::StaggeredVelocityAdvectionWorkspace,
                               scheme)
    destination.mesh === advected.mesh === advecting.mesh === workspace.x.mesh ||
        throw(ArgumentError("all staggered-advection arguments must share a mesh"))
    destination === advected && throw(ArgumentError(
        "convection! destination must not alias its advected field"))
    copyto!(workspace.x, advected)
    copyto!(workspace.y, advected)
    _dispatch_convection!(workspace.xworkspace.tendency, workspace.x,
                          advecting, dt, workspace.xworkspace, scheme)
    _dispatch_convection!(workspace.yworkspace.tendency, workspace.y,
                          advecting, dt, workspace.yworkspace, scheme)
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
                     scheme=Centred())
    dt >= zero(dt) || throw(ArgumentError("dt must be nonnegative"))
    return _velocity_convection!(
        destination, advected, advecting, dt, workspace, scheme)
end

"""
    advect!(field, advecting, dt, workspace; scheme=Upwind())
    advect!(velocity, dt, workspace; scheme=Upwind())

Advance both components of a staggered vector field simultaneously. The
one-velocity shorthand uses the old velocity as both advected and advecting
field; both tendencies are evaluated before either component is updated.
"""
function advect!(field::FaceVelocity, advecting::FaceVelocity, dt,
                 workspace::StaggeredVelocityAdvectionWorkspace;
                 scheme=Upwind())
    field.mesh === advecting.mesh === workspace.x.mesh ||
        throw(ArgumentError("all staggered-advection arguments must share a mesh"))
    dt >= zero(dt) || throw(ArgumentError("dt must be nonnegative"))
    copyto!(workspace.x, field)
    copyto!(workspace.y, field)
    _dispatch_convection!(workspace.xworkspace.tendency, workspace.x,
                          advecting, dt, workspace.xworkspace, scheme)
    _dispatch_convection!(workspace.yworkspace.tendency, workspace.y,
                          advecting, dt, workspace.yworkspace, scheme)
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

"""Return the volume-weighted integral of a uniform-periodic face field."""
function integrate(field::FaceField{Axis},
                   workspace::StaggeredAdvectionWorkspace{Axis}) where {Axis}
    field.mesh === workspace.tendency.mesh ||
        throw(ArgumentError("field and workspace must share a mesh"))
    result = zero(eltype(field))
    for value in field.interior
        result += value * workspace.dual_volume
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
        @inbounds result += a.interior[index] * b.interior[index] * workspace.dual_volume
    end
    return result
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
    inverse_volume = inv(workspace.dual_volume)
    for face in workspace.dual_faces
        @inbounds flux = (velocity.interior[face.flux_a] +
                          velocity.interior[face.flux_b]) * face.area / 2
        rate = abs(flux) * inverse_volume
        @inbounds begin
            workspace.rates.interior[face.minus] += rate
            workspace.rates.interior[face.plus] += rate
        end
    end
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

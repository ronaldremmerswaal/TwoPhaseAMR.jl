@inline function _aliases_staggered_scratch(field, workspace)
    return any(
        scratch ->
        Base.mightalias(field.interior, scratch.interior) ||
            Base.mightalias(field.interior, scratch.boundary) ||
            Base.mightalias(field.boundary, scratch.interior) ||
            Base.mightalias(field.boundary, scratch.boundary),
        (workspace.tendency, workspace.rates)
    )
end

"""
    convection!(destination, advected, velocity, dt, workspace;
                scheme=LaxWendroff(),
                amr_interface_scheme=default_amr_interface_scheme(scheme))

Evaluate the conservative staggered convection operator `div(I(velocity) q)`
on owned dual control volumes. `scheme` may be `Upwind()`, `Centred()`,
`LaxWendroff()`, `Fromm()`, `Minmod()`, `MonotonizedCentral()`, `VanLeer()`,
or `Superbee()` (the corresponding symbols are also accepted); the default is
`LaxWendroff()`.

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
non-regular factored connections. Its default is selected from `scheme`:
upwind remains upwind, centred and LW use MC, and a three-point scheme uses
itself. Upwind at an AMR interface is rejected unless the bulk scheme is also
upwind; an interface-only first-order closure has poor accuracy. The four
limited variants revert to upwind at a local extremum. A hanging connection
without an unambiguous upstream direction also falls back to upwind. Explicit
`Centred()` retains the diagonal and skew identities throughout the mesh. The
keyword is accepted but has no effect on a uniform mesh.
"""
function convection!(
        destination::FaceField{Axis}, advected::FaceField{Axis},
        velocity::FaceVelocity, dt,
        workspace::StaggeredAdvectionWorkspace{Axis};
        scheme = LaxWendroff(),
        boundary = nothing, time = 0,
        amr_interface_scheme =
            default_amr_interface_scheme(scheme)
    ) where {Axis}
    dt >= zero(dt) || throw(ArgumentError("dt must be nonnegative"))
    for field in (destination, advected)
        _aliases_staggered_scratch(field, workspace) &&
            throw(ArgumentError("convection fields must not alias workspace scratch"))
    end
    configuration = isnothing(boundary) ? nothing : (boundary, time)
    _validate_amr_interface_scheme(scheme, amr_interface_scheme)
    if !iszero(dt) && _directionally_split_scheme(
            scheme, amr_interface_scheme, workspace.schedule
        )
        return _split_convection_rate!(
            destination, advected, velocity, dt, workspace,
            scheme, amr_interface_scheme, configuration
        )
    end
    return _dispatch_convection!(
        destination, advected, velocity, dt,
        workspace, scheme, amr_interface_scheme, 0, configuration
    )
end

"""
    advect!(field, velocity, dt, workspace;
            scheme=LaxWendroff(),
            amr_interface_scheme=default_amr_interface_scheme(scheme))

Advance a directional face field using conservative fluxes through its
staggered control volumes. Lax--Wendroff uses two full directional sweeps,
publishing the first intermediate before evaluating the second and alternating
x/y with y/x order on successive calls. The variable-coefficient LW predictor
includes the directional dual divergence of the transporting velocity; the
two directional divergence terms cancel for a divergence-free velocity only
after both sweeps. On a uniform constant-velocity line it reduces to classical
one-dimensional LW. By default LW and centred transport use MC on non-regular
AMR connections, upwind remains globally upwind, and a selected three-point
scheme is used at the interface as well.
"""
function advect!(
        field::FaceField{Axis, T}, velocity::FaceVelocity, dt,
        workspace::StaggeredAdvectionWorkspace{Axis, T};
        scheme = LaxWendroff(),
        boundary = nothing, time = 0,
        amr_interface_scheme =
            default_amr_interface_scheme(scheme)
    ) where {Axis, T}
    _aliases_staggered_scratch(field, workspace) &&
        throw(ArgumentError("the transported field must not alias workspace scratch"))
    dt >= zero(dt) || throw(ArgumentError("dt must be nonnegative"))
    configuration = isnothing(boundary) ? nothing : (boundary, time)
    _validate_amr_interface_scheme(scheme, amr_interface_scheme)
    split = _directionally_split_scheme(
        scheme, amr_interface_scheme, workspace.schedule
    )
    stages = split ?
        (
            workspace.sweep_forward[] ? ((1, dt), (2, dt)) :
            ((2, dt), (1, dt))
        ) :
        ((0, dt),)
    for (sweep_axis, sweep_dt) in stages
        _dispatch_convection!(
            workspace.tendency, field, velocity, sweep_dt,
            workspace,
            scheme, amr_interface_scheme, sweep_axis, configuration
        )
        for index in eachindex(field.interior)
            @inbounds field.interior[index] -=
                sweep_dt * workspace.tendency.interior[index]
        end
    end
    split && !iszero(dt) && (workspace.sweep_forward[] = !workspace.sweep_forward[])
    return field
end

function _velocity_convection!(
        destination::FaceVelocity,
        advected::FaceVelocity,
        advecting::FaceVelocity, dt,
        workspace::StaggeredVelocityAdvectionWorkspace,
        scheme, amr_interface_scheme, configuration
    )
    destination.mesh === advected.mesh === advecting.mesh === workspace.x.mesh ||
        throw(ArgumentError("all staggered-advection arguments must share a mesh"))
    destination === advected && throw(
        ArgumentError(
            "convection! destination must not alias its advected field"
        )
    )
    copyto!(workspace.x, advected)
    copyto!(workspace.y, advected)
    _dispatch_convection!(
        workspace.xworkspace.tendency, workspace.x,
        advecting, dt, workspace.xworkspace,
        scheme, amr_interface_scheme, 0, configuration
    )
    _dispatch_convection!(
        workspace.yworkspace.tendency, workspace.y,
        advecting, dt, workspace.yworkspace,
        scheme, amr_interface_scheme, 0, configuration
    )
    fill!(destination.interior, zero(eltype(destination.interior)))
    fill!(destination.boundary, zero(eltype(destination.boundary)))
    copyto!(destination, workspace.xworkspace.tendency)
    copyto!(destination, workspace.yworkspace.tendency)
    return destination
end

function _split_velocity_convection!(
        destination::FaceVelocity,
        advected::FaceVelocity,
        advecting::FaceVelocity, dt,
        workspace::StaggeredVelocityAdvectionWorkspace,
        scheme, amr_interface_scheme, configuration
    )
    destination.mesh === advected.mesh === advecting.mesh === workspace.x.mesh ||
        throw(ArgumentError("all staggered-advection arguments must share a mesh"))
    destination === advected && throw(
        ArgumentError(
            "convection! destination must not alias its advected field"
        )
    )
    destination === advecting && throw(
        ArgumentError(
            "convection! destination must not alias its advecting field"
        )
    )
    copyto!(destination.interior, advected.interior)
    copyto!(destination.boundary, advected.boundary)
    copyto!(workspace.x, advected)
    copyto!(workspace.y, advected)
    self_advected = advected === advecting
    sweep_axes = workspace.sweep_forward[] ? (1, 2) : (2, 1)
    for sweep_axis in sweep_axes
        sweep_dt = dt
        current_advecting = self_advected ? destination : advecting
        _dispatch_convection!(
            workspace.xworkspace.tendency, workspace.x,
            current_advecting, sweep_dt, workspace.xworkspace,
            scheme, amr_interface_scheme, sweep_axis, configuration
        )
        _dispatch_convection!(
            workspace.yworkspace.tendency, workspace.y,
            current_advecting, sweep_dt, workspace.yworkspace,
            scheme, amr_interface_scheme, sweep_axis, configuration
        )
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
function convection!(
        destination::FaceVelocity, advected::FaceVelocity,
        advecting::FaceVelocity, dt,
        workspace::StaggeredVelocityAdvectionWorkspace;
        scheme = LaxWendroff(),
        boundary = nothing, time = 0,
        amr_interface_scheme =
            default_amr_interface_scheme(scheme)
    )
    dt >= zero(dt) || throw(ArgumentError("dt must be nonnegative"))
    configuration = isnothing(boundary) ? nothing : (boundary, time)
    _validate_amr_interface_scheme(scheme, amr_interface_scheme)
    if !iszero(dt) && _directionally_split_scheme(
            scheme, amr_interface_scheme, workspace.xworkspace.schedule
        )
        return _split_velocity_convection!(
            destination, advected, advecting, dt, workspace,
            scheme, amr_interface_scheme, configuration
        )
    end
    return _velocity_convection!(
        destination, advected, advecting, dt, workspace,
        scheme, amr_interface_scheme, configuration
    )
end

"""
    advect!(field, advecting, dt, workspace;
            scheme=LaxWendroff(),
            amr_interface_scheme=default_amr_interface_scheme(scheme))
    advect!(velocity, dt, workspace;
            scheme=LaxWendroff(),
            amr_interface_scheme=default_amr_interface_scheme(scheme))

Advance both components of a staggered vector field. Within one coordinate
sweep, both component tendencies are evaluated from the same input and then
both components are published. The next coordinate sweep therefore consumes
one consistent intermediate vector rather than updating x and y velocity
components separately. The one-velocity shorthand also uses each published
intermediate as its advecting field. By default LW and centred transport use
MC on non-regular AMR connections, upwind remains globally upwind, and a
selected three-point scheme is retained.
"""
function advect!(
        field::FaceVelocity, advecting::FaceVelocity, dt,
        workspace::StaggeredVelocityAdvectionWorkspace;
        scheme = LaxWendroff(),
        boundary = nothing, time = 0,
        amr_interface_scheme =
            default_amr_interface_scheme(scheme)
    )
    field.mesh === advecting.mesh === workspace.x.mesh ||
        throw(ArgumentError("all staggered-advection arguments must share a mesh"))
    dt >= zero(dt) || throw(ArgumentError("dt must be nonnegative"))
    configuration = isnothing(boundary) ? nothing : (boundary, time)
    _validate_amr_interface_scheme(scheme, amr_interface_scheme)
    copyto!(workspace.x, field)
    copyto!(workspace.y, field)
    self_advected = field === advecting
    split = _directionally_split_scheme(
        scheme, amr_interface_scheme, workspace.xworkspace.schedule
    )
    stages = split ?
        (
            workspace.sweep_forward[] ? ((1, dt), (2, dt)) :
            ((2, dt), (1, dt))
        ) :
        ((0, dt),)
    for (sweep_axis, sweep_dt) in stages
        current_advecting = self_advected ? field : advecting
        _dispatch_convection!(
            workspace.xworkspace.tendency, workspace.x,
            current_advecting, sweep_dt, workspace.xworkspace,
            scheme, amr_interface_scheme, sweep_axis, configuration
        )
        _dispatch_convection!(
            workspace.yworkspace.tendency, workspace.y,
            current_advecting, sweep_dt, workspace.yworkspace,
            scheme, amr_interface_scheme, sweep_axis, configuration
        )
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

advect!(
    velocity::FaceVelocity, dt,
    workspace::StaggeredVelocityAdvectionWorkspace; kwargs...
) =
    advect!(velocity, velocity, dt, workspace; kwargs...)

@inline _dual_volumes(schedule::UniformDualAdvectionSchedule, index) = schedule.volume
@inline _dual_volumes(schedule::AMRDualAdvectionSchedule, index) =
    @inbounds schedule.volumes[index]

"""Return the dual-volume-weighted integral of a periodic face field."""
function integrate(
        field::FaceField{Axis},
        workspace::StaggeredAdvectionWorkspace{Axis}
    ) where {Axis}
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
function staggered_inner_product(
        a::FaceField{Axis}, b::FaceField{Axis},
        workspace::StaggeredAdvectionWorkspace{Axis}
    ) where {Axis}
    a.mesh === b.mesh === workspace.tendency.mesh ||
        throw(ArgumentError("fields and workspace must share a mesh"))
    result = zero(promote_type(eltype(a), eltype(b)))
    for index in eachindex(a.interior, b.interior)
        @inbounds result += a.interior[index] * b.interior[index] *
            _dual_volumes(workspace.schedule, index)
    end
    return result
end

function _accumulate_staggered_rates!(
        rates, velocity,
        workspace::StaggeredAdvectionWorkspace{
            Axis, T, M, <:UniformDualAdvectionSchedule,
        }
    ) where {Axis, T, M}
    schedule = workspace.schedule
    inverse_volume = inv(schedule.volume)
    @_parallel for face_index in 1:length(schedule.faces)
        face = schedule.faces[face_index]
        @inbounds workspace.dual_flux[face_index] = abs(
            (
                velocity.interior[face.flux_a] +
                    velocity.interior[face.flux_b]
            ) * face.area / 2
        )
    end
    @_parallel for output in 1:length(rates)
        value = zero(T)
        for pointer in workspace.dual_offsets[output]:(
                workspace.dual_offsets[output + 1] - 1
            )
            face_index = @inbounds workspace.dual_faces[pointer]
            @inbounds value += workspace.dual_flux[face_index]
        end
        @inbounds rates[output] = value * inverse_volume
    end
    return rates
end

function _accumulate_staggered_rates!(
        rates, velocity,
        workspace::StaggeredAdvectionWorkspace{
            Axis, T, M, <:AMRDualAdvectionSchedule,
        }
    ) where {Axis, T, M}
    schedule = workspace.schedule
    @_parallel for face_index in 1:length(schedule.faces)
        @inbounds workspace.dual_flux[face_index] = abs(
            _amr_dual_flux(schedule.faces[face_index], schedule, velocity)
        )
    end
    @_parallel for output in 1:length(rates)
        value = zero(T)
        for pointer in workspace.dual_offsets[output]:(
                workspace.dual_offsets[output + 1] - 1
            )
            face_index = @inbounds workspace.dual_faces[pointer]
            @inbounds value += workspace.dual_flux[face_index]
        end
        @inbounds rates[output] = value / schedule.volumes[output]
    end
    return rates
end

"""Return a conservative CFL bound for one staggered component."""
function stable_timestep(
        velocity::FaceVelocity{TV},
        workspace::StaggeredAdvectionWorkspace{Axis, TW};
        cfl = 0.9
    ) where {TV, Axis, TW}
    velocity.mesh === workspace.rates.mesh ||
        throw(ArgumentError("velocity and workspace must share a mesh"))
    T = promote_type(TV, TW, typeof(cfl))
    zero(T) < cfl <= one(T) || throw(ArgumentError("cfl must lie in (0, 1]"))
    _accumulate_staggered_rates!(
        workspace.rates.interior, velocity, workspace
    )
    maxrate = zero(T)
    for rate in workspace.rates.interior
        maxrate = max(maxrate, rate)
    end
    return iszero(maxrate) ? T(Inf) : T(cfl) / maxrate
end


function stable_timestep(
        velocity::FaceVelocity,
        workspace::StaggeredVelocityAdvectionWorkspace;
        cfl = 0.9
    )
    return min(
        stable_timestep(velocity, workspace.xworkspace; cfl),
        stable_timestep(velocity, workspace.yworkspace; cfl)
    )
end

"""
    TwoPhaseVelocity

A liquid/gas pair of existing [`FaceVelocity`](@ref) fields, stored as the
ordinary tuple `(liquid_velocity, gas_velocity)`.  The alias adds dispatch and
documentation without introducing another field container or duplicating
velocity storage.
"""
const TwoPhaseVelocity = Tuple{<:FaceVelocity, <:FaceVelocity}

"""
    MomentumAdvectionWorkspace(interface, velocity::TwoPhaseVelocity)

Construct the compatible staggered workspace for separate liquid and gas
momentum transport. The returned object contains intrinsic velocity fields;
it does not retain phase-weighted momentum.
"""
function MomentumAdvectionWorkspace(
        interface::InterfaceField{T, M}, velocity::TwoPhaseVelocity,
    ) where {T, M}
    _validate_two_phase_velocity(velocity, interface.mesh, T)
    return TwoVelocityMomentumAdvectionWorkspace(
        interface, PhaseFaceScalar{1}(interface),
        PhaseFaceScalar{2}(interface),
    )
end

mutable struct ProjectedLiquidAdvectionWorkspace{T, M, O, S, E, FX, FY, SX, SY}
    projected::FaceVelocity{T, M}
    correction::FaceVelocity{T, M}
    pressure::CellField{T, M}
    divergence::CellField{T, M}
    operator::O
    solver::S
    elliptic::E
    fraction_x::FX
    fraction_y::FY
    schedule_x::SX
    schedule_y::SY
    global_to_local_x::Vector{Int}
    global_to_local_y::Vector{Int}
    values_x::Vector{T}
    values_y::Vector{T}
    known_x::BitVector
    known_y::BitVector
    next_known_x::BitVector
    next_known_y::BitVector
    sums_x::Vector{T}
    sums_y::Vector{T}
    counts_x::Vector{Int}
    counts_y::Vector{Int}
    face_coefficient::Vector{T}
    active_cells::BitVector
    cell_volumes::Vector{T}
end

_interface_method_workspace(field, ::FrozenInterfaceFluxes) = nothing

function _projected_cell_activity(field::InterfaceField{T}) where {T}
    mesh = field.mesh
    nx, ny = mesh.block_size
    halo = mesh.halo
    active = falses(ncells(mesh))
    volumes = zeros(T, ncells(mesh))
    for (block, leaf) in pairs(mesh.blocks)
        fraction = @inbounds field.conditional_fraction.blocks[block]
        open = isnothing(field.geometry) ? nothing :
            @inbounds(field.geometry.open_fraction.blocks[block])
        offset = (block - 1) * nx * ny
        @inbounds for j in 1:ny, i in 1:nx
            index = offset + (j - 1) * nx + i
            is_active = fraction[i + halo, j + halo] > zero(T)
            active[index] = is_active
            open_fraction = isnothing(open) ? one(T) : open[i + halo, j + halo]
            volumes[index] = is_active ? leaf.volume * open_fraction : zero(T)
        end
    end
    return active, volumes
end

function _projected_face_data!(
        coefficients, fraction_x, fraction_y, field::InterfaceField{T},
        global_to_local_x, global_to_local_y,
    ) where {T}
    _phase_face_average!(
        fraction_x, field.conditional_fraction, field.geometry,
    )
    _phase_face_average!(
        fraction_y, field.conditional_fraction, field.geometry,
    )
    geometry = field.geometry
    for (index, face) in pairs(field.mesh.faces)
        local_index = face.axis == 1 ? global_to_local_x[index] :
            global_to_local_y[index]
        fraction = face.axis == 1 ? fraction_x.interior[local_index] :
            fraction_y.interior[local_index]
        active = fraction > zero(T)
        aperture_squared = isnothing(geometry) ? one(T) :
            geometry.aperture[index]^2
        @inbounds coefficients[index] = active ? aperture_squared : zero(T)
    end
    return coefficients
end

function _projected_operator(
        field::InterfaceField{T}, coefficients, active, volumes,
    ) where {T}
    gradient_operator = isnothing(field.geometry) ?
        GradientOperator(field.mesh, T) : GradientOperator(field.geometry)
    return EllipticOperator(
        field.mesh; reaction = 0, diffusion = 1,
        face_coefficient = coefficients, gradient_operator,
        active_cells = active, volumes,
        inactive_condition = :dirichlet, T,
    )
end

function _interface_method_workspace(
        field::InterfaceField{T, M}, ::ProjectedLiquidVelocity,
    ) where {T, M}
    reconstruct!(field)
    mesh = field.mesh
    fraction_x = _phase_face_average!(
        FaceField{1}(mesh, T), field.conditional_fraction, field.geometry,
    )
    fraction_y = _phase_face_average!(
        FaceField{2}(mesh, T), field.conditional_fraction, field.geometry,
    )
    global_to_local_x, count_x = _component_face_indices(mesh, 1)
    global_to_local_y, count_y = _component_face_indices(mesh, 2)
    coefficients = zeros(T, length(mesh.faces))
    _projected_face_data!(
        coefficients, fraction_x, fraction_y, field,
        global_to_local_x, global_to_local_y,
    )
    active, volumes = _projected_cell_activity(field)
    operator = _projected_operator(field, coefficients, active, volumes)
    solver = EllipticSolver(operator; preconditioner = nothing)
    schedule_x = _dual_advection_schedule(mesh, Val(1), T, false)
    schedule_y = _dual_advection_schedule(mesh, Val(2), T, false)
    return ProjectedLiquidAdvectionWorkspace(
        FaceVelocity(mesh, T), FaceVelocity(mesh, T),
        CellField(mesh, T), CellField(mesh, T), operator, solver,
        EllipticWorkspace(operator), fraction_x, fraction_y,
        schedule_x, schedule_y, global_to_local_x, global_to_local_y,
        zeros(T, count_x), zeros(T, count_y),
        falses(count_x), falses(count_y),
        falses(count_x), falses(count_y),
        zeros(T, count_x), zeros(T, count_y),
        zeros(Int, count_x), zeros(Int, count_y),
        coefficients, active, volumes,
    )
end

function _refresh_projected_operator!(
        state::ProjectedLiquidAdvectionWorkspace{T},
        field::InterfaceField{T},
    ) where {T}
    _projected_face_data!(
        state.face_coefficient, state.fraction_x, state.fraction_y, field,
        state.global_to_local_x, state.global_to_local_y,
    )
    active, volumes = _projected_cell_activity(field)
    singular = !any(!, active)
    singular == state.operator.singular || return false
    copyto!(state.active_cells, active)
    copyto!(state.cell_volumes, volumes)
    _update_elliptic_activity!(state.operator, active, volumes)
    _update_elliptic_face_coefficient!(
        state.operator, state.face_coefficient,
    )
    return true
end

function _extend_projected_component!(
        destination::FaceVelocity{T}, source::FaceVelocity{T}, axis,
        fraction, schedule, global_to_local, values, known, next_known,
        sums, counts, layers,
    ) where {T}
    fill!(values, zero(T))
    fill!(known, false)
    for (face_index, face) in pairs(source.mesh.faces)
        face.axis == axis || continue
        local_index = global_to_local[face_index]
        active = fraction.interior[local_index] > zero(T)
        @inbounds begin
            known[local_index] = active
            values[local_index] = active ? source.interior[face_index] : zero(T)
        end
    end
    for _ in 1:layers
        copyto!(next_known, known)
        fill!(sums, zero(T))
        fill!(counts, 0)
        for connection in schedule.faces
            minus, plus = connection.minus, connection.plus
            if known[minus] && !known[plus]
                @inbounds begin
                    sums[plus] += values[minus]
                    counts[plus] += 1
                end
            elseif known[plus] && !known[minus]
                @inbounds begin
                    sums[minus] += values[plus]
                    counts[minus] += 1
                end
            end
        end
        changed = false
        for index in eachindex(values)
            if !known[index] && counts[index] > 0
                @inbounds begin
                    values[index] = sums[index] / counts[index]
                    next_known[index] = true
                end
                changed = true
            end
        end
        copyto!(known, next_known)
        changed || break
    end
    for (face_index, face) in pairs(source.mesh.faces)
        face.axis == axis || continue
        @inbounds destination.interior[face_index] =
            values[global_to_local[face_index]]
    end
    return destination
end

function _project_liquid_velocity!(
        work::WYAdvectionWorkspace{T}, velocity::TwoPhaseVelocity,
    ) where {T}
    field = work.interface
    method = work.interface_method
    method isa ProjectedLiquidVelocity || throw(
        ArgumentError(
            "the WY workspace does not use projected liquid advection",
        )
    )
    state = work.method_workspace[]
    reconstruct!(field)
    if !_refresh_projected_operator!(state, field)
        state = _interface_method_workspace(field, method)
        work.method_workspace[] = state
    end
    liquid = velocity[1]
    _extend_projected_component!(
        state.projected, liquid, 1, state.fraction_x, state.schedule_x,
        state.global_to_local_x, state.values_x, state.known_x,
        state.next_known_x, state.sums_x, state.counts_x,
        method.extension_layers,
    )
    _extend_projected_component!(
        state.projected, liquid, 2, state.fraction_y, state.schedule_y,
        state.global_to_local_y, state.values_y, state.known_y,
        state.next_known_y, state.sums_y, state.counts_y,
        method.extension_layers,
    )
    fill!(state.projected.boundary, zero(T))
    if !any(state.active_cells)
        fill!(state.correction.interior, zero(T))
        fill!(state.correction.boundary, zero(T))
        diagnostics = work.diagnostics
        diagnostics.projection_initial_divergence = zero(T)
        diagnostics.projection_final_divergence = zero(T)
        diagnostics.projection_iterations = 0
        return state.projected
    end
    isnothing(field.geometry) ?
        divergence!(state.divergence, state.projected) :
        divergence!(state.divergence, state.projected, field.geometry)
    initial = zero(T)
    for (index, ref) in pairs(state.operator.references)
        state.active_cells[index] || continue
        initial = max(initial, abs(_get(state.divergence, ref)))
        _set!(state.divergence, -_get(state.divergence, ref), ref)
    end
    _zero_interior!(state.pressure)
    stats = solve!(
        state.pressure, state.solver, state.divergence;
        rtol = zero(T), atol = T(method.tolerance),
        max_cycles = method.max_cycles,
        compatibility = state.operator.singular ? :project : :error,
    )
    stats.converged || error(
        "projected liquid-velocity extension failed: cycles=$(stats.cycles), " *
            "initial_linf=$(stats.initial_linf), " *
            "final_linf=$(stats.final_linf), target=$(stats.target_linf)",
    )
    if isnothing(field.geometry)
        weighted_gradient!(
            state.correction, state.pressure,
            state.operator, state.elliptic,
        )
    else
        _cut_cell_weighted_gradient!(
            state.correction, state.pressure,
            state.operator, state.elliptic, field.geometry,
        )
    end
    @_parallel for index in eachindex(state.projected.interior)
        @inbounds state.projected.interior[index] -=
            state.correction.interior[index]
    end
    fill!(state.projected.boundary, zero(T))
    isnothing(field.geometry) ?
        divergence!(state.divergence, state.projected) :
        divergence!(state.divergence, state.projected, field.geometry)
    final = zero(T)
    for (index, ref) in pairs(state.operator.references)
        state.active_cells[index] || continue
        final = max(final, abs(_get(state.divergence, ref)))
    end
    diagnostics = work.diagnostics
    diagnostics.projection_initial_divergence = initial
    diagnostics.projection_final_divergence = final
    diagnostics.projection_iterations = stats.cycles
    return state.projected
end

"""Validate the storage contract shared by all two-velocity entry points."""
function _validate_two_phase_velocity(
        velocity::TwoPhaseVelocity, mesh::BlockTreeMesh, ::Type{T},
    ) where {T}
    for (phase, component) in zip((:liquid, :gas), velocity)
        component.mesh === mesh || throw(
            ArgumentError(
                "$phase velocity and the two-phase field must share a mesh",
            )
        )
        eltype(component) === T || throw(
            ArgumentError(
                "$phase velocity has element type $(eltype(component)); expected $T",
            )
        )
        length(component.interior) == length(mesh.faces) || throw(
            DimensionMismatch("$phase velocity has the wrong interior-face layout"),
        )
        length(component.boundary) == length(mesh.boundary_faces) || throw(
            DimensionMismatch("$phase velocity has the wrong boundary-face layout"),
        )
    end
    return velocity
end

"""Set both phase velocities from `(liquid_function, gas_function)`."""
function set_velocities!(
        velocity::TwoPhaseVelocity, functions::Tuple; time = 0,
    )
    length(functions) == 2 || throw(
        ArgumentError(
            "two phase-velocity initializers are required",
        )
    )
    liquid, gas = velocity
    _validate_two_phase_velocity(velocity, liquid.mesh, eltype(liquid))
    set_velocities!(liquid, functions[1]; time)
    set_velocities!(gas, functions[2]; time)
    return velocity
end

"""Apply the impermeable-domain condition to both phase velocities."""
function enforce_no_permeation!(velocity::TwoPhaseVelocity)
    foreach(enforce_no_permeation!, velocity)
    return velocity
end

"""Apply the fixed-solid no-slip mask to both phase velocities."""
function enforce_geometry_no_slip!(
        velocity::TwoPhaseVelocity, geometry::GeometryField,
    )
    _validate_two_phase_velocity(
        velocity, geometry.mesh, eltype(geometry),
    )
    foreach(component -> enforce_geometry_no_slip!(component, geometry), velocity)
    return velocity
end

"""
Register both intrinsic velocity fields for AMR transfer.

The existing face interpolation and divergence-preserving refinement are used
for each phase.  The global face-integral correction is disabled: these fields
store velocity, not phase momentum.
"""
function register_adaptive_field!(
        workspace::AMRWorkspace{T}, velocity::TwoPhaseVelocity,
    ) where {T}
    liquid, gas = velocity
    _validate_two_phase_velocity(
        velocity, workspace.mesh, T,
    )
    for component in (liquid, gas)
        register_adaptive_field!(
            workspace, component;
            piecewise_constant = false, conserve_integral = false,
        )
    end
    return workspace
end

"""
    WYAdvectionWorkspace(interface, velocity::TwoPhaseVelocity; kwargs...)

Construct the existing geometric-advection workspace after validating the
liquid/gas velocity tuple.  Phase flux storage continues to live in the WY
workspace rather than in the velocity fields.
"""
function WYAdvectionWorkspace(
        field::InterfaceField{T}, velocity::TwoPhaseVelocity;
        interface_method = FrozenInterfaceFluxes(), kwargs...,
    ) where {T}
    _validate_two_phase_velocity(velocity, field.mesh, T)
    interface_method isa _TwoVelocityInterfaceAdvection || throw(
        ArgumentError(
            "two-velocity interface_method must be " *
                "ProjectedLiquidVelocity() or FrozenInterfaceFluxes()",
        ),
    )
    return WYAdvectionWorkspace(field; interface_method, kwargs...)
end

"""
Conservative pre-transport CFL bound for a two-velocity tuple.

Until a method-specific interface flux is selected, both phase fields must
individually satisfy the existing multidimensional WY incoming/outgoing-volume
and face-extrusion restrictions.  The minimum of those bounds is therefore a
safe common step.
"""
function stable_timestep(
        velocity::TwoPhaseVelocity, work::WYAdvectionWorkspace{T}; cfl = 0.5,
    ) where {T}
    _validate_two_phase_velocity(velocity, work.mesh, T)
    if work.interface_method isa FrozenInterfaceFluxes
        reconstruct!(work.interface)
        _freeze_phase_apertures!(work.interface, work)
        return _frozen_interface_stable_timestep(
            velocity, work; cfl,
        )
    elseif work.interface_method isa ProjectedLiquidVelocity
        projected = _project_liquid_velocity!(work, velocity)
        return min(
            stable_timestep(projected, work; cfl),
            stable_timestep(velocity[2], work; cfl),
        )
    end
    return min(
        stable_timestep(velocity[1], work; cfl),
        stable_timestep(velocity[2], work; cfl),
    )
end

"""
    advect!(interface, velocity::TwoPhaseVelocity, dt, workspace; kwargs...)

Common tuple entry point. Every velocity tuple uses the explicitly selected
projected-liquid or frozen-interface policy, including tuples whose phase
velocities happen to have equal values. Both policies advance centred phase
scalars with the phase fluxes used by the interface update.
"""
function advect!(
        field::InterfaceField{T}, velocity::TwoPhaseVelocity, dt,
        work::WYAdvectionWorkspace{T}; kwargs...,
    ) where {T}
    _validate_two_phase_velocity(velocity, field.mesh, T)
    momentum = get(kwargs, :momentum, nothing)
    if work.interface_method isa FrozenInterfaceFluxes
        isnothing(momentum) || momentum isa TwoPhaseVelocity || throw(
            ArgumentError(
                "two-phase velocities require a liquid/gas momentum " *
                    "destination tuple",
            ),
        )
        # These are the time-step-start cut faces of the method, not cached
        # geometry from the most recent CFL query.  `advect!` must remain
        # correct when its caller supplies an independently chosen safe step.
        reconstruct!(field)
        _freeze_phase_apertures!(field, work)
        return _advect_wy!(field, velocity, dt, work; kwargs...)
    elseif work.interface_method isa ProjectedLiquidVelocity
        isnothing(momentum) || momentum isa TwoPhaseVelocity || throw(
            ArgumentError(
                "two-phase velocities require a liquid/gas momentum " *
                    "destination tuple",
            ),
        )
        _reset!(work.diagnostics)
        projected = _project_liquid_velocity!(work, velocity)
        limit = min(
            stable_timestep(projected, work),
            stable_timestep(velocity[2], work),
        )
        dt <= limit * (1 + 32eps(T)) || throw(
            ArgumentError(
                "time step exceeds the projected two-velocity WY CFL limit",
            )
        )
        return _advect_wy!(
            field, (projected, velocity[2]), dt, work;
            momentum_source = velocity, reset_diagnostics = false,
            check_timestep = false, kwargs...,
        )
    end
    error("unsupported two-velocity interface-advection policy")
end

function _uses_auxiliary_gas_carrier(
        ::TwoPhaseVelocity, ::WYAdvectionWorkspace{T, M, P},
    ) where {T, M, P <: ProjectedLiquidVelocity}
    return true
end

"""
Populate the projected-liquid physical flux and the auxiliary gas-carrier
flux for one split sweep. The liquid flux is cut from the donating region of
the projected liquid velocity. The gas flux is cut independently from the
donating region of the gas velocity; it therefore need not complement the
liquid flux. `open_flux` is the projected carrier flux used by the physical
liquid WY update.
"""
function _compute_directional_wy_fluxes!(
        field::InterfaceField{T}, velocity::TwoPhaseVelocity, dt,
        work::WYAdvectionWorkspace{T, M, <:ProjectedLiquidVelocity}, axis,
    ) where {T, M}
    mesh = field.mesh
    geometry = field.geometry
    liquid_velocity, gas_velocity = velocity
    @_parallel for face_index in eachindex(mesh.faces)
        face = mesh.faces[face_index]
        face.axis == axis || continue
        aperture = isnothing(geometry) ? one(T) :
            @inbounds(geometry.aperture[face_index])

        liquid_u = @inbounds liquid_velocity.interior[face_index]
        liquid_volume = T(dt) * liquid_u * face.area * aperture
        liquid_flux = zero(T)
        if !iszero(liquid_volume)
            donor = liquid_u >= 0 ? face.minus : face.plus
            liquid_fraction = _swept_fraction(
                field, face, donor, liquid_u, dt,
                abs(liquid_volume) / mesh.blocks[donor.block].volume,
            )
            liquid_flux = liquid_volume * liquid_fraction
        end

        gas_u = @inbounds gas_velocity.interior[face_index]
        gas_volume = T(dt) * gas_u * face.area * aperture
        gas_flux = zero(T)
        if !iszero(gas_volume)
            donor = gas_u >= 0 ? face.minus : face.plus
            liquid_fraction = _swept_fraction(
                field, face, donor, gas_u, dt,
                abs(gas_volume) / mesh.blocks[donor.block].volume,
            )
            gas_flux = gas_volume * (one(T) - liquid_fraction)
        end

        @inbounds begin
            work.open_flux[face_index] = liquid_volume
            work.liquid_flux[face_index] = liquid_flux
            work.gas_flux[face_index] = gas_flux
        end
    end
    return work
end

@inline function _averaged_tangent_normal(field, face, tangent)
    return (
        (
            tangent == 1 ? _get(field.normal_x, face.minus) :
                _get(field.normal_y, face.minus)
        ) +
            (
            tangent == 1 ? _get(field.normal_x, face.plus) :
                _get(field.normal_y, face.plus)
        )
    ) / 2
end

"""Freeze the time-step-start physical phase apertures on every open face."""
function _freeze_phase_apertures!(
        field::InterfaceField{T}, work::WYAdvectionWorkspace{T},
    ) where {T}
    field.mesh === work.mesh || throw(
        ArgumentError(
            "interface and WY workspace must share a mesh",
        )
    )
    geometry = field.geometry
    @_parallel for index in eachindex(field.mesh.faces)
        liquid, gas = _phase_face_apertures(field, geometry, index)
        @inbounds begin
            work.liquid_aperture[index] = liquid
            work.gas_aperture[index] = gas
        end
    end
    return work
end

"""Frozen open/phase subface interval in coordinates of the donor cell."""
function _frozen_subface_interval(
        field::InterfaceField{T}, work::WYAdvectionWorkspace{T},
        face, face_index, donor, liquid_carrier::Bool,
    ) where {T}
    full_lo, full_hi = _face_local_tangent_interval(
        field.mesh, donor, face,
    )
    full_width = full_hi - full_lo
    geometry = field.geometry
    open_aperture = isnothing(geometry) ? one(T) :
        @inbounds(geometry.aperture[face_index])
    open_width = clamp(open_aperture, zero(T), one(T)) * full_width
    open_lo, open_hi = full_lo, full_hi
    if open_width < full_width
        tangent = 3 - face.axis
        solid_tangent = _averaged_tangent_normal(
            geometry.solid, face, tangent,
        )
        if solid_tangent > zero(T)
            open_lo = full_hi - open_width
        else
            open_hi = full_lo + open_width
        end
    end
    phase_aperture = liquid_carrier ?
        @inbounds(work.liquid_aperture[face_index]) :
        @inbounds(work.gas_aperture[face_index])
    phase_width = clamp(phase_aperture, zero(T), open_aperture) *
        full_width
    phase_width > zero(T) || return open_lo, open_lo
    phase_width >= open_width && return open_lo, open_hi

    tangent = 3 - face.axis
    liquid_tangent = _averaged_tangent_normal(field, face, tangent)
    liquid_is_lower = liquid_tangent >= zero(T)
    phase_is_lower = liquid_carrier == liquid_is_lower
    return phase_is_lower ?
        (open_lo, open_lo + phase_width) :
        (open_hi - phase_width, open_hi)
end

"""Liquid fraction of one frozen phase-subface donating region."""
function _frozen_subface_swept_fraction(
        field::InterfaceField{T}, face, donor, u, dt,
        tangent_lo, tangent_hi, target,
    ) where {T}
    target > zero(T) || return zero(T)
    transverse_width = T(tangent_hi - tangent_lo)
    transverse_width > zero(T) || return zero(T)
    mesh = field.mesh
    axis = face.axis
    function swept_bounds(distance)
        normal_lo, normal_hi = u >= 0 ?
            (T(0.5) - distance, T(0.5)) :
            (-T(0.5), -T(0.5) + distance)
        return axis == 1 ?
            (normal_lo, normal_hi, T(tangent_lo), T(tangent_hi)) :
            (T(tangent_lo), T(tangent_hi), normal_lo, normal_hi)
    end

    geometry = field.geometry
    distance = clamp(target / transverse_width, zero(T), one(T))
    if !isnothing(geometry)
        maximum_area = _open_subcell_area(
            geometry.solid, donor, swept_bounds(one(T))...,
        )
        tolerance = T(256) * eps(T) * max(one(T), maximum_area)
        target <= maximum_area + tolerance || error(
            "frozen phase-subface flux exceeds the donor open volume",
        )
        lo, hi = zero(T), one(T)
        for _ in 1:54
            middle = (lo + hi) / 2
            candidate = _open_subcell_area(
                geometry.solid, donor, swept_bounds(middle)...,
            )
            if candidate < target
                lo = middle
            else
                hi = middle
            end
        end
        distance = (lo + hi) / 2
    end
    bounds = swept_bounds(distance)
    xlo, xhi, ylo, yhi = bounds
    rectangle = NTuple{2, T}[
        (xlo, ylo), (xhi, ylo), (xhi, yhi), (xlo, yhi),
    ]
    open_polygon = isnothing(geometry) ? rectangle : _clip_polygon(
            rectangle,
            _get(geometry.solid.normal_x, donor),
            _get(geometry.solid.normal_y, donor),
            _get(geometry.solid.intercept, donor); keep_less = false,
        )
    liquid_polygon = _clip_polygon(
        open_polygon,
        _get(field.normal_x, donor), _get(field.normal_y, donor),
        _get(field.intercept, donor); keep_less = true,
    )
    return clamp(_polygon_area(liquid_polygon) / target, zero(T), one(T))
end

function _compute_directional_wy_fluxes!(
        field::InterfaceField{T}, velocity::TwoPhaseVelocity, dt,
        work::WYAdvectionWorkspace{T, M, <:FrozenInterfaceFluxes}, axis,
    ) where {T, M}
    mesh = field.mesh
    liquid_velocity, gas_velocity = velocity
    @_parallel for face_index in eachindex(mesh.faces)
        face = mesh.faces[face_index]
        face.axis == axis || continue
        open_flux = liquid_flux = zero(T)
        for liquid_carrier in (true, false)
            u = liquid_carrier ?
                liquid_velocity.interior[face_index] :
                gas_velocity.interior[face_index]
            aperture = liquid_carrier ?
                work.liquid_aperture[face_index] :
                work.gas_aperture[face_index]
            (iszero(u) || iszero(aperture)) && continue
            donor = u >= 0 ? face.minus : face.plus
            volume = T(dt) * u * face.area * aperture
            tangent_lo, tangent_hi = _frozen_subface_interval(
                field, work, face, face_index, donor, liquid_carrier,
            )
            fraction = _frozen_subface_swept_fraction(
                field, face, donor, u, dt, tangent_lo, tangent_hi,
                abs(volume) / mesh.blocks[donor.block].volume,
            )
            open_flux += volume
            liquid_flux += volume * fraction
        end
        @inbounds begin
            work.open_flux[face_index] = open_flux
            work.liquid_flux[face_index] = liquid_flux
            work.gas_flux[face_index] = open_flux - liquid_flux
        end
    end
    return work
end

function _frozen_interface_stable_timestep(
        velocity::TwoPhaseVelocity, work::WYAdvectionWorkspace{T};
        cfl = 0.5,
    ) where {T}
    isfinite(cfl) && 0 < cfl <= 0.5 || throw(
        ArgumentError("WY requires 0 < cfl <= 0.5"),
    )
    foreach(component -> _check_wy_workspace(component, work), velocity)
    mesh = work.mesh
    nx, ny = mesh.block_size
    halo = mesh.halo
    cells_per_block = nx * ny
    _parallel_block_columns(mesh) do block, first_j, last_j
        incoming_values = @inbounds work.incoming.blocks[block]
        outgoing_values = @inbounds work.outgoing.blocks[block]
        offset = (block - 1) * cells_per_block
        cell_volume = mesh.blocks[block].volume
        @inbounds for j in first_j:last_j, i in 1:nx
            cell = offset + (j - 1) * nx + i
            incoming = outgoing = zero(T)
            for pointer in work.cell_face_offsets[cell]:(
                    work.cell_face_offsets[cell + 1] - 1
                )
                face_index = work.cell_faces[pointer]
                face = mesh.faces[face_index]
                orientation = work.cell_face_orientations[pointer]
                for phase in 1:2
                    rate = velocity[phase].interior[face_index] * face.area *
                        (
                        phase == 1 ? work.liquid_aperture[face_index] :
                            work.gas_aperture[face_index]
                    )
                    outward_rate = orientation * rate
                    outgoing += max(outward_rate, zero(T))
                    incoming += max(-outward_rate, zero(T))
                end
            end
            open = isnothing(work.geometry) ? one(T) :
                work.geometry.open_fraction.blocks[block][i + halo, j + halo]
            effective_volume = cell_volume * open
            incoming_values[i + halo, j + halo] =
                effective_volume > zero(T) ? incoming / effective_volume : zero(T)
            outgoing_values[i + halo, j + halo] =
                effective_volume > zero(T) ? outgoing / effective_volume : zero(T)
        end
    end
    maxrate = zero(T)
    for (face_index, face) in pairs(mesh.faces), phase in 1:2
        aperture = phase == 1 ? work.liquid_aperture[face_index] :
            work.gas_aperture[face_index]
        aperture > zero(T) || continue
        u = velocity[phase].interior[face_index]
        donor = u >= 0 ? face.minus : face.plus
        maxrate = max(
            maxrate,
            abs(u) / mesh.blocks[donor.block].spacing[face.axis],
        )
    end
    for block in eachindex(mesh.blocks)
        incoming = @inbounds work.incoming.blocks[block]
        outgoing = @inbounds work.outgoing.blocks[block]
        @inbounds for j in 1:ny, i in 1:nx
            maxrate = max(
                maxrate, incoming[i + halo, j + halo],
                outgoing[i + halo, j + halo],
            )
        end
    end
    return iszero(maxrate) ? T(Inf) : T(cfl) / maxrate
end

"""Physical liquid and gas fractions of a full Cartesian cell."""
@inline function _phase_cell_fractions(
        field::InterfaceField{T}, ::Nothing, ref::CellRef,
    ) where {T}
    liquid = _get(field.fraction, ref)
    return liquid, one(T) - liquid
end

@inline function _phase_cell_fractions(
        field::InterfaceField{T}, geometry::GeometryField{T}, ref::CellRef,
    ) where {T}
    liquid = _get(field.fraction, ref)
    open = _get(geometry.open_fraction, ref)
    return liquid, open - liquid
end

@inline _phase_cell_fractions(field::InterfaceField, ref::CellRef) =
    _phase_cell_fractions(field, field.geometry, ref)

@inline _select_phase(values, ::Val{:liquid}) = values[1]
@inline _select_phase(values, ::Val{:gas}) = values[2]

"""Whether a phase occupies a positive physical volume in `ref`."""
@inline function _phase_cell_active(
        field::InterfaceField, geometry, ref::CellRef,
        phase::Union{Val{:liquid}, Val{:gas}},
    )
    return _select_phase(
        _phase_cell_fractions(field, geometry, ref), phase,
    ) > 0
end

@inline function _phase_cell_active(
        field::InterfaceField, ref::CellRef, phase::Union{Val{:liquid}, Val{:gas}},
    )
    return _phase_cell_active(field, field.geometry, ref, phase)
end

@inline function _staggered_open_volume(
        field::InterfaceField, ::Nothing, fraction::FaceField{Axis}, index;
        boundary = false,
    ) where {Axis}
    return one(eltype(field))
end

@inline function _staggered_open_volume(
        field::InterfaceField, geometry::GeometryField,
        fraction::FaceField{Axis}, index; boundary = false,
    ) where {Axis}
    volumes = boundary ? geometry.boundary_dual_volumes[Axis] :
        geometry.dual_volumes[Axis]
    return @inbounds volumes[index]
end

"""
Liquid and gas fractions relative to the open staggered control volume.

`fraction` is the existing geometry-weighted staggered interpolation of the
conditional liquid fraction.  A fully solid dual volume has `(0, 0)` so that
neither velocity is active there.
"""
@inline function _phase_staggered_fractions(
        field::InterfaceField{T}, geometry, fraction::FaceField, index;
        boundary = false,
    ) where {T}
    open_volume = _staggered_open_volume(
        field, geometry, fraction, index; boundary,
    )
    open_volume > zero(T) || return zero(T), zero(T)
    liquid = boundary ? @inbounds(fraction.boundary[index]) :
        @inbounds(fraction.interior[index])
    liquid = clamp(liquid, zero(T), one(T))
    return liquid, one(T) - liquid
end

@inline function _phase_staggered_fractions(
        field::InterfaceField, fraction::FaceField, index;
        boundary = false,
    )
    return _phase_staggered_fractions(
        field, field.geometry, fraction, index; boundary,
    )
end

"""Whether a phase velocity is active on a staggered control volume."""
@inline function _phase_staggered_active(
        field::InterfaceField, geometry, fraction::FaceField, index,
        phase::Union{Val{:liquid}, Val{:gas}}; boundary = false,
    )
    values = _phase_staggered_fractions(
        field, geometry, fraction, index; boundary,
    )
    return _select_phase(values, phase) > 0
end

@inline function _phase_staggered_active(
        field::InterfaceField, fraction::FaceField, index,
        phase::Union{Val{:liquid}, Val{:gas}}; boundary = false,
    )
    return _phase_staggered_active(
        field, field.geometry, fraction, index, phase; boundary,
    )
end

"""Physical phase apertures on one owned interior face."""
@inline function _phase_face_apertures(
        field::InterfaceField{T}, geometry, index,
    ) where {T}
    face = @inbounds field.mesh.faces[index]
    open = isnothing(geometry) ? one(T) :
        @inbounds(geometry.aperture[index])
    liquid = (
        _face_aperture(field, geometry, face, face.minus) +
            _face_aperture(field, geometry, face, face.plus)
    ) / 2
    liquid = clamp(liquid, zero(T), open)
    return liquid, open - liquid
end

@inline _phase_face_apertures(field::InterfaceField, index) =
    _phase_face_apertures(field, field.geometry, index)

"""Physical phase apertures on one domain-boundary face."""
@inline function _phase_boundary_face_apertures(
        field::InterfaceField{T}, geometry, index,
    ) where {T}
    face = @inbounds field.mesh.boundary_faces[index]
    open = isnothing(geometry) ? one(T) :
        @inbounds(geometry.boundary_aperture[index])
    liquid = clamp(
        _face_aperture(field, geometry, face, face.cell), zero(T), open,
    )
    return liquid, open - liquid
end

@inline _phase_boundary_face_apertures(field::InterfaceField, index) =
    _phase_boundary_face_apertures(field, field.geometry, index)

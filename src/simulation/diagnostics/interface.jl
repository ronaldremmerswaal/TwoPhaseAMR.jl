"""Supertype for quantities evaluated from a [`SimulationState`](@ref)."""
abstract type AbstractDiagnostic end

"""
    Diagnostic(name, measure; label, units="", dimension=nothing,
               components=String[], kind=:physical, scale=:linear)

Define a custom diagnostic. `measure(state)` may return any consistently typed
value; scalar and small static-vector diagnostics are both supported. The
keywords form its [`DiagnosticInfo`](@ref); `label` defaults to a name-derived
label.
"""
struct Diagnostic{F} <: AbstractDiagnostic
    name::Symbol
    measure::F
    info::DiagnosticInfo
end

function Diagnostic(
        name::Symbol, measure; label = _default_diagnostic_label(name),
        units = "", dimension = nothing, components = String[],
        kind = :physical, scale = :linear,
    )
    info = DiagnosticInfo(; label, units, dimension, components, kind, scale)
    return Diagnostic(name, measure, info)
end
Diagnostic(measure, name::Symbol; metadata...) =
    Diagnostic(name, measure; metadata...)
Diagnostic(::Symbol, ::Symbol; metadata...) = throw(
    ArgumentError("a diagnostic measure cannot be a Symbol"),
)

"""Return the stable symbol used to identify a diagnostic history."""
diagnostic_name(diagnostic::Diagnostic) = diagnostic.name

"""Evaluate `diagnostic` for one completed [`SimulationState`](@ref)."""
diagnostic_value(diagnostic::Diagnostic, state::SimulationState) =
    diagnostic.measure(state)

"""Total kinetic energy of the configured single- or two-phase fluid."""
struct KineticEnergy <: AbstractDiagnostic end
diagnostic_name(::KineticEnergy) = :kinetic_energy

"""
    PotentialEnergy()

Gravitational potential energy of the configured fluids. The potential is
zero at the coordinate origin. Single-phase simulations currently have no
gravity model and therefore report zero.
"""
struct PotentialEnergy <: AbstractDiagnostic end
diagnostic_name(::PotentialEnergy) = :potential_energy

"""
    SurfaceEnergy()

Surface-tension coefficient times the physical phase-interface measure.
Single-phase simulations report zero.
"""
struct SurfaceEnergy <: AbstractDiagnostic end
diagnostic_name(::SurfaceEnergy) = :surface_energy

"""Sum of kinetic, gravitational-potential, and surface energies."""
struct TotalEnergy <: AbstractDiagnostic end
diagnostic_name(::TotalEnergy) = :total_energy

"""Two-component volume-integrated linear momentum."""
struct LinearMomentum <: AbstractDiagnostic end
diagnostic_name(::LinearMomentum) = :linear_momentum

"""Liquid and gas volumes as a two-component static vector."""
struct PhaseVolume <: AbstractDiagnostic end
diagnostic_name(::PhaseVolume) = :phase_volume

"""Liquid and gas masses as a two-component static vector."""
struct PhaseMass <: AbstractDiagnostic end
diagnostic_name(::PhaseMass) = :phase_mass

"""Physical measure (line length in two dimensions) of the phase interface."""
struct InterfaceMeasure <: AbstractDiagnostic end
diagnostic_name(::InterfaceMeasure) = :interface_measure

"""Maximum absolute component stored on the staggered velocity field(s)."""
struct MaximumVelocity <: AbstractDiagnostic end
diagnostic_name(::MaximumVelocity) = :maximum_velocity

"""
    CFLTimeStep(; cfl=0.45)

Current convective timestep bound computed by the configured flow model.
"""
struct CFLTimeStep{T <: Real} <: AbstractDiagnostic
    cfl::T
    function CFLTimeStep(; cfl::T = 0.45) where {T <: Real}
        isfinite(cfl) && cfl > zero(T) || throw(
            ArgumentError("cfl must be finite and strictly positive"),
        )
        return new{T}(cfl)
    end
end
diagnostic_name(::CFLTimeStep) = :cfl_timestep

"""Volume-weighted L2 norm of the velocity divergence."""
struct DivergenceNorm <: AbstractDiagnostic end
diagnostic_name(::DivergenceNorm) = :divergence_norm

"""Number of leaf blocks in the current mesh."""
struct LeafBlockCount <: AbstractDiagnostic end
diagnostic_name(::LeafBlockCount) = :leaf_blocks

"""Number of active finite-volume cells in the current mesh."""
struct CellCount <: AbstractDiagnostic end
diagnostic_name(::CellCount) = :cells

function _diagnostic_face_volumes(state::SimulationState)
    return isnothing(state.geometry) ?
        state.solver.mesh.face_dual_volumes :
        state.geometry.face_dual_volumes
end

function _one_velocity_kinetic_energy(
        velocity::FaceVelocity{T}, interface::InterfaceField{T},
        geometry, density::NTuple{2, T}, volumes::AbstractVector{T},
    ) where {T}
    result = zero(T)
    for (index, face) in pairs(interface.mesh.faces)
        fraction = _staggered_liquid_fraction(interface, geometry, face)
        rho = density[1] * fraction + density[2] * (one(T) - fraction)
        @inbounds result += rho * abs2(velocity.interior[index]) *
            volumes[index] / 2
    end
    return result
end

function _two_velocity_kinetic_energy(
        velocity, interface::InterfaceField{T}, geometry,
        density::NTuple{2, T}, volumes::AbstractVector{T},
    ) where {T}
    rho_liquid, rho_gas = density
    result = zero(T)
    # Passing the concrete interface and geometry across this function
    # boundary specializes the whole face loop. Accessing them through the
    # partially parameterized solver/model on each face allocated heavily.
    for (index, face) in pairs(interface.mesh.faces)
        fraction = _staggered_liquid_fraction(interface, geometry, face)
        iszero(volumes[index]) && continue
        @inbounds result += volumes[index] * (
            (
                iszero(fraction) ? zero(T) :
                    rho_liquid * fraction * abs2(velocity[1].interior[index])
            ) +
                (
                fraction == one(T) ? zero(T) :
                    rho_gas * (one(T) - fraction) * abs2(velocity[2].interior[index])
            )
        ) / 2
    end
    return result
end

function diagnostic_value(::KineticEnergy, state::SimulationState)
    velocity = state.velocity
    volumes = _diagnostic_face_volumes(state)
    if velocity isa FaceVelocity
        model = state.solver.two_phase
        if isnothing(model)
            result = zero(eltype(velocity))
            for index in eachindex(volumes)
                @inbounds result += state.solver.density *
                    abs2(velocity.interior[index]) * volumes[index] / 2
            end
            return result
        end
        interface = getfield(model, :interface)
        return _one_velocity_kinetic_energy(
            velocity, interface, state.geometry,
            getfield(model, :density), volumes,
        )
    end

    model = state.solver.two_phase
    isnothing(model) && throw(
        ArgumentError("two-velocity kinetic energy requires two-phase physics"),
    )
    return _two_velocity_kinetic_energy(
        velocity, getfield(model, :interface), state.geometry,
        getfield(model, :density), volumes,
    )
end

function diagnostic_value(::PotentialEnergy, state::SimulationState)
    model = state.solver.two_phase
    if isnothing(model)
        return zero(eltype(state.pressure))
    end
    return _gravitational_potential_energy(
        model.interface, state.geometry;
        density = model.density, gravity = model.gravity,
    )
end

function diagnostic_value(::SurfaceEnergy, state::SimulationState)
    model = state.solver.two_phase
    if isnothing(model)
        return zero(eltype(state.pressure))
    end
    iszero(model.sigma) && return zero(model.sigma)
    return model.sigma * diagnostic_value(InterfaceMeasure(), state)
end

diagnostic_value(::TotalEnergy, state::SimulationState) =
    diagnostic_value(KineticEnergy(), state) +
    diagnostic_value(PotentialEnergy(), state) +
    diagnostic_value(SurfaceEnergy(), state)

function _two_velocity_linear_momentum(
        velocity, interface::InterfaceField{T}, geometry,
        density::NTuple{2, T}, volumes::AbstractVector{T},
    ) where {T}
    rho_liquid, rho_gas = density
    x = y = zero(T)
    for (index, face) in pairs(interface.mesh.faces)
        fraction = _staggered_liquid_fraction(interface, geometry, face)
        iszero(volumes[index]) && continue
        @inbounds contribution = volumes[index] * (
            _phase_product(rho_liquid * fraction, velocity[1].interior[index]) +
                _phase_product(
                rho_gas * (one(T) - fraction),
                velocity[2].interior[index]
            )
        )
        if face.axis == 1
            x += contribution
        else
            y += contribution
        end
    end
    return SVector(x, y)
end

function _one_velocity_linear_momentum(
        velocity::FaceVelocity{T}, interface::InterfaceField{T},
        geometry, density::NTuple{2, T}, volumes::AbstractVector{T},
    ) where {T}
    x = y = zero(T)
    for (index, face) in pairs(interface.mesh.faces)
        fraction = _staggered_liquid_fraction(interface, geometry, face)
        rho = density[1] * fraction + density[2] * (one(T) - fraction)
        @inbounds contribution =
            rho * velocity.interior[index] * volumes[index]
        if face.axis == 1
            x += contribution
        else
            y += contribution
        end
    end
    return SVector(x, y)
end

function diagnostic_value(::LinearMomentum, state::SimulationState)
    velocity = state.velocity
    volumes = _diagnostic_face_volumes(state)
    if velocity isa FaceVelocity
        T = eltype(velocity)
        model = state.solver.two_phase
        if !isnothing(model)
            interface = getfield(model, :interface)
            return _one_velocity_linear_momentum(
                velocity, interface, state.geometry,
                getfield(model, :density), volumes,
            )
        end
        x = y = zero(T)
        for (index, face) in pairs(state.solver.mesh.faces)
            @inbounds contribution = state.solver.density *
                velocity.interior[index] * volumes[index]
            if face.axis == 1
                x += contribution
            else
                y += contribution
            end
        end
        return SVector(x, y)
    end
    model = state.solver.two_phase
    isnothing(model) && throw(
        ArgumentError("two-velocity momentum requires two-phase physics"),
    )
    return _two_velocity_linear_momentum(
        velocity, getfield(model, :interface), state.geometry,
        getfield(model, :density), volumes,
    )
end

function diagnostic_value(::PhaseVolume, state::SimulationState)
    interface = state.interface
    isnothing(interface) && throw(
        ArgumentError("phase volume is only defined for two-phase simulations"),
    )
    liquid = integrate(interface)
    open_volume = isnothing(state.geometry) ?
        prod(interface.mesh.extent) : integrate(state.geometry.open_fraction)
    return SVector(liquid, open_volume - liquid)
end

function diagnostic_value(::PhaseMass, state::SimulationState)
    model = state.solver.two_phase
    isnothing(model) && throw(
        ArgumentError("phase mass is only defined for two-phase simulations"),
    )
    volume = diagnostic_value(PhaseVolume(), state)
    return SVector(
        model.density[1] * volume[1], model.density[2] * volume[2],
    )
end

function diagnostic_value(::InterfaceMeasure, state::SimulationState)
    interface = state.interface
    isnothing(interface) && throw(
        ArgumentError(
            "interface measure is only defined for two-phase simulations",
        ),
    )
    T = eltype(interface)
    measure = zero(T)
    for (first_point, second_point) in plic_segments(interface)
        measure += hypot(
            second_point[1] - first_point[1],
            second_point[2] - first_point[2],
        )
    end
    # A PLIC segment exists only in a mixed cell. An interface that lies
    # exactly on a mesh face instead separates two pure cells and would
    # otherwise have zero reported measure.
    floor = interface.fraction_floor
    geometry = state.geometry
    for (index, face) in pairs(interface.mesh.faces)
        minus = _conditional_fraction(interface, face.minus)
        plus = _conditional_fraction(interface, face.plus)
        separated = (minus <= floor && plus >= one(T) - floor) ||
            (plus <= floor && minus >= one(T) - floor)
        separated || continue
        aperture = isnothing(geometry) ? one(T) : geometry.aperture[index]
        measure += face.area * aperture
    end
    return measure
end

function _maximum_face_velocity(velocity::FaceVelocity)
    maximum_value = zero(eltype(velocity))
    for value in velocity.interior
        maximum_value = max(maximum_value, abs(value))
    end
    for value in velocity.boundary
        maximum_value = max(maximum_value, abs(value))
    end
    return maximum_value
end

function _maximum_two_phase_velocity(velocity, interface::InterfaceField)
    geometry = interface.geometry
    maximum_value = zero(eltype(interface))
    for (phase, name) in ((1, :liquid), (2, :gas))
        for (index, face) in pairs(interface.mesh.faces)
            _phase_face_is_active(interface, geometry, index, face, name) || continue
            maximum_value = max(maximum_value, abs(velocity[phase].interior[index]))
        end
        for (index, face) in pairs(interface.mesh.boundary_faces)
            _phase_cell_is_active(interface, geometry, face.cell, name) || continue
            maximum_value = max(maximum_value, abs(velocity[phase].boundary[index]))
        end
    end
    return maximum_value
end

diagnostic_value(::MaximumVelocity, state::SimulationState) =
    state.velocity isa FaceVelocity ?
    _maximum_face_velocity(state.velocity) :
    _maximum_two_phase_velocity(state.velocity, getfield(state.solver.two_phase, :interface))

function diagnostic_value(diagnostic::CFLTimeStep, state::SimulationState)
    isnothing(state.workspace) && throw(
        ArgumentError("CFLTimeStep requires a simulation workspace"),
    )
    return stable_timestep(
        state.velocity, state.solver, state.workspace; cfl = diagnostic.cfl,
    )
end

function _diagnostic_mixture_velocity!(
        destination::FaceVelocity{T}, source::TwoPhaseVelocity,
        interface::InterfaceField{T}, geometry, density::NTuple{2, T},
    ) where {T}
    destination.mesh === interface.mesh || throw(
        ArgumentError("diagnostic mixture velocity has the wrong mesh"),
    )
    rho_liquid, rho_gas = density
    for (index, face) in pairs(interface.mesh.faces)
        if !isnothing(geometry) && iszero(geometry.face_dual_volumes[index])
            destination.interior[index] = zero(T)
            continue
        end
        fraction = _staggered_liquid_fraction(interface, geometry, face)
        liquid_mass = rho_liquid * fraction
        gas_mass = rho_gas * (one(T) - fraction)
        total_mass = liquid_mass + gas_mass
        @inbounds destination.interior[index] = (
            _phase_product(liquid_mass, source[1].interior[index]) +
                _phase_product(gas_mass, source[2].interior[index])
        ) / total_mass
    end
    fill!(destination.boundary, zero(T))
    return destination
end

function _diagnostic_divergence_velocity(state::SimulationState)
    state.velocity isa FaceVelocity && return state.velocity
    model = state.solver.two_phase
    T = eltype(state.velocity[1])
    # Divergence needs a complete mixture face field, not a fraction array.
    # Reuse the projection's face scratch inside a FlowSimulation. The
    # staggered phase fraction and mixture density are used immediately at
    # each face; neither needs a stored vector for this diagnostic.
    velocity = state.workspace isa NavierStokesWorkspace ?
        state.workspace.projection.weighted_gradient :
        FaceVelocity(state.solver.mesh, T)
    return _diagnostic_mixture_velocity!(
        velocity, state.velocity, getfield(model, :interface),
        state.geometry, getfield(model, :density),
    )
end

function diagnostic_value(::DivergenceNorm, state::SimulationState)
    velocity = _diagnostic_divergence_velocity(state)
    divergence = state.workspace isa NavierStokesWorkspace ?
        state.workspace.projection.divergence :
        CellField(state.solver.mesh, eltype(velocity))
    isnothing(state.geometry) ?
        divergence!(divergence, velocity) :
        divergence!(divergence, velocity, state.geometry)
    result = zero(eltype(velocity))
    nx, ny = state.solver.mesh.block_size
    h = state.solver.mesh.halo
    for (block_index, block) in pairs(state.solver.mesh.blocks)
        values = divergence.blocks[block_index]
        open = isnothing(state.geometry) ? nothing :
            state.geometry.open_fraction.blocks[block_index]
        @inbounds for j in 1:ny, i in 1:nx
            fraction = isnothing(open) ? one(eltype(velocity)) :
                open[i + h, j + h]
            result += abs2(values[i + h, j + h]) * block.volume * fraction
        end
    end
    return sqrt(result)
end

diagnostic_value(::LeafBlockCount, state::SimulationState) =
    nblocks(state.solver.mesh)
diagnostic_value(::CellCount, state::SimulationState) =
    ncells(state.solver.mesh)

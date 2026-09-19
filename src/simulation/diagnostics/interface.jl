"""Supertype for quantities evaluated from a [`SimulationState`](@ref)."""
abstract type AbstractDiagnostic end

"""
    Diagnostic(name, measure)

Define a custom diagnostic. `measure(state)` may return any consistently typed
value; scalar and small static-vector diagnostics are both supported.
"""
struct Diagnostic{F} <: AbstractDiagnostic
    name::Symbol
    measure::F
end

Diagnostic(measure, name::Symbol) = Diagnostic(name, measure)

"""Return the stable symbol used to identify a diagnostic history."""
diagnostic_name(diagnostic::Diagnostic) = diagnostic.name

"""Evaluate `diagnostic` for one completed [`SimulationState`](@ref)."""
diagnostic_value(diagnostic::Diagnostic, state::SimulationState) =
    diagnostic.measure(state)

"""Total kinetic energy of the configured single- or two-phase fluid."""
struct KineticEnergy <: AbstractDiagnostic end
diagnostic_name(::KineticEnergy) = :kinetic_energy

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
    mesh = state.solver.mesh
    T = eltype(mesh.origin)
    return isnothing(state.geometry) ?
        _uncut_face_dual_volumes(mesh, T) :
        _component_volumes_by_global_face(
            mesh, state.geometry.dual_volumes, T;
            covered_value = zero(T),
        )
end

function _one_velocity_density(state::SimulationState)
    model = state.solver.two_phase
    return isnothing(model) ? state.solver.density :
        mixture_density(model.interface, model.density)
end

function diagnostic_value(::KineticEnergy, state::SimulationState)
    velocity = state.velocity
    volumes = _diagnostic_face_volumes(state)
    if velocity isa FaceVelocity
        density = _one_velocity_density(state)
        result = zero(eltype(velocity))
        for index in eachindex(volumes)
            rho = density isa Number ? density : density[index]
            @inbounds result += rho * abs2(velocity.interior[index]) *
                volumes[index] / 2
        end
        return result
    end

    model = state.solver.two_phase
    isnothing(model) && throw(
        ArgumentError("two-velocity kinetic energy requires two-phase physics"),
    )
    rho_liquid, rho_gas = model.density
    result = zero(eltype(velocity[1]))
    for (index, face) in pairs(model.interface.mesh.faces)
        fraction = _staggered_liquid_fraction(
            model.interface, state.geometry, face,
        )
        @inbounds result += volumes[index] * (
            rho_liquid * fraction * abs2(velocity[1].interior[index]) +
                rho_gas * (one(fraction) - fraction) *
                abs2(velocity[2].interior[index])
        ) / 2
    end
    return result
end

function diagnostic_value(::LinearMomentum, state::SimulationState)
    velocity = state.velocity
    volumes = _diagnostic_face_volumes(state)
    T = velocity isa FaceVelocity ? eltype(velocity) : eltype(velocity[1])
    result = zeros(T, 2)
    if velocity isa FaceVelocity
        density = _one_velocity_density(state)
        for (index, face) in pairs(state.solver.mesh.faces)
            rho = density isa Number ? density : density[index]
            @inbounds result[face.axis] +=
                rho * velocity.interior[index] * volumes[index]
        end
    else
        model = state.solver.two_phase
        isnothing(model) && throw(
            ArgumentError("two-velocity momentum requires two-phase physics"),
        )
        rho_liquid, rho_gas = model.density
        for (index, face) in pairs(model.interface.mesh.faces)
            fraction = _staggered_liquid_fraction(
                model.interface, state.geometry, face,
            )
            @inbounds result[face.axis] += volumes[index] * (
                rho_liquid * fraction * velocity[1].interior[index] +
                    rho_gas * (one(fraction) - fraction) *
                    velocity[2].interior[index]
            )
        end
    end
    return SVector(result[1], result[2])
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

diagnostic_value(::MaximumVelocity, state::SimulationState) =
    state.velocity isa FaceVelocity ?
    _maximum_face_velocity(state.velocity) :
    max(
        _maximum_face_velocity(state.velocity[1]),
        _maximum_face_velocity(state.velocity[2]),
    )

function diagnostic_value(diagnostic::CFLTimeStep, state::SimulationState)
    isnothing(state.workspace) && throw(
        ArgumentError("CFLTimeStep requires a simulation workspace"),
    )
    return stable_timestep(
        state.velocity, state.solver, state.workspace; cfl = diagnostic.cfl,
    )
end

function _diagnostic_divergence_velocity(state::SimulationState)
    state.velocity isa FaceVelocity && return state.velocity
    model = state.solver.two_phase
    T = eltype(state.velocity[1])
    velocity = FaceVelocity(state.solver.mesh, T)
    density = Vector{T}(undef, length(state.solver.mesh.faces))
    return _mass_weighted_one_velocity!(
        velocity, density, state.velocity, model, state.geometry,
    )
end

function diagnostic_value(::DivergenceNorm, state::SimulationState)
    velocity = _diagnostic_divergence_velocity(state)
    divergence = CellField(state.solver.mesh, eltype(velocity))
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

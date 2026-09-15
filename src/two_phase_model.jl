struct TwoPhaseModel{T, M}
    interface::InterfaceField{T, M}
    density::NTuple{2, T}
    viscosity::NTuple{2, T}
    sigma::T
    gravity::NTuple{2, T}
    formulation::Union{CSF, GFM}
    scalars::Tuple
end
function TwoPhaseModel(
        interface::InterfaceField{T, M}; density, viscosity,
        surface_tension, gravity, formulation, scalars
    ) where {T, M}
    density isa NTuple{2} || throw(ArgumentError("density must be a (liquid, gas) tuple"))
    mu = viscosity isa Number ? (viscosity, viscosity) : viscosity
    mu isa NTuple{2} || throw(ArgumentError("two-phase viscosity must be a dynamic-viscosity pair or scalar"))
    all(rho -> isfinite(rho)&&rho > 0, density) || throw(ArgumentError("phase densities must be finite and positive"))
    all(v -> isfinite(v)&&v >= 0, mu) && (all(iszero, mu)||all(>(0), mu)) ||
        throw(ArgumentError("dynamic viscosities must both be positive or both zero"))
    isfinite(surface_tension)&&surface_tension >= 0 || throw(ArgumentError("surface_tension must be finite and nonnegative"))
    gravity isa NTuple{2} && all(isfinite, gravity) || throw(ArgumentError("gravity must be a finite two-tuple"))
    formulation isa Union{CSF, GFM} || throw(ArgumentError("surface_tension_formulation must be CSF() or GFM()"))
    scalars isa Tuple && all(q -> q isa PhaseScalar && q.interface === interface, scalars) ||
        throw(ArgumentError("phase_scalars must be a tuple of PhaseScalars belonging to the interface"))
    any(a -> interface.mesh.periodic[a]&&!iszero(gravity[a]), 1:2) &&
        throw(ArgumentError("gravity requires nonperiodic boundaries in its direction"))
    return TwoPhaseModel{T, M}(interface, T.(density), T.(mu), T(surface_tension), T.(gravity), formulation, scalars)
end
mutable struct TwoPhaseFlowWorkspace{T, M, D}
    advection::WYAdvectionWorkspace{T, M}
    momentum::MomentumAdvectionWorkspace{T, M}
    acceleration::FaceVelocity{T, M}
    diffusion::D
    forward::Bool
end
function TwoPhaseFlowWorkspace(
        model::TwoPhaseModel{T, M}; boundary = NoSlip()
    ) where {T, M}
    field = model.interface
    diffusion = all(iszero, model.viscosity) ? nothing :
        StaggeredDiffusionOperator(
            field; viscosity = model.viscosity, boundary
        )
    return TwoPhaseFlowWorkspace{T, M, typeof(diffusion)}(
        WYAdvectionWorkspace(field),
        MomentumAdvectionWorkspace(field), FaceVelocity(field.mesh, T),
        diffusion, true
    )
end

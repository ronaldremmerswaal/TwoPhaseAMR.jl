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
mutable struct TwoPhaseFlowWorkspace{T, M, D, S}
    advection::WYAdvectionWorkspace{T, M}
    momentum::MomentumAdvectionWorkspace{T, M}
    acceleration::FaceVelocity{T, M}
    diffusion::D
    diffusion_solver::S
    density::Vector{T}
    forward::Bool
end
function TwoPhaseFlowWorkspace(
        model::TwoPhaseModel{T, M}; boundary = NoSlip(), dt = nothing,
        diffusion_krylov = Krylov.FgmresWorkspace,
        diffusion_krylov_options = (;),
        diffusion_preconditioner = :geometric,
        diffusion_preconditioner_options = (;),
        coarse_size = 64, pre_sweeps = 2, post_sweeps = 2,
        max_levels = 32
    ) where {T, M}
    field = model.interface
    diffusion = all(iszero, model.viscosity) ? nothing :
        StaggeredDiffusionOperator(
            field; viscosity = model.viscosity, boundary
        )
    density = mixture_density(field, model.density)
    diffusion_solver = if isnothing(diffusion)
        nothing
    else
        # Only the concrete solver/preconditioner types are needed here. A
        # tiny positive seed avoids constructing a diffusion-dominated
        # smoothed prolongation before adaptive time-step selection supplies
        # the physical step size.
        initial_dt = isnothing(dt) ? sqrt(eps(T)) : T(dt)
        VariableStaggeredDiffusionSolver(
            diffusion, density, initial_dt;
            krylov = diffusion_krylov,
            krylov_options = diffusion_krylov_options,
            preconditioner = diffusion_preconditioner,
            preconditioner_options = diffusion_preconditioner_options,
            coarse_size, pre_sweeps, post_sweeps, max_levels
        )
    end
    return TwoPhaseFlowWorkspace{
        T, M, typeof(diffusion), typeof(diffusion_solver),
    }(
        WYAdvectionWorkspace(field),
        MomentumAdvectionWorkspace(field), FaceVelocity(field.mesh, T),
        diffusion, diffusion_solver, density, true
    )
end

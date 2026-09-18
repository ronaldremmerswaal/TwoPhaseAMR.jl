struct TwoPhaseModel{T, M, I}
    interface::InterfaceField{T, M}
    density::NTuple{2, T}
    dynamic_viscosity::NTuple{2, T}
    sigma::T
    gravity::NTuple{2, T}
    formulation::Union{CSF, GFM}
    scalars::Tuple
    interface_method::I
end
function TwoPhaseModel(
        interface::InterfaceField{T, M}; density, dynamic_viscosity,
        surface_tension, gravity, formulation, scalars,
        interface_method = FrozenInterfaceFluxes(),
    ) where {T, M}
    density isa NTuple{2} || throw(ArgumentError("density must be a (liquid, gas) tuple"))
    mu = dynamic_viscosity isa Number ?
        (dynamic_viscosity, dynamic_viscosity) : dynamic_viscosity
    mu isa NTuple{2} || throw(
        ArgumentError(
            "dynamic_viscosity must be a (liquid, gas) pair or scalar",
        ),
    )
    all(rho -> isfinite(rho)&&rho > 0, density) || throw(ArgumentError("phase densities must be finite and positive"))
    all(v -> isfinite(v)&&v >= 0, mu) && (all(iszero, mu)||all(>(0), mu)) ||
        throw(ArgumentError("dynamic viscosities must both be positive or both zero"))
    isfinite(surface_tension)&&surface_tension >= 0 || throw(ArgumentError("surface_tension must be finite and nonnegative"))
    gravity isa NTuple{2} && all(isfinite, gravity) || throw(ArgumentError("gravity must be a finite two-tuple"))
    formulation isa Union{CSF, GFM} || throw(ArgumentError("surface_tension_formulation must be CSF() or GFM()"))
    scalars isa Tuple && all(q -> q isa PhaseScalar && q.interface === interface, scalars) ||
        throw(ArgumentError("phase_scalars must be a tuple of PhaseScalars belonging to the interface"))
    interface_method isa _TwoVelocityInterfaceAdvection || throw(
        ArgumentError(
            "interface_method must be ProjectedLiquidVelocity() or " *
                "FrozenInterfaceFluxes()",
        ),
    )
    any(a -> interface.mesh.periodic[a]&&!iszero(gravity[a]), 1:2) &&
        throw(ArgumentError("gravity requires nonperiodic boundaries in its direction"))
    return TwoPhaseModel{T, M, typeof(interface_method)}(
        interface, T.(density), T.(mu), T(surface_tension), T.(gravity),
        formulation, scalars, interface_method,
    )
end
mutable struct TwoVelocityFlowWorkspace{A, M, V, F, S, W}
    advection::A
    momentum::M
    old_velocity::V
    convected::V
    mixture_velocity::F
    diffused_mixture::F
    mimetic_solver::S
    mimetic_projection::W
end

mutable struct TwoPhaseFlowWorkspace{T, M, D, S, C}
    advection::WYAdvectionWorkspace{T, M}
    momentum::MomentumAdvectionWorkspace{T, M}
    acceleration::FaceVelocity{T, M}
    diffusion::D
    diffusion_solver::S
    density::Vector{T}
    diffusion_cache::C
    forward::Bool
    two_velocity::Base.RefValue{Any}
end

function _two_velocity_flow_workspace(model::TwoPhaseModel{T}) where {T}
    field = model.interface
    old_velocity = (
        FaceVelocity(field.mesh, T), FaceVelocity(field.mesh, T),
    )
    convected = (
        FaceVelocity(field.mesh, T), FaceVelocity(field.mesh, T),
    )
    mixture_velocity = FaceVelocity(field.mesh, T)
    diffused_mixture = FaceVelocity(field.mesh, T)
    advection = WYAdvectionWorkspace(
        field, old_velocity; interface_method = model.interface_method,
    )
    momentum = MomentumAdvectionWorkspace(field, old_velocity)
    solver = MimeticProjectionSolver(
        field; density = model.density, preconditioner = :cholesky,
    )
    return TwoVelocityFlowWorkspace(
        advection, momentum, old_velocity, convected,
        mixture_velocity, diffused_mixture, solver,
        MimeticProjectionWorkspace(solver),
    )
end

function _two_velocity_flow_workspace!(workspace, model)
    state = workspace.two_velocity[]
    if isnothing(state)
        state = _two_velocity_flow_workspace(model)
        workspace.two_velocity[] = state
    end
    return state
end
function TwoPhaseFlowWorkspace(
        model::TwoPhaseModel{T, M}; boundary = NoSlip(), dt = nothing,
        diffusion_krylov = Krylov.CgWorkspace,
        diffusion_krylov_options = (;),
        diffusion_preconditioner = :tree_geometric,
        diffusion_preconditioner_options = (;),
        coarse_size = 64, pre_sweeps = 2, post_sweeps = 2,
        max_levels = 32,
        diffusion_cache = VariableDiffusionConstructionCache(T),
        reuse_diffusion = nothing,
        reuse_diffusion_preconditioner = nothing,
        reuse_diffusion_solver = nothing,
        reuse_density = nothing,
        reuse_workspace = nothing,
    ) where {T, M}
    field = model.interface
    diffusion = all(iszero, model.dynamic_viscosity) ? nothing :
        _staggered_diffusion_operator(
            field; viscosity = model.dynamic_viscosity, boundary,
            construction_cache = diffusion_cache,
            reuse = reuse_diffusion,
        )
    density = if reuse_density isa Vector{T}
        resize!(reuse_density, length(field.mesh.faces))
        _mixture_density!(reuse_density, field, model.density)
    else
        mixture_density(field, model.density)
    end
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
            coarse_size, pre_sweeps, post_sweeps, max_levels,
            reuse_preconditioner = reuse_diffusion_preconditioner,
            reuse_solver = reuse_diffusion_solver,
        )
    end
    advection = isnothing(reuse_workspace) ? WYAdvectionWorkspace(field) :
        _reconfigure_wy_workspace!(reuse_workspace.advection, field)
    momentum = isnothing(reuse_workspace) ? MomentumAdvectionWorkspace(field) :
        _reconfigure_momentum_workspace!(reuse_workspace.momentum, field)
    acceleration = if isnothing(reuse_workspace)
        FaceVelocity(field.mesh, T)
    else
        _resize_zero_velocity!(reuse_workspace.acceleration, field.mesh)
    end
    return TwoPhaseFlowWorkspace{
        T, M, typeof(diffusion), typeof(diffusion_solver),
        typeof(diffusion_cache),
    }(
        advection, momentum, acceleration,
        diffusion, diffusion_solver, density, diffusion_cache, true,
        Ref{Any}(nothing),
    )
end

struct TwoPhaseModel{
        T, M, F <: AbstractSurfaceTensionMethod,
        I <: AbstractInterfaceTransportMethod,
        P <: AbstractTwoVelocityProjection, G <: InterfaceField{T, M}, Q <: Tuple,
    }
    interface::G
    density::NTuple{2, T}
    dynamic_viscosity::NTuple{2, T}
    sigma::T
    gravity::NTuple{2, T}
    formulation::F
    scalars::Q
    interface_transport::I
    pressure_projection::P
end
function TwoPhaseModel(
        physics::TwoPhase{<:InterfaceField{T, M}};
        interface_transport = FrozenInterfaceFluxes(),
        pressure_projection = CMDGFMProjection(),
    ) where {T, M}
    interface = physics.interface
    interface_transport isa _TwoVelocityInterfaceAdvection || throw(
        ArgumentError(
            "interface_transport must be ProjectedLiquidVelocity() or " *
                "FrozenInterfaceFluxes()",
        ),
    )
    pressure_projection isa AbstractTwoVelocityProjection || throw(
        ArgumentError(
            "pressure_projection must be a CMDGFMProjection()",
        ),
    )
    any(a -> interface.mesh.periodic[a] && !iszero(physics.gravity[a]), 1:2) &&
        throw(ArgumentError("gravity requires nonperiodic boundaries in its direction"))
    tension = physics.surface_tension
    return TwoPhaseModel{
        T, M, typeof(tension.method), typeof(interface_transport),
        typeof(pressure_projection), typeof(interface), typeof(physics.scalars),
    }(
        interface, T.(physics.density), T.(physics.dynamic_viscosity),
        T(tension.coefficient), T.(physics.gravity), tension.method,
        physics.scalars, interface_transport, pressure_projection,
    )
end


mutable struct TwoPhaseFlowWorkspace{T, M, D, S, C, A, N, V}
    advection::A
    momentum::N
    acceleration::FaceVelocity{T, M}
    diffusion::D
    diffusion_solver::S
    density::Vector{T}
    diffusion_cache::C
    forward::Bool
    two_velocity::Base.RefValue{Union{Nothing, V}}

    function TwoPhaseFlowWorkspace{T, M, D, S, C, A, N, V}(
            advection, momentum, acceleration, diffusion, diffusion_solver,
            density, diffusion_cache, forward, two_velocity,
        ) where {T, M, D, S, C, A, N, V}
        return new{T, M, D, S, C, A, N, V}(
            advection, momentum, acceleration, diffusion, diffusion_solver,
            density, diffusion_cache, forward, two_velocity,
        )
    end
end

# The intrinsic workspace is allocated lazily. Its transport schedule has the
# same type as the already-constructed shared-velocity momentum schedule.
# Regridding rebuilds this enclosing workspace with the new schedule types.
function _two_velocity_workspace_type(
        model::TwoPhaseModel{T, M}, advection::A,
        momentum::MomentumAdvectionWorkspace{T, M, WX, WY},
    ) where {T, M, A, WX, WY}
    F = typeof(model.interface)
    X = PhaseFaceScalar{1, T, M, PhaseFaceAdvectionScratch{T}, WX, F}
    Y = PhaseFaceScalar{2, T, M, PhaseFaceAdvectionScratch{T}, WY, F}
    N = TwoVelocityMomentumAdvectionWorkspace{T, M, X, Y, typeof(model.interface.geometry)}
    V = FaceVelocity{T, M}
    projection = model.pressure_projection
    S = CMDGFMProjectionSolver{
        T, F, typeof(projection.preconditioner),
        typeof(projection.preconditioner_options),
    }
    return TwoVelocityFlowWorkspace{A, N, Tuple{V, V}, V, S, CMDGFMProjectionWorkspace{T, M}}
end


function TwoPhaseFlowWorkspace(
        model::TwoPhaseModel{T, M}; boundary = NoSlip(), dt = nothing,
        diffusion_krylov = Krylov.CgWorkspace,
        diffusion_krylov_options = (;),
        diffusion_preconditioner = :tree_geometric,
        diffusion_preconditioner_options = (;),
        multigrid::MultigridOptions = MultigridOptions(),
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
            multigrid,
            reuse_preconditioner = reuse_diffusion_preconditioner,
            reuse_solver = reuse_diffusion_solver,
        )
    end
    # The same WY scratch serves the shared- and intrinsic-velocity paths.
    # Its policy is relevant only to the intrinsic path; a single-velocity
    # `advect!` dispatches directly to the ordinary WY sweep.
    advection = isnothing(reuse_workspace) ? WYAdvectionWorkspace(
            field; interface_transport = model.interface_transport,
        ) :
        _reconfigure_wy_workspace!(reuse_workspace.advection, field)
    momentum = isnothing(reuse_workspace) ? MomentumAdvectionWorkspace(field) :
        _reconfigure_momentum_workspace!(reuse_workspace.momentum, field)
    acceleration = if isnothing(reuse_workspace)
        FaceVelocity(field.mesh, T)
    else
        _resize_zero_velocity!(reuse_workspace.acceleration, field.mesh)
    end
    velocity_type = _two_velocity_workspace_type(model, advection, momentum)
    velocity_state = isnothing(reuse_workspace) ? nothing :
        _reconfigure_two_velocity_flow_workspace(
            reuse_workspace.two_velocity[], model, advection,
        )
    return TwoPhaseFlowWorkspace{
        T, M, typeof(diffusion), typeof(diffusion_solver),
        typeof(diffusion_cache), typeof(advection), typeof(momentum), velocity_type,
    }(
        advection, momentum, acceleration,
        diffusion, diffusion_solver, density, diffusion_cache, true,
        Ref{Union{Nothing, velocity_type}}(velocity_state),
    )
end

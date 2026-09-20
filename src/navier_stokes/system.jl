"""
    NavierStokesSystem(mesh; density, dynamic_viscosity,
        boundary=NoSlip(), scheme=LW(),
        amr_interface_scheme=default_amr_interface_scheme(scheme),
        pressure_krylov=nothing,
        diffusion_krylov=nothing,
        pressure_krylov_options=(;), diffusion_krylov_options=(;),
        pressure_preconditioner=:auto,
        diffusion_preconditioner=:auto,
        two_velocity_projection=CMDGFMProjection(),
        refinement_indicators=(), adaptive_fields=())

Incompressible Navier--Stokes discretization. Supplying `interface` selects
the one-velocity two-phase coefficients and compatible geometric transport;
without it the original single-phase operators are used. Both configurations
run the same staged `step!` procedure. One time step is split into:

1. a conservative staggered convection update (Lax--Wendroff by default and
   optionally using a three-point limited AMR-interface flux), applied as
   sequential coordinate sweeps whose order alternates between time steps,
2. the optional user body force followed by a Crank--Nicolson/implicit-midpoint
   staggered viscous solve,
3. optional phase forces (surface tension and gravity), and
4. one compatible pressure projection.

`density` and `dynamic_viscosity` are material properties in both modes. For
single-phase flow they are scalars; the diffusion operator uses the derived
kinematic viscosity `dynamic_viscosity / density`, while the pressure
projection uses the inverse density so that the stored pressure has physical
units. For two phases they are `(liquid, gas)` pairs (a scalar dynamic
viscosity is also accepted when both phases match). The coupled diffusion
operator evaluates the volume-fraction-weighted geometric mean of dynamic
viscosity at each normal- and shear-stress location.

The tree may contain static or dynamically adapted local refinement. Optional
`refinement_indicators` are evaluated after completed steps according to the
mesh's `MeshAdaptationOptions`; `velocity`, `pressure`, indicator fields, and
the additional `adaptive_fields` are transferred in place before
topology-dependent operators are rebuilt. The stored pressure is the latest
projection pressure. Automatic Krylov selection uses PCG for symmetric
positive (semi-)definite systems and BiCGSTAB otherwise. Thus pressure and
coupled two-phase stress diffusion use PCG, while a corrected nonsymmetric
single-phase AMR diffusion operator uses BiCGSTAB. Explicit choices take
precedence. The two preconditioners are independent. Both configurations use
rediscretized tree-geometric hierarchies by default.

For a liquid/gas velocity tuple, `two_velocity_projection` selects the
physical pressure-gradient model. `CMDGFMProjection()` is the default
nonsymmetric conservative multidimensional ghost-fluid system and BiCGSTAB
solve. `MimeticProjection()` selects the symmetric three-block formulation.
`LocalMimeticProjection()` replaces its aperture-divergence jump condition by
a rank-safe local aperture-normal condition while keeping the symmetric
energy projection. Its default is the cell-centred form with `c*(1-c)` jump
interpolation; the face-local and purely geometric interpolation variants
remain selectable.
"""
mutable struct IncompressibleNavierStokesSolver{
        T, M <: BlockTreeMesh, R, MU, V, D, B, S, A, PK, DK, PKO, DKO,
        PP, DP, PO, QO, RI, AF, G,
    }
    mesh::M
    density::R
    dynamic_viscosity::MU
    kinematic_viscosity::V
    diffusion::D
    boundary::B
    # The compatible face metric is diagonal on a uniform mesh and sparse on
    # a mixed-level mesh. Dynamic adaptation may therefore change the
    # concrete EllipticSolver type while preserving its scalar type.
    pressure_solver::EllipticSolver{T}
    scheme::S
    amr_interface_scheme::A
    coarse_size::Int
    pre_sweeps::Int
    post_sweeps::Int
    max_levels::Int
    pressure_krylov::PK
    diffusion_krylov::DK
    pressure_krylov_options::PKO
    diffusion_krylov_options::DKO
    pressure_preconditioner::PP
    diffusion_preconditioner::DP
    pressure_preconditioner_options::PO
    diffusion_preconditioner_options::QO
    refinement_indicators::RI
    adaptive_fields::AF
    two_phase::Union{Nothing, TwoPhaseModel{T, M}}
    geometry::G
end

"""Concise alias for [`IncompressibleNavierStokesSolver`](@ref)."""
const NavierStokesSolver = IncompressibleNavierStokesSolver

"""
    NavierStokesSystem

Preferred name for the configured incompressible Navier--Stokes equation
system. `NavierStokesSolver` remains available for compatibility; linear
algebraic solution strategies are configured separately with
[`LinearSolverOptions`](@ref).
"""
const NavierStokesSystem = IncompressibleNavierStokesSolver

function IncompressibleNavierStokesSolver(
        mesh::M; density, dynamic_viscosity,
        boundary = NoSlip(), scheme = LW(),
        amr_interface_scheme = default_amr_interface_scheme(scheme), coarse_size = 64,
        pre_sweeps = 2, post_sweeps = 2, max_levels = 32,
        pressure_krylov = nothing,
        diffusion_krylov = nothing,
        pressure_krylov_options = (;),
        diffusion_krylov_options = (;),
        pressure_preconditioner = :auto,
        diffusion_preconditioner = :auto,
        pressure_preconditioner_options = (;),
        diffusion_preconditioner_options = (;),
        refinement_indicators = (), adaptive_fields = (),
        geometry = nothing,
        interface = nothing, surface_tension = 0.0,
        surface_tension_formulation = CGFM(), gravity = (0.0, 0.0),
        phase_scalars = (),
        interface_transport = nothing, interface_method = nothing,
        two_velocity_projection = CMDGFMProjection(),
        T = eltype(mesh.origin)
    ) where {M <: BlockTreeMesh}
    isnothing(geometry) || (
        geometry isa GeometryField{T, M} && geometry.mesh === mesh
    ) || throw(
        ArgumentError("geometry must share solver mesh and scalar type"),
    )
    model, configured_density, configured_dynamic_viscosity = if isnothing(interface)
        density isa Number || throw(
            ArgumentError("single-phase density must be a scalar"),
        )
        dynamic_viscosity isa Number || throw(
            ArgumentError("single-phase dynamic_viscosity must be a scalar"),
        )
        rho = T(density)
        mu = T(dynamic_viscosity)
        isfinite(rho) && rho > zero(T) || throw(
            ArgumentError("density must be finite and strictly positive"),
        )
        isfinite(mu) && mu > zero(T) || throw(
            ArgumentError(
                "dynamic_viscosity must be finite and strictly positive",
            ),
        )
        nothing, rho, mu
    else
        interface isa InterfaceField{T, M} && interface.mesh === mesh ||
            throw(ArgumentError("interface must share solver mesh and scalar type"))
        transport = _select_interface_transport(
            interface_transport, interface_method, FrozenInterfaceFluxes(),
        )
        phase_model = TwoPhaseModel(
            interface; density, dynamic_viscosity, surface_tension, gravity,
            formulation = surface_tension_formulation,
            scalars = phase_scalars,
            interface_transport = transport,
            pressure_projection = two_velocity_projection,
        )
        phase_model, phase_model.density, phase_model.dynamic_viscosity
    end
    selected_pressure_krylov = isnothing(pressure_krylov) ?
        Krylov.CgWorkspace : pressure_krylov
    # Keep `nothing` as the automatic-selection sentinel for single-phase
    # diffusion: AMR can change that operator from symmetric to nonsymmetric
    # at a later topology rebuild. Coupled two-phase diffusion always has the
    # SPD factorization M_rho + dt/2 E'W_mu E.
    selected_diffusion_krylov = isnothing(diffusion_krylov) &&
        !isnothing(model) ? Krylov.CgWorkspace : diffusion_krylov
    selected_pressure_preconditioner = pressure_preconditioner === :auto ?
        :tree_geometric :
        pressure_preconditioner
    selected_diffusion_preconditioner = diffusion_preconditioner === :auto ?
        :tree_geometric :
        diffusion_preconditioner
    configured_pressure_preconditioner_options =
        !isnothing(geometry) && selected_pressure_preconditioner in
        (:geometric, :tree_geometric) ?
        merge(pressure_preconditioner_options, (; geometry)) :
        pressure_preconditioner_options
    _validate_amr_interface_scheme(scheme, amr_interface_scheme)
    kinematic_viscosity, diffusion = if isnothing(model)
        kinematic_viscosity = configured_dynamic_viscosity / configured_density
        operator = isnothing(geometry) ? StaggeredDiffusionOperator(
                mesh; viscosity = kinematic_viscosity, boundary, T,
            ) : StaggeredDiffusionOperator(
                geometry; viscosity = kinematic_viscosity, boundary,
            )
        kinematic_viscosity, operator
    else
        # The coupled two-phase operator depends on the advected interface.
        # TwoPhaseFlowWorkspace owns it and refreshes its stress-location
        # dynamic viscosities before every viscous substep.
        nothing, nothing
    end
    pressure_solver = EllipticSolver(
        isnothing(model) ? (
                isnothing(geometry) ? PoissonOperator(
                    mesh; face_coefficient = inv(configured_density), T,
                ) : PoissonOperator(
                    geometry; face_coefficient = inv(configured_density),
                )
            ) : (
                isnothing(geometry) ? PoissonOperator(
                    interface; density = model.density,
                    formulation = model.formulation,
                ) : PoissonOperator(
                    interface, geometry; density = model.density,
                    formulation = model.formulation,
                )
            ); krylov = selected_pressure_krylov,
        krylov_options = pressure_krylov_options,
        preconditioner = selected_pressure_preconditioner,
        preconditioner_options = configured_pressure_preconditioner_options,
        coarse_size, pre_sweeps, post_sweeps, max_levels
    )
    indicators = _as_configuration_tuple(refinement_indicators)
    all(indicator -> indicator isa AbstractRefinementIndicator, indicators) ||
        throw(
        ArgumentError(
            "refinement_indicators must contain refinement indicators"
        )
    )
    all(indicator -> _indicator_field(indicator).mesh === mesh, indicators) ||
        throw(ArgumentError("refinement indicator fields must use solver mesh"))
    fields = _as_configuration_tuple(adaptive_fields)
    all(field -> field isa AbstractMeshField, fields) ||
        throw(
        ArgumentError(
            "adaptive_fields must contain cell or face fields"
        )
    )
    all(field -> field.mesh === mesh, fields) ||
        throw(ArgumentError("adaptive fields must use solver mesh"))
    return IncompressibleNavierStokesSolver{
        T, M, typeof(configured_density), typeof(configured_dynamic_viscosity),
        typeof(kinematic_viscosity), typeof(diffusion), typeof(boundary),
        typeof(scheme), typeof(amr_interface_scheme),
        typeof(selected_pressure_krylov), typeof(selected_diffusion_krylov),
        typeof(pressure_krylov_options), typeof(diffusion_krylov_options),
        typeof(selected_pressure_preconditioner),
        typeof(selected_diffusion_preconditioner),
        typeof(configured_pressure_preconditioner_options),
        typeof(diffusion_preconditioner_options),
        typeof(indicators), typeof(fields), typeof(geometry),
    }(
        mesh, configured_density, configured_dynamic_viscosity,
        kinematic_viscosity, diffusion, boundary, pressure_solver,
        scheme, amr_interface_scheme, coarse_size, pre_sweeps,
        post_sweeps, max_levels, selected_pressure_krylov,
        selected_diffusion_krylov,
        pressure_krylov_options, diffusion_krylov_options,
        selected_pressure_preconditioner, selected_diffusion_preconditioner,
        configured_pressure_preconditioner_options,
        diffusion_preconditioner_options,
        indicators, fields, model, geometry
    )
end

@inline _as_configuration_tuple(::Nothing) = ()
@inline _as_configuration_tuple(value::Tuple) = value
@inline _as_configuration_tuple(value::AbstractVector) = Tuple(value)
@inline _as_configuration_tuple(value) = (value,)

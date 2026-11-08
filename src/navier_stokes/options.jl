"""
    AdvectionOptions(; scheme=LaxWendroff(),
                     amr_interface_scheme=default_amr_interface_scheme(scheme))

Staggered momentum-advection choices for a Navier--Stokes system.
"""
struct AdvectionOptions{
        S <: AbstractStaggeredAdvectionScheme,
        A <: AbstractStaggeredAdvectionScheme,
    }
    scheme::S
    amr_interface_scheme::A
end

function AdvectionOptions(;
        scheme = LaxWendroff(),
        amr_interface_scheme = default_amr_interface_scheme(scheme),
    )
    scheme isa AbstractStaggeredAdvectionScheme || throw(
        ArgumentError("scheme must be a staggered-advection scheme"),
    )
    amr_interface_scheme isa AbstractStaggeredAdvectionScheme || throw(
        ArgumentError(
            "amr_interface_scheme must be a staggered-advection scheme",
        ),
    )
    _validate_amr_interface_scheme(scheme, amr_interface_scheme)
    return AdvectionOptions(scheme, amr_interface_scheme)
end

"""
    NavierStokesOptions(; advection=AdvectionOptions(),
                        pressure=LinearSolverOptions(),
                        diffusion=LinearSolverOptions(),
                        multigrid=MultigridOptions())

Numerical configuration for the incompressible Navier--Stokes equation
system. Physics, velocity-model choices, runtime AMR fields, and evolving
simulation state are deliberately configured elsewhere.
"""
struct NavierStokesOptions{
        A <: AdvectionOptions,
        P <: LinearSolverOptions, D <: LinearSolverOptions,
        M <: MultigridOptions,
    }
    advection::A
    pressure::P
    diffusion::D
    multigrid::M
end

function NavierStokesOptions(;
        advection = AdvectionOptions(),
        pressure = LinearSolverOptions(),
        diffusion = LinearSolverOptions(),
        multigrid = MultigridOptions(),
    )
    advection isa AdvectionOptions || throw(
        ArgumentError("advection must be AdvectionOptions"),
    )
    pressure isa LinearSolverOptions || throw(
        ArgumentError("pressure must be LinearSolverOptions"),
    )
    diffusion isa LinearSolverOptions || throw(
        ArgumentError("diffusion must be LinearSolverOptions"),
    )
    multigrid isa MultigridOptions || throw(
        ArgumentError("multigrid must be MultigridOptions"),
    )
    return NavierStokesOptions(
        advection, pressure, diffusion, multigrid,
    )
end

function _navier_stokes_keywords(
        options::NavierStokesOptions, adaptivity::Adaptivity,
    )
    advection = options.advection
    pressure = options.pressure
    diffusion = options.diffusion
    multigrid = options.multigrid
    return (;
        scheme = advection.scheme,
        amr_interface_scheme = advection.amr_interface_scheme,
        pressure_krylov = pressure.krylov,
        pressure_krylov_options = pressure.krylov_options,
        pressure_preconditioner = pressure.preconditioner,
        pressure_preconditioner_options = pressure.preconditioner_options,
        diffusion_krylov = diffusion.krylov,
        diffusion_krylov_options = diffusion.krylov_options,
        diffusion_preconditioner = diffusion.preconditioner,
        diffusion_preconditioner_options = diffusion.preconditioner_options,
        multigrid,
        refinement_indicators = adaptivity.criteria,
        adaptive_fields = adaptivity.fields,
    )
end

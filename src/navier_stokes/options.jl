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

    function AdvectionOptions(scheme, amr_interface_scheme)
        scheme isa AbstractStaggeredAdvectionScheme || throw(
            ArgumentError("scheme must be a staggered-advection scheme"),
        )
        amr_interface_scheme isa AbstractStaggeredAdvectionScheme || throw(
            ArgumentError(
                "amr_interface_scheme must be a staggered-advection scheme",
            ),
        )
        _validate_amr_interface_scheme(scheme, amr_interface_scheme)
        return new{typeof(scheme), typeof(amr_interface_scheme)}(
            scheme, amr_interface_scheme,
        )
    end
end

function AdvectionOptions(;
        scheme = LaxWendroff(),
        amr_interface_scheme = default_amr_interface_scheme(scheme),
    )
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

    function NavierStokesOptions(advection, pressure, diffusion, multigrid)
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
        return new{
            typeof(advection), typeof(pressure), typeof(diffusion), typeof(multigrid),
        }(advection, pressure, diffusion, multigrid)
    end
end

function NavierStokesOptions(;
        advection = AdvectionOptions(),
        pressure = LinearSolverOptions(),
        diffusion = LinearSolverOptions(),
        multigrid = MultigridOptions(),
    )
    return NavierStokesOptions(advection, pressure, diffusion, multigrid)
end

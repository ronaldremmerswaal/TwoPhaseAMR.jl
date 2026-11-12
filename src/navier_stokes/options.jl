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
                        multigrid=MultigridOptions(),
                        viscosity_averaging=HarmonicViscosityAveraging())

Numerical configuration for the incompressible Navier--Stokes equation
system. Physics, velocity-model choices, runtime AMR fields, and evolving
simulation state are deliberately configured elsewhere.

`viscosity_averaging` selects how the one-velocity `div(2 mu D(u))` operator
mixes the two phase viscosities, in the cells and at the stress vertices: a
[`HarmonicViscosityAveraging`](@ref) or a
[`GeometricViscosityAveraging`](@ref) rule.
"""
struct NavierStokesOptions{
        A <: AdvectionOptions,
        P <: LinearSolverOptions, D <: LinearSolverOptions,
        M <: MultigridOptions, V <: AbstractViscosityAveraging,
    }
    advection::A
    pressure::P
    diffusion::D
    multigrid::M
    viscosity_averaging::V

    function NavierStokesOptions(
            advection, pressure, diffusion, multigrid, viscosity_averaging,
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
        viscosity_averaging isa AbstractViscosityAveraging || throw(
            ArgumentError(
                "viscosity_averaging must be HarmonicViscosityAveraging() or GeometricViscosityAveraging()",
            ),
        )
        return new{
            typeof(advection), typeof(pressure), typeof(diffusion),
            typeof(multigrid), typeof(viscosity_averaging),
        }(advection, pressure, diffusion, multigrid, viscosity_averaging)
    end
end

function NavierStokesOptions(;
        advection = AdvectionOptions(),
        pressure = LinearSolverOptions(),
        diffusion = LinearSolverOptions(),
        multigrid = MultigridOptions(),
        viscosity_averaging = HarmonicViscosityAveraging(),
    )
    return NavierStokesOptions(
        advection, pressure, diffusion, multigrid, viscosity_averaging,
    )
end

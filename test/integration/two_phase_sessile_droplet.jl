using Test, TwoPhaseAMR

function _sessile_profile(
        initial_radius, contact_angle, density, sigma, gravity;
        points = 800
    )
    target_area = initial_radius^2 * (
        contact_angle - sin(contact_angle) * cos(contact_angle)
    )
    coefficient = (density[1] - density[2]) * gravity / sigma
    theta0 = pi - contact_angle

    function trajectory(omega0, length)
        step = length / points
        theta = theta0
        omega = omega0
        x = 0.0
        y = 0.0
        xs = Vector{Float64}(undef, points + 1)
        ys = similar(xs)
        xs[1] = x
        ys[1] = y
        derivative(state) = (
            state[2], -coefficient * sin(state[1]),
            cos(state[1]), sin(state[1]),
        )
        for index in 1:points
            state = (theta, omega, x, y)
            k1 = derivative(state)
            middle1 = ntuple(i -> state[i] + step * k1[i] / 2, 4)
            k2 = derivative(middle1)
            middle2 = ntuple(i -> state[i] + step * k2[i] / 2, 4)
            k3 = derivative(middle2)
            endpoint = ntuple(i -> state[i] + step * k3[i], 4)
            k4 = derivative(endpoint)
            theta, omega, x, y = ntuple(
                i -> state[i] + step *
                    (k1[i] + 2k2[i] + 2k3[i] + k4[i]) / 6,
                4
            )
            xs[index + 1] = x
            ys[index + 1] = y
        end
        xs .-= last(xs)
        area = 0.0
        for index in 1:points
            area -= (ys[index] + ys[index + 1]) *
                (xs[index + 1] - xs[index])
        end
        return theta, area, reverse(xs), reverse(ys)
    end

    unknown = [inv(initial_radius), initial_radius * contact_angle]
    for _ in 1:15
        theta, area, _, _ = trajectory(unknown...)
        residual = [theta - pi, (area - target_area) / target_area]
        maximum(abs, residual) < 2.0e-12 && break
        jacobian = zeros(2, 2)
        for column in 1:2
            perturbed = copy(unknown)
            increment = 1.0e-6max(abs(unknown[column]), 1.0)
            perturbed[column] += increment
            theta_p, area_p, _, _ = trajectory(perturbed...)
            jacobian[:, column] .= (
                [theta_p - pi, (area_p - target_area) / target_area] .-
                    residual
            ) ./ increment
        end
        unknown .-= jacobian \ residual
        all(>(0), unknown) ||
            error("sessile-profile Newton iteration left the physical branch")
    end
    theta, area, x, y = trajectory(unknown...)
    abs(theta - pi) < 1.0e-10 ||
        error("sessile reference failed its symmetry condition")
    abs(area - target_area) < 1.0e-10target_area ||
        error("sessile reference failed its volume constraint")
    return (; x, y, area = target_area)
end

function _sessile_polygon(profile, centre_x)
    vertices = [
        (centre_x + profile.x[index], profile.y[index])
            for index in reverse(eachindex(profile.x))
    ]
    append!(
        vertices,
        (
            (centre_x - profile.x[index], profile.y[index])
                for index in 2:length(profile.x)
        ),
    )
    return PolygonIndicator(vertices)
end

# Largest velocity: the face velocity, or the largest present phase value.
_sessile_maximum_velocity(velocity) = maximum(abs, velocity.interior)
_sessile_maximum_velocity(velocity::Tuple) = maximum(
    maximum(abs, filter(isfinite, phase.interior); init = 0.0) for phase in velocity
)

function sessile_droplet_case(
        resolution; contact_angle = pi / 2, kind = :uniform,
        velocity_model = OneVelocityModel(), bond = 1.21,
    )
    kind in (:uniform, :mixed, :amr) ||
        throw(ArgumentError("unknown mesh kind $kind"))
    initial_radius = 0.2
    density = (1.0, 1.0e-3)
    sigma = 1.0
    gravity_magnitude = bond * sigma /
        ((density[1] - density[2]) * initial_radius^2)
    laplace = 5.5e2
    liquid_viscosity = sqrt(
        sigma * density[1] * initial_radius / laplace
    )
    viscosity = (liquid_viscosity, 0.0175liquid_viscosity)
    profile = _sessile_profile(
        initial_radius, contact_angle, density,
        sigma, gravity_magnitude
    )
    polygon = _sessile_polygon(profile, 0.5)
    mesh = BlockTreeMesh(
        block_size = (resolution, resolution), max_level = 2,
        periodic = (false, false),
        adaptation = MeshAdaptationOptions(coarsen_delay = 2),
        refine = (key, origin, extent) -> kind == :mixed &&
            (
            key.level == 0 ||
                (key.level == 1 && key.j == 0)
        )
    )
    interface = InterfaceField(
        mesh, polygon;
        contact_angle = (nothing, nothing, contact_angle, nothing)
    )
    indicator = InterfaceRefinementIndicator(interface; level = 2)
    curvature_indicator = CurvatureRefinementIndicator(
        interface; cells_per_radius = 8
    )
    if kind == :amr
        adaptation = AMRWorkspace(mesh, interface)
        for step in 1:3
            stats = adapt!(
                adaptation, (indicator, curvature_indicator);
                step, force = true
            )
            stats.changed || break
            # Exact resampling is restricted to construction of the initial
            # mesh. Evolution and subsequent AMR use geometric PLIC transfer.
            initialize!(interface, polygon)
        end
    end
    reference = copy(interface.fraction)
    initial_volume = integrate(interface)
    flow = FlowSimulation(
        mesh, TwoPhase(
            interface; density, dynamic_viscosity = viscosity,
            surface_tension = SurfaceTension(sigma; method = CGFM()),
            gravity = (0.0, -gravity_magnitude)
        );
        boundary = NoSlip(), velocity_model,
        adaptivity = Adaptivity(
            criteria = kind == :amr ?
                (indicator, curvature_indicator) : ()
        ),
    )
    capillary_time = sqrt(
        sum(density) * initial_radius^3 / sigma
    )
    final_time = capillary_time / 4
    simulate!(
        flow, final_time; tolerance = 1.0e-9, max_cycles = 200
    )
    shape_error = sum(
        mesh.blocks[ref.block].volume * abs(
            TwoPhaseAMR._get(interface.fraction, ref) -
                TwoPhaseAMR._get(reference, ref)
        ) for ref in TwoPhaseAMR._cell_references(mesh)
    )
    return (;
        shape_error, maximum_velocity = _sessile_maximum_velocity(flow.velocity),
        volume_error = abs(integrate(interface) - initial_volume),
        initialization_error = abs(initial_volume - profile.area),
        reference_area = profile.area,
    )
end

@testset "Sessile droplet equilibrium convergence" begin
    # The thesis reports roughly h^1.5 convergence of the equilibrium
    # interface. The right-angle case isolates the bulk and symmetry-wall
    # method; all five triple-point regimes have their own curvature-order
    # study in height_functions.jl.
    results = [
        sessile_droplet_case(resolution; contact_angle = pi / 2)
            for resolution in (12, 24, 48)
    ]
    errors = getproperty.(results, :shape_error)
    rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
    println(
        "sessile convergence n=(12,24,48) errors=$errors rates=$rates"
    )
    @test all(diff(errors) .< 0)
    @test all(rates .>= 1.2)
    @test maximum(getproperty.(results, :initialization_error)) <= 4096eps()
    @test all(isfinite, getproperty.(results, :maximum_velocity))
    @test maximum(getproperty.(results, :volume_error)) <= 4096eps()

    # Exercise the same balanced equilibrium on a fixed nonconforming
    # tree and an interface/curvature-adapted tree. The refinement study
    # above supplies the accuracy assertion.
    for kind in (:mixed, :amr)
        result = sessile_droplet_case(
            12; contact_angle = pi / 2, kind
        )
        @test isfinite(result.shape_error)
        @test isfinite(result.maximum_velocity)
        @test result.initialization_error <= 4096eps()
        @test result.volume_error <= 4096eps()
    end
end

@testset "Sessile droplet with the two-velocity projections" begin
    # The equilibrium of the convergence study with both two-velocity
    # projections. The mimetic projection keeps the wall side of the floor
    # cells (contact cells included) in their Gram tensors and restricts the
    # metric to the interior sides (techref ch. 10, Walls).
    resolutions = (12, 24, 48)
    models = (
        ("CMDGFM", TwoVelocityModel()),
        ("MM", TwoVelocityModel(pressure_projection = MimeticProjection())),
    )
    results = Dict(
        name => [
            sessile_droplet_case(resolution; velocity_model = model)
                for resolution in resolutions
        ] for (name, model) in models
    )
    rates = Dict{String, Vector{Float64}}()
    for (name, _) in models
        errors = getproperty.(results[name], :shape_error)
        rates[name] = log2.(errors[1:(end - 1)] ./ errors[2:end])
        println(
            "sessile $name n=$resolutions errors=$errors rates=$(rates[name]) ",
            "max|u|=$(getproperty.(results[name], :maximum_velocity))",
        )
        @test all(diff(errors) .< 0)
        @test all(isfinite, getproperty.(results[name], :maximum_velocity))
        @test maximum(getproperty.(results[name], :volume_error)) <= 4096eps()
    end
    @test all(rates["CMDGFM"] .>= 1.2)
    # NUMERICS CONCERN: with gravity at the density ratio 1000 (Bo = 1.21) the
    # mimetic shape converges at about first order (rates 0.91, 0.79) and its
    # largest phase value stays near 1 (1.04 at n = 48 against 0.064 for the
    # CMDGFM): gas values of small aperture near the top of the drop, seven
    # cells above the floor. The wall coupling of gravity changes nothing there;
    # the cause is the cell-to-cell scatter of the height-function curvature,
    # which the per-phase force puts on the liquid (with an affine curvature
    # the drop stays at rest to 1e-8; numerics review, mm_walls/review).
    @test_broken all(rates["MM"] .>= 1.2)
    @test_broken last(results["MM"]).maximum_velocity <=
        2last(results["CMDGFM"]).maximum_velocity

    # Without gravity (a circular cap) at n = 48: at 90° the mimetic drop
    # matches the CMDGFM (shape error 1.08e-5 against 1.03e-5, largest phase
    # value 2.0e-3 against 1.8e-3).
    cap(angle) = Dict(
        name => sessile_droplet_case(48; contact_angle = angle, velocity_model = model, bond = 0.0)
            for (name, model) in models
    )
    right = cap(pi / 2)
    @test right["MM"].shape_error <= 1.25right["CMDGFM"].shape_error
    @test right["MM"].maximum_velocity <= 1.25right["CMDGFM"].maximum_velocity
    # NUMERICS CONCERN: at 60° and 120° the mimetic currents are 9 and 2.5
    # times those of the CMDGFM (0.23 against 0.025, 0.11 against 0.044) and
    # the shape errors 2.5 and 1.8 times. With the exact constant curvature
    # the same drops stay at rest to 1e-10 for a capillary time (the wall
    # blocks are consistent). At 60° the height-function contact-angle closure
    # over the first wall rows drives it (the exact curvature in rows 1 and 2
    # gives 0.019 against 0.053 for the CMDGFM); at 120° the near-top slivers
    # with C ≈ 0.015–0.023 and curvatures 6–9 % high (numerics review,
    # mm_walls/review).
    for angle in (pi / 3, 2pi / 3)
        inclined = cap(angle)
        @test isfinite(inclined["MM"].maximum_velocity)
        @test_broken inclined["MM"].maximum_velocity <= 2inclined["CMDGFM"].maximum_velocity
        @test_broken inclined["MM"].shape_error <= 1.25inclined["CMDGFM"].shape_error
    end
end

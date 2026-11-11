using Test, TwoPhaseAMR
const VOF = TwoPhaseAMR
include("circle_geometry.jl")
cell_values(field) = reduce(
    vcat,
    [vec(collect(interior(field, b))) for b in 1:nblocks(field.mesh)]
)
function vortex_mesh(kind, n)
    return BlockTreeMesh(
        block_size = (n, n), max_level = 2,
        adaptation = MeshAdaptationOptions(coarsen_delay = 0),
        refine = (k, o, e) -> kind == :mixed && (k.level == 0 || (k.level == 1 && k.i == 0 && k.j == 1))
    )
end
function phase_vortex(kind, n; scheme = Fromm(), two_velocity = false)
    mesh = vortex_mesh(kind, n)
    shape = (x, y) -> hypot(x - 0.5, y - 0.75) - 0.15
    f = InterfaceField(mesh, shape; lipschitz = 1)
    q = PhaseScalar(f; liquid = (x, y) -> 1 + 0.1sinpi(2x), gas = (x, y) -> 2 + 0.1cospi(2y))
    one = PhaseScalar(f)
    staggered_functions = (
        liquid = (x, y) -> sinpi(4x) * sinpi(4y),
        gas = (x, y) -> cospi(2x) * cospi(2y),
    )
    qx = PhaseFaceScalar{1}(
        f; liquid = staggered_functions.liquid,
        gas = staggered_functions.gas,
    )
    qy = PhaseFaceScalar{2}(f; liquid = 1, gas = 1)
    amr = AMRWorkspace(mesh, f, q, one, qx, qy)
    indicator = InterfaceRefinementIndicator(f; level = 2)
    initial_volume = integrate(f)
    initial_mass = (integrate(q, :liquid), integrate(q, :gas))
    function velocity_workspace()
        two_velocity || return FaceVelocity(mesh), WYAdvectionWorkspace(f)
        velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
        @test velocity[1] !== velocity[2]
        return velocity, WYAdvectionWorkspace(f, velocity)
    end
    velocity, work = velocity_workspace()
    psi(x, y) = sinpi(x)^2 * sinpi(y)^2 / pi
    function setflow!(time)
        carriers = velocity isa FaceVelocity ? (velocity,) : velocity
        for component in carriers
            for (i, face) in pairs(mesh.faces)
                x, y = face.center;h = face.area / 2
                component.interior[i] = cospi(time) * (
                    face.axis == 1 ?
                        (psi(x, y + h) - psi(x, y - h)) / face.area : -(psi(x + h, y) - psi(x - h, y)) / face.area
                )
            end
        end
        return
    end
    time = 0.0;step = 0;updates = 0;intrinsic_min = Inf;intrinsic_max = -Inf
    while time < 1 - 1.0e-14
        if kind == :amr && step % 5 == 0
            stats = adapt!(amr, (indicator,); step, force = true)
            if stats.changed
                velocity, work = velocity_workspace();updates += 1
            end
        end
        # Fix the peak-velocity WY CFL across refinement levels. Using the
        # peak also bounds the velocity throughout the step near reversal.
        setflow!(0.0)
        dt = min(stable_timestep(velocity, work; cfl = 0.4), 1 - time)
        setflow!(time + dt / 2)
        advect!(
            f, velocity, dt, work; scalars = (q, one), staggered_scalars = (qx, qy),
            sweep_order = iseven(step) ? (1, 2) : (2, 1), scheme
        )
        # These are the stored intrinsic values themselves. Checking every
        # entry catches unsafe small-fraction division without masking it by a
        # phase-weighted norm or by an arbitrary volume-fraction cutoff.
        diagnostics = (
            (:center_liquid, cell_values(q.liquid), cell_values(f)),
            (:center_gas, cell_values(q.gas), 1 .- cell_values(f)),
            (:face_liquid, qx.liquid.interior, qx.fraction.interior),
            (:face_gas, qx.gas.interior, 1 .- qx.fraction.interior),
        )
        for (name, values, carriers) in diagnostics
            intrinsic_min = min(intrinsic_min, minimum(values))
            intrinsic_max = max(intrinsic_max, maximum(values))
            @test all(isfinite, values)
        end
        time += dt;step += 1
    end
    reference = InterfaceField(mesh, shape; lipschitz = 1)
    reference_q = PhaseScalar(reference; liquid = (x, y) -> 1 + 0.1sinpi(2x), gas = (x, y) -> 2 + 0.1cospi(2y))
    reference_face = PhaseFaceScalar{1}(
        reference; liquid = staggered_functions.liquid,
        gas = staggered_functions.gas,
    )
    error = scalar_error = maximum_error = 0.0
    for (b, block) in pairs(mesh.blocks)
        difference = abs.(interior(f, b) .- interior(reference, b))
        error += block.volume * sum(difference)
        maximum_error = max(maximum_error, maximum(difference))
        scalar_error += block.volume * sum(
            interior(reference, b) .* abs.(interior(q.liquid, b) .- interior(reference_q.liquid, b)) +
                (1 .- interior(reference, b)) .* abs.(interior(q.gas, b) .- interior(reference_q.gas, b))
        )
        @test interior(one.liquid, b) ≈ ones(size(interior(one.liquid, b))) atol = 1.0e-11
        @test interior(one.gas, b) ≈ ones(size(interior(one.gas, b))) atol = 1.0e-11
    end
    VOF._phase_face_average!(qy.fraction, f.fraction)
    @test qy.liquid.interior ≈ ones(length(qy.liquid.interior)) atol = 1.0e-11
    @test qy.gas.interior ≈ ones(length(qy.gas.interior)) atol = 1.0e-11
    # Each split step accumulates one contribution per face. Bound global
    # conservation by the corresponding worst-case roundoff accumulation,
    # rather than by a resolution-independent decimal constant.
    conservation_tolerance =
        16 * step * length(mesh.faces) * eps(eltype(f))
    volume_error = abs(integrate(f) - initial_volume)
    liquid_amount_error = abs(integrate(q, :liquid) - initial_mass[1])
    gas_amount_error = abs(integrate(q, :gas) - initial_mass[2])
    @test volume_error <= conservation_tolerance
    @test liquid_amount_error <= conservation_tolerance
    @test gas_amount_error <= conservation_tolerance
    @test kind != :amr || updates > 1
    face_liquid_error = face_gas_error = 0.0;idx = 0
    for face in mesh.faces
        face.axis == 1 || continue
        idx += 1
        volume = face.area * (mesh.blocks[face.minus.block].spacing[1] + mesh.blocks[face.plus.block].spacing[1]) / 2
        cref = reference_face.fraction.interior[idx]
        face_liquid_error += volume * cref * abs(
            qx.liquid.interior[idx] - reference_face.liquid.interior[idx],
        )
        face_gas_error += volume * (1 - cref) * abs(
            qx.gas.interior[idx] - reference_face.gas.interior[idx],
        )
    end
    face_error = face_liquid_error + face_gas_error
    densities = (liquid = 1000.0, gas = 1.0)
    liquid_momentum_error = densities.liquid * face_liquid_error
    gas_momentum_error = densities.gas * face_gas_error
    symmetric_difference = circle_symmetric_difference(f, reference)
    method_name = two_velocity ? :two_velocity : :one_velocity
    println("vortex symmetric_difference=$symmetric_difference kind=$kind method=$method_name n=$n L1_fraction=$error Linf_fraction=$maximum_error L1_centered=$scalar_error L1_staggered_liquid=$face_liquid_error L1_staggered_gas=$face_gas_error L1_liquid_momentum=$liquid_momentum_error L1_gas_momentum=$gas_momentum_error liquid_amount_error=$liquid_amount_error gas_amount_error=$gas_amount_error steps=$step regrids=$updates intrinsic_extrema=($intrinsic_min,$intrinsic_max)")
    flush(stdout)
    return (;
        geometry = symmetric_difference, fraction = error,
        maximum_fraction = maximum_error, centered = scalar_error,
        staggered = face_error, staggered_liquid = face_liquid_error,
        staggered_gas = face_gas_error,
        liquid_momentum = liquid_momentum_error,
        gas_momentum = gas_momentum_error,
        volume = volume_error, liquid_amount = liquid_amount_error,
        gas_amount = gas_amount_error,
    )
end

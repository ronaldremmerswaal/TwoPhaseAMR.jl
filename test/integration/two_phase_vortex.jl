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
function phase_vortex(kind, n; scheme = Fromm())
    mesh = vortex_mesh(kind, n)
    shape = (x, y) -> hypot(x - 0.5, y - 0.75) - 0.15
    f = InterfaceField(mesh, shape; lipschitz = 1)
    q = PhaseScalar(f; liquid = (x, y) -> 1 + 0.1sinpi(2x), gas = (x, y) -> 2 + 0.1cospi(2y))
    one = PhaseScalar(f)
    qx = PhaseFaceScalar{1}(f; liquid = 1, gas = 1)
    qy = PhaseFaceScalar{2}(f; liquid = (x, y) -> 1 + 0.1sinpi(2x), gas = (x, y) -> 2 + 0.1cospi(2y))
    amr = AMRWorkspace(mesh, f, q, one, qx, qy)
    indicator = InterfaceRefinementIndicator(f; level = 2)
    initial_volume = integrate(f)
    initial_mass = (integrate(q, :liquid), integrate(q, :gas))
    velocity = FaceVelocity(mesh);work = WYAdvectionWorkspace(f)
    psi(x, y) = sinpi(x)^2 * sinpi(y)^2 / pi
    function setflow!(time)
        for (i, face) in pairs(mesh.faces)
            x, y = face.center;h = face.area / 2
            velocity.interior[i] = cospi(time) * (
                face.axis == 1 ?
                    (psi(x, y + h) - psi(x, y - h)) / face.area : -(psi(x + h, y) - psi(x - h, y)) / face.area
            )
        end
        return
    end
    time = 0.0;step = 0;updates = 0;intrinsic_min = Inf;intrinsic_max = -Inf
    while time < 1 - 1.0e-14
        if kind == :amr && step % 5 == 0
            stats = adapt!(amr, (indicator,); step, force = true)
            if stats.changed
                velocity = FaceVelocity(mesh);work = WYAdvectionWorkspace(f);updates += 1
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
            (:face_liquid, qy.liquid.interior, qy.fraction.interior),
            (:face_gas, qy.gas.interior, 1 .- qy.fraction.interior),
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
    reference_face = PhaseFaceScalar{2}(reference; liquid = (x, y) -> 1 + 0.1sinpi(2x), gas = (x, y) -> 2 + 0.1cospi(2y))
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
    VOF._phase_face_average!(qx.fraction, f.fraction)
    @test qx.liquid.interior ≈ ones(length(qx.liquid.interior)) atol = 1.0e-11
    @test qx.gas.interior ≈ ones(length(qx.gas.interior)) atol = 1.0e-11
    @test integrate(f) ≈ initial_volume atol = 1.0e-11
    @test integrate(q, :liquid) ≈ initial_mass[1] atol = 1.0e-11
    @test integrate(q, :gas) ≈ initial_mass[2] atol = 1.0e-11
    @test kind != :amr || updates > 1
    face_error = 0.0;idx = 0
    for face in mesh.faces
        face.axis == 2 || continue
        idx += 1
        volume = face.area * (mesh.blocks[face.minus.block].spacing[2] + mesh.blocks[face.plus.block].spacing[2]) / 2
        cref = reference_face.fraction.interior[idx]
        face_error += volume * (
            cref * abs(qy.liquid.interior[idx] - reference_face.liquid.interior[idx]) +
                (1 - cref) * abs(qy.gas.interior[idx] - reference_face.gas.interior[idx])
        )
    end
    symmetric_difference = circle_symmetric_difference(f, reference)
    println("vortex symmetric_difference=$symmetric_difference kind=$kind n=$n L1_fraction=$error Linf_fraction=$maximum_error L1_centered=$scalar_error L1_staggered=$face_error steps=$step regrids=$updates intrinsic_extrema=($intrinsic_min,$intrinsic_max)")
    flush(stdout)
    return (; geometry = symmetric_difference, fraction = error, maximum_fraction = maximum_error, centered = scalar_error, staggered = face_error)
end
# The thesis int_validation vortex figure reports O(h) in L∞(volume fraction).
# PLIC geometric error is O(h²); volume-weighted L¹ additionally measures
# cancellation within cut cells and is reported separately from geometric error.
if abspath(PROGRAM_FILE) == @__FILE__
    @testset "Classical T=1 vortex refinement study" begin
        for kind in (:uniform, :mixed, :amr)
            resolutions = kind == :uniform ? (16, 32, 64) : (8, 16, 32)
            errors = [phase_vortex(kind, n) for n in resolutions]
            for metric in keys(first(errors))
                values = [getproperty(e, metric) for e in errors]
                rates = log2.(values[1:(end - 1)] ./ values[2:end])
                println("convergence mesh=$kind metric=$metric n=$resolutions errors=$values rates=$rates")
            end
            # The thesis reports second-order symmetric-difference error for PLIC
            # reconstruction and close to second-order L1 error for the modified
            # high-order phase-scalar flux interpolant. L∞ fraction error is
            # first-order and is reported above, but is too alignment-sensitive for
            # a three-grid pass/fail estimate.
            for metric in (:geometry, :fraction, :centered, :staggered)
                values = [getproperty(e, metric) for e in errors]
                @test all(log2.(values[1:(end - 1)] ./ values[2:end]) .>= 1.5)
            end
        end
    end
end

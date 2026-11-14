using Test, LinearAlgebra, Random, TwoPhaseAMR

# The conservative height-function (CHF) curvature of the mimetic projection's
# capillary force and the internal force variants (techref ch. 8, "Curvature
# of the mimetic projection"; audit note on mimetic surface tension, sec. chf).

function _mimetic_curvature_case(
        n, levelset; periodic = (true, false), block = 8, density = (1.0, 1.0),
    )
    mesh = BlockTreeMesh(; block_size = (block, block), base_mesh = (n, n), periodic)
    interface = InterfaceField(mesh, levelset; lipschitz = 1)
    solver = MimeticProjectionSolver(interface; density)
    work = MimeticProjectionWorkspace(solver)
    reconstruct!(interface)
    TwoPhaseAMR._mimetic_assemble!(solver, work)
    kappa = TwoPhaseAMR._mimetic_chf_curvature!(fill(NaN, length(work.references)), work, interface)
    return (; mesh, interface, solver, work, kappa, h = 1 / n)
end

# Lattice index (I, J) of every cell, from its centre.
function _mimetic_curvature_lattice(case)
    return map(case.work.references) do reference
        x, y = cellcenter(case.mesh, reference.block, reference.i, reference.j)
        (floor(Int, x / case.h) + 1, floor(Int, y / case.h) + 1)
    end
end

_mimetic_ellipse(n; a = 0.3, b = 0.2) =
    (x, y) -> b * (hypot((x - 0.5 - 0.21 / n) / a, (y - 0.5 - 0.37 / n) / b) - 1)

@testset "Mimetic CHF curvature: height functions in divergence form, bound 2/h" begin
    S(D) = D / sqrt(1 + D^2)
    for (name, n, levelset, radius) in (
            ("circle", 32, (x, y) -> hypot(x - 0.5 - 0.21 / 32, y - 0.5 - 0.37 / 32) - 0.25, 0.25),
            ("circle", 64, (x, y) -> hypot(x - 0.5 - 0.21 / 64, y - 0.5 - 0.37 / 64) - 0.25, 0.25),
            ("ellipse", 64, _mimetic_ellipse(64), NaN),
            ("seam", 32, (x, y) -> hypot(mod(x + 0.48, 1) - 0.5, mod(y - 0.47, 1) - 0.5) - 0.2, 0.2),
        )
        periodic = name == "seam" ? (true, true) : (true, false)
        case = _mimetic_curvature_case(n, levelset; periodic)
        interface, h = case.interface, case.h
        lattice = _mimetic_curvature_lattice(case)
        fraction = zeros(n, n)
        code = zeros(Int, n, n)
        for (cell, reference) in pairs(case.work.references)
            c = TwoPhaseAMR._conditional_fraction(interface, reference)
            fraction[lattice[cell]...] = c
            0 < c < 1 || continue
            nx = TwoPhaseAMR._get(interface.normal_x, reference)
            ny = TwoPhaseAMR._get(interface.normal_y, reference)
            code[lattice[cell]...] = abs(ny) >= abs(nx) ? (ny > 0 ? 2 : -2) : (nx > 0 ? 1 : -1)
        end
        inside(k, axis) = periodic[axis] || 1 <= k <= n
        value(o, column, position) = o == 2 ? fraction[mod1(column, n), mod1(position, n)] :
            fraction[mod1(position, n), mod1(column, n)]
        compared = invalid = 0
        deviation = 0.0
        for (cell, (I, J)) in pairs(lattice)
            o = abs(code[I, J])
            if iszero(o)
                @test isnan(case.kappa[cell])
                continue
            end
            column, position = o == 2 ? (I, J) : (J, I)
            rows = (position - 3):(position + 3)
            liquid, gas = code[I, J] > 0 ? (first(rows), last(rows)) : (last(rows), first(rows))
            valid = inside(first(rows), o) && inside(last(rows), o) &&
                all(inside(q, 3 - o) && value(o, q, liquid) == 1 && value(o, q, gas) == 0 for q in (column - 1):(column + 1))
            if !valid
                @test isnan(case.kappa[cell])
                invalid += 1
                continue
            end
            H = [sum(value(o, q, r) for r in rows) for q in (column - 1):(column + 1)]
            @test case.kappa[cell] ≈ -(S(H[3] - H[2]) - S(H[2] - H[1])) / h rtol = 1.0e-13 atol = 1.0e-12
            isnan(radius) || (deviation = max(deviation, abs(case.kappa[cell] * radius - 1)))
            compared += 1
        end
        println("mimetic CHF ($name, n = $n): $compared cut cells with a valid window, $invalid without; max |κR - 1| = $deviation")
        @test compared > 0
        @test maximum(abs, filter(isfinite, case.kappa)) * h < 2
        # Second-order accuracy on the circle (measured 0.0200 at n = 32 and
        # 0.0056 at n = 64; second order would give 0.005 from 0.02).
        name == "circle" && @test deviation <= (n == 32 ? 0.03 : 0.0085)
    end
    # The bound holds on arbitrary lattices: |S| < 1 in every valid window.
    TP = TwoPhaseAMR
    rng = Xoshiro(20261114)
    largest = 0.0
    for periodic in ((true, true), (true, false), (false, false)), trial in 1:200
        n = 16
        mesh = BlockTreeMesh(; block_size = (8, 8), base_mesh = (n, n), periodic)
        curvature = TP._mimetic_curvature_lattice!(TP._MimeticCurvature(Float64), mesh, TP._mimetic_topology(mesh))
        for J in 1:n, I in 1:n
            u = rand(rng)
            band = trial % 2 == 0 ? mod(J + (I ÷ 4), 6) < 3 : (mod(I, 5) < 2) ⊻ (mod(J, 6) < 3)
            c = u < 0.3 ? rand(rng) : band ? 1.0 : 0.0
            curvature.fraction[I, J] = c
            curvature.orientation[I, J] = 0 < c < 1 ? rand(rng, Int8[-2, -1, 1, 2]) : Int8(0)
        end
        kappa = TP._mimetic_chf_values!(fill(NaN, n * n), curvature, 1 / n)
        finite = filter(isfinite, kappa)
        isempty(finite) || (largest = max(largest, maximum(abs, finite) / n))
    end
    println("mimetic CHF bound: largest |κ|h on adversarial lattices $largest (bound 2)")
    @test 1 < largest < 2
end

@testset "Mimetic CHF curvature is the derivative of the height-polyline length in a patch" begin
    # A periodic graph: one patch of valid columns. With the column heights
    # H_a = Σ_r C_{a,r} the length L = Σ_a h √(1 + (H_{a+1} - H_a)²) has
    # ∂L/∂V_c = κ_c on every cut cell (V = h² C).
    for n in (32, 64)
        graph(x, y) = y - 0.5 - 0.37 / n - 0.06cospi(2x) - 0.02sinpi(6x)
        case = _mimetic_curvature_case(n, graph)
        lattice = _mimetic_curvature_lattice(case)
        h = case.h
        C = zeros(n, n)
        for (cell, reference) in pairs(case.work.references)
            C[lattice[cell]...] = TwoPhaseAMR._conditional_fraction(case.interface, reference)
        end
        length_of(C) = sum(h * sqrt(1 + (sum(C[mod1(a + 1, n), :]) - sum(C[a, :]))^2) for a in 1:n)
        epsilon = 1.0e-6
        worst = 0.0
        cut = 0
        for (cell, (I, J)) in pairs(lattice)
            0 < C[I, J] < 1 || continue
            cut += 1
            @test isfinite(case.kappa[cell])
            C[I, J] += epsilon
            plus = length_of(C)
            C[I, J] -= 2epsilon
            minus = length_of(C)
            C[I, J] += epsilon
            derivative = (plus - minus) / (2epsilon * h^2)
            worst = max(worst, abs(derivative - case.kappa[cell]) / max(abs(case.kappa[cell]), 1.0))
        end
        println("mimetic CHF patch identity (graph, n = $n): $cut cut cells, worst relative defect $worst")
        @test cut >= n
        @test worst <= 1.0e-6
    end
end

@testset "Mimetic capillary force variants: Galilean invariance and the dual aperture" begin
    TP = TwoPhaseAMR
    n = 32
    drop(x, y) = hypot(mod(x - 0.43 - 0.21 / n + 0.5, 1) - 0.5, mod(y - 0.52 - 0.37 / n + 0.5, 1) - 0.5) - 0.23
    increments = map(((false, false), (true, false), (true, true))) do (chf, dual)
        TP._MIMETIC_CHF_CURVATURE[] = chf
        TP._MIMETIC_DUAL_APERTURE[] = dual
        case = try
            _mimetic_curvature_case(n, drop; periodic = (true, true), density = (1000.0, 1.0))
        finally
            TP._MIMETIC_CHF_CURVATURE[] = false
            TP._MIMETIC_DUAL_APERTURE[] = false
        end
        @test (case.solver.chf_curvature, case.solver.dual_aperture) == (chf, dual)
        rng = Xoshiro(7)
        moving = (; liquid_flux = rand(rng, length(case.mesh.faces)) .* (0.1 / n^2), liquid_velocity = FaceVelocity(case.mesh))
        moving.liquid_velocity.interior .= 0.3 .+ rand(rng, length(case.mesh.faces))
        result = map((nothing, moving)) do transport
            velocity = (FaceVelocity(case.mesh), FaceVelocity(case.mesh))
            TP._mark_inactive_phase_velocities!(velocity, case.interface)
            unresolved = Ref(-1)
            TP._mimetic_capillary_force!(
                velocity, case.solver, case.work, 0.01; sigma = 1.0, transport,
                unresolved_curvature_faces = unresolved,
            )
            @test unresolved[] == 0
            [copy(phase.interior) for phase in velocity]
        end
        (result, copy(case.work.cell_curvature))
    end
    (hf, hf_kappa), (chf, chf_kappa), (dual, _) = increments
    # The per-phase force does not read the velocity (bitwise), with either
    # curvature; the dual variant takes its liquid apertures from the transport.
    @test isequal(hf[1], hf[2])
    @test isequal(chf[1], chf[2])
    @test isequal(dual[1], chf[1])
    @test !isequal(dual[2], dual[1])
    # The CHF option changes the cut-cell curvature, by its O(h²) difference
    # from the height functions.
    @test !isequal(hf_kappa, chf_kappa)
    @test maximum(abs, filter(isfinite, hf_kappa .- chf_kappa)) <= 0.5
    # A flat interface at any offset: zero curvature and zero force.
    for theta in (0.1, 0.37, 0.8)
        TP._MIMETIC_CHF_CURVATURE[] = true
        case = try
            _mimetic_curvature_case(n, (x, y) -> y - 0.5 - theta / n; density = (1000.0, 1.0))
        finally
            TP._MIMETIC_CHF_CURVATURE[] = false
        end
        velocity = (FaceVelocity(case.mesh), FaceVelocity(case.mesh))
        TP._mark_inactive_phase_velocities!(velocity, case.interface)
        TP._mimetic_capillary_force!(velocity, case.solver, case.work, 1.0e-3; sigma = 1.0)
        @test maximum(abs, filter(isfinite, case.work.cell_curvature)) == 0
        @test maximum(maximum(abs, filter(isfinite, phase.interior)) for phase in velocity) == 0
    end
end

@testset "Mimetic CHF curvature allocates nothing after warm-up" begin
    case = _mimetic_curvature_case(32, (x, y) -> hypot(x - 0.5 - 0.21 / 32, y - 0.5 - 0.37 / 32) - 0.25)
    curvature!(case) = TwoPhaseAMR._mimetic_cell_curvature!(case.work, case.interface, nothing, true)
    curvature!(case)
    @test (@allocated curvature!(case)) == 0
end

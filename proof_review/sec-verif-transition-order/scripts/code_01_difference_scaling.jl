# Code check 01: the order 1/2 of the L2 difference between the stress with uniform viscosity and the scalar Laplacian on
# composite meshes comes from rows of order one on a strip of area O(h) (sec:verif:orders, closing paragraph).
#
# Label:  (no \label: the paragraph after tab:verif:orders; slug sec-verif-transition-order)
# Source: docs/techref/chapters/14_verification.tex:473-481 ("... converges with order 0.5 on composite meshes ... the L^2 order of a
#         difference of order one in the rows along a level transition, whose cells cover an area of order h_min"),
#         test/operators/embedded_variable_wall_mms.jl ("uniform-viscosity stress versus scalar diffusion (#106)", operator_difference)
# Checks (composite periodic-in-x box of 8 x 8, right half refined, transitions on x = 4 and on the periodic seam x = 0 = 8, equal viscosities (1,1), no-slip walls,
#         exactly divergence-free field of the test; n = 16, 32, 64, 128 cells across the coarse half):
#   T1  the order of the total L2 difference is 1/2 (the value printed by the test, 0.48 - 0.50);
#   T2  the maximum |difference| over the strips (distance <= 2 h_c to a transition) does not decrease (order 0): rows of order ONE;
#   T3  the area of the rows with |difference| >= 0.1 max is proportional to h (order 1): the cells cover an area O(h_min);
#   T4  L2 over the strip scales as h^(1/2), L2 over the rest decays at least like h (so the strip dominates);
#   T5  consequence stated as a condition: the maximum norm of the difference does not converge (order 0), it is not asserted;
#   T6  the truncation of the stress operator alone is order one, that of the scalar band rows O(h);
#   T7  CONDITION on the field: the test field has u = 0 on the transitions (sin(Kx) = 0 at x = 0, 4, 8). For the same field with a phase
#       (sin(Kx + phi)) the difference GROWS: L2 order -1/2, maximum order -1 (flux inconsistency of order one on the strip);
#   T8  the embedded-wall variant of the test (solid y <= 1) has the same order 1/2;
#   T9  isolating experiment: fully periodic two-level mesh, divergence-free fields (sin 2 pi y, 0) and (0, sin 2 pi x): the truncation of
#       the stress operator (mu = 1, through CellField and through InterfaceField) on the transition rows is O(1/h), the scalar band O(h).
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/sec-verif-transition-order/scripts/code_01_difference_scaling.jl

using Test, TwoPhaseAMR

const L = 8.0
const K = 2pi / L

function mesh_of(n, refined)
    return BlockTreeMesh(;
        base_mesh = (n, n), block_size = (4, 4), extent = (L, L),
        periodic = (true, false), max_level = 1,
        refine = (key, origin, extent) -> refined && key.level == 0 && origin[1] >= L / 2,
    )
end

function difference_field(n, phi = 0.0)
    mesh = mesh_of(n, true)
    interface = InterfaceField(mesh)
    initialize!(interface, (0.0, 1.0), 4.0)
    variable = StaggeredDiffusionOperator(interface; viscosity = (1.0, 1.0), boundary = NoSlip())
    scalar = StaggeredDiffusionOperator(mesh; viscosity = 1.0, boundary = NoSlip())
    volumes = mesh.face_dual_volumes
    a = pi / L
    velocity = FaceVelocity(mesh)
    for (i, face) in pairs(mesh.faces)
        volumes[i] > 0 || continue
        x, y = face.center
        s, c = sincos(a * y)
        velocity.interior[i] = face.axis == 1 ? sin(K * x + phi) * (2a * s * c) : -K * cos(K * x + phi) * s^2
    end
    velocity.interior[variable.dirichlet_faces] .= 0
    r1, r2 = FaceVelocity(mesh), FaceVelocity(mesh)
    diffusion!(r1, velocity, scalar)
    diffusion!(r2, velocity, variable)
    return mesh, volumes, r2.interior .- r1.interior
end

function summary(n)
    mesh, V, d = difference_field(n)
    hc = L / n
    xs = [f.center[1] for f in mesh.faces]
    axes_ = [f.axis for f in mesh.faces]
    # two transitions: x = 4 (coarse left, fine right) and the periodic seam x = 0 = 8 (fine left, coarse right)
    strip = min.(abs.(xs .- L / 2), xs, L .- xs) .<= 2.01hc
    total = sum(V)
    l2(mask) = sqrt(sum(V[i] * d[i]^2 for i in eachindex(d) if mask[i]) / total)
    dmax = maximum(abs, d)
    big = abs.(d) .>= 0.1 * dmax
    return (;
        n, hc, l2 = l2(trues(length(d))), strip = l2(strip), rest = l2(.!strip),
        maxstrip = maximum(abs(d[i]) for i in eachindex(d) if strip[i]),
        maxrest = maximum(abs(d[i]) for i in eachindex(d) if !strip[i]),
        area = sum(V[big]), nbig = count(big), allbig_in_strip = all(strip[big]),
    )
end

rate(a, b) = log2(a / b)

@testset "order 1/2 of the stress-versus-Laplacian difference" begin
    results = [summary(n) for n in (16, 32, 64, 128)]
    for r in results
        println(
            "n=$(r.n): L2 ", round(r.l2, sigdigits = 4), "  L2(strip) ", round(r.strip, sigdigits = 4), "  L2(rest) ", round(r.rest, sigdigits = 4),
            "  max|d| strip ", round(r.maxstrip, sigdigits = 4), "  max|d| rest ", round(r.maxrest, sigdigits = 4),
            "  area(|d|>=0.1max) ", round(r.area, sigdigits = 4), " (", r.nbig, " faces, all in strip: ", r.allbig_in_strip, ")",
        )
    end
    for (a, b) in zip(results[1:(end - 1)], results[2:end])
        println(
            "  $(a.n)->$(b.n): order L2 ", round(rate(a.l2, b.l2), digits = 3), ", L2 strip ", round(rate(a.strip, b.strip), digits = 3),
            ", L2 rest ", round(rate(a.rest, b.rest), digits = 3), ", max strip ", round(rate(a.maxstrip, b.maxstrip), digits = 3),
            ", area ", round(rate(a.area, b.area), digits = 3),
        )
    end
    last2 = results[(end - 1):end]
    @test 0.4 < rate(last2[1].l2, last2[2].l2) < 0.6                       # T1
    @test all(rate(a.maxstrip, b.maxstrip) < 0.3 for (a, b) in zip(results[1:(end - 1)], results[2:end]))   # T2 (T5)
    @test all(a.allbig_in_strip for a in results if a.n >= 32)       # n = 16 is pre-asymptotic (a few wall-corner rows lie just outside)
    # T3: the area of the large rows is proportional to h: area(n)/area(2n) -> 2 (asymptotically; the coarse pairs are pre-asymptotic)
    @test 0.8 < rate(last2[1].area, last2[2].area) < 1.2
    # T4
    @test 0.4 < rate(last2[1].strip, last2[2].strip) < 0.6
    @test rate(last2[1].rest, last2[2].rest) > 0.9
    @test last2[2].strip > 5 * last2[2].rest
end

# ---- T6: the truncation of the stress operator itself (against the analytic Laplacian of the same field) is order one, that of the
# scalar band Laplacian is O(h): the difference of order one comes from the stress rows.
function laplacian_exact(x, y, axis)
    a = pi / L
    return axis == 1 ? -(K^2 + 4a^2) * sin(K * x) * a * sin(2a * y) :
        (K^3 / 2) * cos(K * x) - (K / 2) * (K^2 + 4a^2) * cos(K * x) * cos(2a * y)
end

@testset "truncation of the stress and of the scalar operator separately" begin
    stress_max = Float64[]
    scalar_max = Float64[]
    for n in (16, 32, 64, 128)
        mesh = mesh_of(n, true)
        interface = InterfaceField(mesh)
        initialize!(interface, (0.0, 1.0), 4.0)
        variable = StaggeredDiffusionOperator(interface; viscosity = (1.0, 1.0), boundary = NoSlip())
        scalar = StaggeredDiffusionOperator(mesh; viscosity = 1.0, boundary = NoSlip())
        a = pi / L
        velocity = FaceVelocity(mesh)
        for (i, face) in pairs(mesh.faces)
            x, y = face.center
            s, c = sincos(a * y)
            velocity.interior[i] = face.axis == 1 ? sin(K * x) * (2a * s * c) : -K * cos(K * x) * s^2
        end
        velocity.interior[variable.dirichlet_faces] .= 0
        r1, r2 = FaceVelocity(mesh), FaceVelocity(mesh)
        diffusion!(r1, velocity, scalar)
        diffusion!(r2, velocity, variable)
        hc = L / n
        ms = mc = 0.0
        for (i, f) in pairs(mesh.faces)
            x, y = f.center
            min(abs(x - L / 2), x, L - x) <= 2.01hc && min(y, L - y) >= 3hc || continue
            ms = max(ms, abs(r2.interior[i] - laplacian_exact(x, y, f.axis)))
            mc = max(mc, abs(r1.interior[i] - laplacian_exact(x, y, f.axis)))
        end
        push!(stress_max, ms)
        push!(scalar_max, mc)
    end
    println("max truncation over the transition rows (wall rows excluded): stress ", round.(stress_max, sigdigits = 4), " scalar ", round.(scalar_max, sigdigits = 4))
    @test rate(stress_max[3], stress_max[4]) < 0.05                    # order one: no decay
    @test 0.9 < rate(scalar_max[3], scalar_max[4]) < 1.1               # band rows: O(h)
end

# ---- T7: dependence on the field (phase shift of the streamfunction: the normal velocity on the transitions no longer vanishes)
@testset "the order 1/2 is a property of the test field" begin
    for (phi, l2lo, l2hi, maxhi) in ((0.0, 0.4, 0.6, 0.3), (0.7, -0.6, -0.4, -0.7), (pi / 2, -0.6, -0.4, -0.9))
        ls = Float64[]
        ms = Float64[]
        for n in (16, 32, 64, 128)
            mesh, V, d = difference_field(n, phi)
            push!(ls, sqrt(sum(V .* d .^ 2) / sum(V)))
            push!(ms, maximum(abs, d))
        end
        println("phi=$phi: L2 ", round.(ls, sigdigits = 4), " orders ", round.(log2.(ls[1:3] ./ ls[2:4]), digits = 3), "  max ", round.(ms, sigdigits = 4), " orders ", round.(log2.(ms[1:3] ./ ms[2:4]), digits = 3))
        @test l2lo < rate(ls[3], ls[4]) < l2hi
        phi == 0 ? (@test rate(ms[3], ms[4]) < maxhi) : (@test rate(ms[3], ms[4]) < maxhi)
    end
end

# ---- T8: embedded-wall variant (solid y <= 1, interface at 4.5, as in the test)
function embedded_difference(n)
    mesh = mesh_of(n, true)
    solid = InterfaceField(mesh)
    initialize!(solid, (0.0, 1.0), 1.0)
    geometry = GeometryField(solid)
    interface = InterfaceField(mesh; geometry)
    initialize!(interface, (0.0, 1.0), 4.5)
    variable = StaggeredDiffusionOperator(interface; viscosity = (1.0, 1.0), boundary = NoSlip())
    scalar = StaggeredDiffusionOperator(geometry; viscosity = 1.0, boundary = NoSlip())
    volumes = geometry.face_dual_volumes
    a = pi / (L - 1.0)
    velocity = FaceVelocity(mesh)
    for (i, face) in pairs(mesh.faces)
        volumes[i] > 0 || continue
        x, y = face.center
        s, c = sincos(a * (y - 1.0))
        velocity.interior[i] = face.axis == 1 ? sin(K * x) * (2a * s * c) : -K * cos(K * x) * s^2
    end
    velocity.interior[variable.dirichlet_faces] .= 0
    r1, r2 = FaceVelocity(mesh), FaceVelocity(mesh)
    diffusion!(r1, velocity, scalar)
    diffusion!(r2, velocity, variable)
    d = r2.interior .- r1.interior
    return sqrt(sum(volumes[i] * d[i]^2 for i in eachindex(d) if volumes[i] > 0) / sum(volumes))
end

@testset "embedded-wall variant" begin
    ls = [embedded_difference(n) for n in (16, 32, 64, 128)]
    println("embedded wall: L2 ", round.(ls, sigdigits = 4), " orders ", round.(log2.(ls[1:3] ./ ls[2:4]), digits = 3))
    @test 0.4 < rate(ls[2], ls[3]) < 0.6
    @test 0.4 < rate(ls[3], ls[4]) < 0.6
end

# ---- T9: stress truncation on a fully periodic two-level mesh (left half refined), fields with nonzero normal velocity on the transition
@testset "stress truncation O(1/h) for a field with normal velocity on the transition" begin
    for kind in (:u_of_y, :v_of_x), path in (:interface, :cellfield)
        smax = Float64[]
        cmax = Float64[]
        for n in (16, 32, 64, 128)
            mesh = BlockTreeMesh(base_mesh = (n, n), block_size = (4, 4), periodic = (true, true), max_level = 1, refine = (key, origin, extent) -> key.level == 0 && origin[1] < 0.5)
            iface = InterfaceField(mesh)
            initialize!(iface, (0.0, 1.0), 0.5)
            stress = path == :interface ? StaggeredDiffusionOperator(iface; viscosity = (1.0, 1.0)) :
                StaggeredDiffusionOperator(mesh; viscosity = CellField(mesh, (x, y) -> 1.0))
            scalar = StaggeredDiffusionOperator(mesh; viscosity = 1.0)
            value(x, y, a) = kind == :u_of_y ? (a == 1 ? sinpi(2y) : 0.0) : (a == 1 ? 0.0 : sinpi(2x))
            lap(x, y, a) = -4pi^2 * value(x, y, a)
            u = FaceVelocity(mesh, (x, y, t, a) -> value(x, y, a))
            t1, t2 = FaceVelocity(mesh), FaceVelocity(mesh)
            diffusion!(t1, u, stress)
            diffusion!(t2, u, scalar)
            hc = 1 / n
            m1 = m2 = 0.0
            for (i, f) in pairs(mesh.faces)
                x, y = f.center
                min(abs(x - 0.5), x, 1 - x) <= 2.01hc || continue
                m1 = max(m1, abs(t1.interior[i] - lap(x, y, f.axis)))
                m2 = max(m2, abs(t2.interior[i] - lap(x, y, f.axis)))
            end
            push!(smax, m1)
            push!(cmax, m2)
        end
        println(kind, " ", path, ": max|tau| stress ", round.(smax, sigdigits = 4), " (orders ", round.(log2.(smax[1:3] ./ smax[2:4]), digits = 3), ")  scalar ", round.(cmax, sigdigits = 4))
        @test all(log2.(smax[1:3] ./ smax[2:4]) .< -0.9)             # grows like 1/h
        @test log2(cmax[3] / cmax[4]) > 0.7                           # band rows decay
    end
end

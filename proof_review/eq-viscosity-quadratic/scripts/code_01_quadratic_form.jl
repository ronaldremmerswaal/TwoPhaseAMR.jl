# Code check 01: eq:viscosity:quadratic and the sentence "the stiffness is symmetric positive semidefinite".
#
# Label:  eq:viscosity:quadratic (also :operator)
# Source: docs/techref/chapters/09_viscosity.tex:296-318; src/two_phase/diffusion/variable_diffusion_{geometry,operator}.jl
# Checks:
#   Q1  v'S v = sum_c,k 2 mu_c |c^o| (d_k u_k)_c^2 + sum_v mu_v |omega_v| (d_2 u_1 + d_1 u_2)_v^2 + sum_w cond_w mu_c u_f^2
#       with the independent strain map and weights of _shared/viscosity_audit.jl (walled, cut, periodic; random v),
#   Q2  S is exactly symmetric, eigenvalues >= -1e-13 max (PSD); number of zero eigenvalues: periodic 2 (the two constants),
#       walled no-slip 0 (positive definite), x-periodic channel no-slip 0, free-slip walls: kernel dimension reported,
#   Q3  consistency with the dissipation integral: for u = curl psi, psi = sin^2(pi x/L) sin^2(pi y/L) (no-slip, solenoidal,
#       compact support) and constant mu, v'Sv -> int 2 mu eps(u):eps(u) dV; observed order on the uniform and on the composite
#       mesh (right half refined).  Order 2 on the uniform mesh; the composite value is reported.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-viscosity-quadratic/scripts/code_01_quadratic_form.jl
using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ViscosityAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "viscosity_audit.jl"))
using .ViscosityAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR
const MA = ViscosityAudit.MA
fix_drop_ls(x, y) = hypot(x - 4.3, y - 4.1) - 2.23

@testset "Q1: quadratic form = sum of weighted squares" begin
    rng = MersenneTwister(31)
    for fx in (fixture_walled, fixture_periodic, fixture_cut)
        field = fx()
        op = StaggeredDiffusionOperator(field; viscosity = FIX_VISC)
        E, meas, logmu, _ = independent_strain(field, FIX_VISC)
        w = meas .* exp.(logmu)
        n = size(E, 1)
        nf = length(field.mesh.faces)
        for trial in 1:5
            v = randn(rng, nf)
            q = dot(v, op.stiffness * v)
            # independent: normal + shear rows of E, plus wall rows of the code (cut) = the remaining rows of op.strain
            q_indep = dot(E * v, w .* (E * v))
            q_wall = 0.0
            if size(op.strain, 1) > n
                Ew = op.strain[(n + 1):end, :]
                ww = op.stress_weights[(n + 1):end]
                q_wall = dot(Ew * v, ww .* (Ew * v))
            end
            @test abs(q - q_indep - q_wall) <= 1.0e-12 * q
        end
    end
end

@testset "Q2: symmetric, positive semidefinite, kernel" begin
    function spectrum(field; boundary = NoSlip())
        op = StaggeredDiffusionOperator(field; viscosity = FIX_VISC, boundary)
        S = Matrix(op.stiffness)
        asym = maximum(abs, S - S') / maximum(abs, S)
        ev = eigvals(Symmetric(S))
        return asym, ev[1] / ev[end], count(<(1.0e-10 * ev[end]), ev), ev[end]
    end
    for (name, fx) in ((:walled, fixture_walled), (:periodic, fixture_periodic), (:cut, fixture_cut))
        asym, rel_min, nker, emax = spectrum(fx())
        println(rpad(name, 9), "asymmetry ", asym, "  lambda_min/lambda_max ", rel_min, "  kernel dim ", nker)
        @test asym == 0
        @test rel_min >= -1.0e-13
        @test nker == (name == :periodic ? 2 : 0)
    end
    for (periodic, boundary) in (((true, false), FreeSlip()), ((false, false), FreeSlip()), ((true, false), NoSlip()))
        mesh = BlockTreeMesh(;
            base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic, max_level = 1,
            refine = (key, o, e) -> key.level == 0 && key.i >= 2
        )
        field = InterfaceField(mesh, fix_drop_ls; lipschitz = 1)
        asym, rel_min, nker, _ = spectrum(field; boundary)
        println("periodic=", periodic, " ", typeof(boundary).name.name, ": asymmetry ", asym, " lambda_min/lambda_max ", rel_min, " kernel dim ", nker)
        @test asym == 0 && rel_min >= -1.0e-13
    end
end

# Q3: consistency with int 2 mu eps:eps
function dissipation_exact(L, mu; m = 1600)
    h = L / m
    xs = range(h / 2, L - h / 2; length = m)
    # psi = s(x) s(y), s = sin^2(pi t / L); u1 = psi_y = s(x) s'(y), u2 = -psi_x = -s'(x) s(y)
    s(t) = sin(pi * t / L)^2; s1(t) = (pi / L) * sin(2pi * t / L); s2(t) = 2 * (pi / L)^2 * cos(2pi * t / L)
    total = 0.0
    for y in xs, x in xs
        e11 = s1(x) * s1(y)                  # d u1/dx = s'(x) s'(y)
        e22 = -s1(x) * s1(y)                 # d u2/dy
        e12 = 0.5 * (s(x) * s2(y) - s2(x) * s(y))
        total += 2 * mu * (e11^2 + e22^2 + 2 * e12^2) * h^2
    end
    return total
end

function discrete_dissipation(n, refined, L, mu_levelset_gas)
    mesh = BlockTreeMesh(;
        base_mesh = (n, n), block_size = (4, 4), extent = (L, L), periodic = (false, false), max_level = refined ? 1 : 0,
        refine = (key, o, e) -> refined && key.level == 0 && key.i >= div(n, 8)
    )
    field = InterfaceField(mesh, (x, y) -> mu_levelset_gas; lipschitz = 1)       # +1: all gas, mu = FIX_VISC[2]
    op = StaggeredDiffusionOperator(field; viscosity = FIX_VISC)
    s(t) = sin(pi * t / L)^2; s1(t) = (pi / L) * sin(2pi * t / L)
    vel = FaceVelocity(mesh, (x, y, time, axis) -> axis == 1 ? s(x) * s1(y) : -s1(x) * s(y))
    return dot(vel.interior, op.stiffness * vel.interior)
end

@testset "Q3: consistency with the dissipation integral" begin
    L = 8.0
    exact = dissipation_exact(L, FIX_VISC[2])
    println("exact int 2 mu eps:eps = ", exact)
    for refined in (false, true)
        prev = nothing
        for n in (16, 32, 64)
            q = discrete_dissipation(n, refined, L, 1.0)
            err = abs(q - exact) / exact
            order = prev === nothing ? NaN : log2(prev / err)
            println("  refined=", refined, " n=", n, "  relative error ", err, "   observed order ", order)
            prev = err
            if !refined && n == 64
                @test order > 1.8
            end
        end
    end
end

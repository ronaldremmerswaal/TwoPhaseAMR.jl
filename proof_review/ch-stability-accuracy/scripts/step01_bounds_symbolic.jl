# Steps 1-4 of the audit of ch:stability-accuracy (Chapter 13): the algebra of the capillary and gravity bounds.
#
# Label:  ch:stability-accuracy (eq:stab:flow-bound, dev:stab:gravity; sec:stab:accuracy)
# Source: docs/techref/chapters/13_stability_accuracy.tex:76-112 (capillary bound, Denner--van Wachem comparison, 1.16),
#         107-128 (gravity), 66-75 (scalings), 489-497 (orders in h); thesis (4.1) st_intro.tex:20, (7.4) pt2_overview.tex:56-60
# Checks:
#   S1  c_sigma = (pi sigma/(rho h))^(1/2) is the phase speed omega/k of the capillary wave of wavelength 2h (omega^2 = sigma k^3/rho, rho = rho_l + rho_g);
#       dt_cap = h/(2 c_sigma) = (rho h^3/(4 pi sigma))^(1/2) = eq:vof:dt-forces = thesis (4.1);
#   S2  dt_cap = 2^(-1/2) T(lambda = h), T the period of the capillary wave; Denner--van Wachem static bound h/(sqrt2 c_sigma) = T(lambda = h) = sqrt2 dt_cap;
#       Doppler form h/(sqrt2 c_sigma + U) = period of the wave of wavelength h seen with the frequency shift k U;
#   S3  speed-corrected excess: R(u) = min{cfl/u, 1/2} (sqrt2 + u), u = U/c_sigma, has maximum cfl + 2^(-1/2) at u = 2 cfl (equal bounds), 1.157 at cfl = 0.45;
#   S4  dt_grav (code) = 2^(-1/2) x thesis (7.4) = 2^(-1/2) x period of the gravity wave of wavelength h/(2 pi); the code is sqrt2 stricter;
#   S5  scalings: d log dt_cap/d log h = 3/2, d log dt_grav/d log h = 1/2, d log dt_adv/d log h = 1; the ratios dt_cap/dt_adv ~ h^(1/2) -> 0,
#       dt_grav/dt_adv ~ h^(-1/2) -> infinity; an error C dt^p gives the order p, 3p/2, p/2 in h (here p = 1: 1, 3/2, 1/2).
# Run:    julia --project=@claude-symbolics proof_review/ch-stability-accuracy/scripts/step01_bounds_symbolic.jl

using Test, Random, Symbolics

Random.seed!(13)
@variables h sigma rho drho g cfl u s2 Uv

# Dispersion relations (infinite depth, thesis Lamb Art. 266 and 231): omega^2 = sigma k^3/rho (capillary), omega^2 = g k drho/rho (gravity),
# rho = rho_l + rho_g, drho = |rho_l - rho_g|.
omega_cap2(k) = sigma * k^3 / rho
omega_grav2(k) = g * k * drho / rho
period2(omega2) = 4pi^2 / omega2

c2 = pi * sigma / (rho * h)                  # c_sigma^2 of the chapter
dtcap2 = rho * h^3 / (4pi * sigma)           # eq:vof:dt-forces, code
dtgrav2_code = rho * h / (2 * drho * g)      # eq:vof:dt-forces, code
dtgrav2_thesis = rho * h / (drho * g)        # thesis (7.4)

zero_symbolic(expr) = isequal(Symbolics.simplify(Symbolics.expand(expr)), 0) ||
    isequal(Symbolics.simplify(Symbolics.simplify_fractions(expr)), 0)

@testset "ch13 bounds: symbolic" begin
    @testset "S1 phase speed of the wave of wavelength 2h" begin
        k2h = 2pi / (2h)
        @test zero_symbolic(omega_cap2(k2h) / k2h^2 - c2)                 # (omega/k)^2 = c_sigma^2
        @test zero_symbolic(h^2 / (4c2) - dtcap2)                         # (h/(2 c_sigma))^2 = dt_cap^2
    end
    @testset "S2 period of the wave of wavelength h; Denner--van Wachem" begin
        Th2 = period2(omega_cap2(2pi / h))                               # T(lambda = h)^2
        @test zero_symbolic(dtcap2 / Th2 - 1 // 2)                        # dt_cap = 2^(-1/2) T(lambda = h)
        @test zero_symbolic(h^2 / (2c2) - Th2)                            # h/(sqrt2 c_sigma) = T(lambda = h)
        @test zero_symbolic(h^2 / (2c2) - 2dtcap2)                        # = sqrt2 dt_cap (squares)
        # Doppler: wave speed at lambda = h is sqrt(2) c_sigma (omega/k with k = 2 pi/h); shift k U gives the frequency (sqrt2 c + U)/h
        c_h2 = omega_cap2(2pi / h) / (2pi / h)^2
        @test zero_symbolic(c_h2 - 2c2)
    end
    @testset "S3 speed-corrected excess" begin
        # u = U/c_sigma; uniform translation: dt_wy = cfl h/U, dt_cap = h/(2 c): both in units h/c
        R_cap(u) = (sqrt(2) + u) / 2                                      # branch u <= 2 cfl (capillary bound active)
        R_adv(u, cfl) = cfl * (sqrt(2) + u) / u                           # branch u >= 2 cfl (advective bound active)
        @test isequal(Symbolics.simplify(Symbolics.derivative(R_cap(u), u)), 1 // 2)          # increasing
        @test isequal(Symbolics.simplify(Symbolics.derivative(R_adv(u, cfl), u) * u^2 / cfl), -sqrt(2))  # decreasing
        # branches agree at the crossing u = 2 cfl (cfl h/U = h/(2 c)), value cfl + 2^(-1/2)
        @test zero_symbolic(R_cap(2cfl) - R_adv(2cfl, cfl))
        @test zero_symbolic((s2 + 2cfl) / 2 - (cfl + s2 / 2))             # R_cap(2 cfl) = cfl + s2/2 with s2 = sqrt(2)
        @test sqrt(2) / 2 ≈ 1 / sqrt(2) rtol = 1.0e-15                  # s2/2 = 2^(-1/2)
        # numerical maximisation over u for random cfl
        for cflv in (0.45, 0.5, 0.3, 0.05, rand(5)...)
            R(uv) = min(cflv / uv, 0.5) * (sqrt(2) + uv)
            grid = range(1.0e-6, 50; length = 2_000_001)
            vals = R.(grid)
            imax = argmax(vals)
            @test isapprox(vals[imax], cflv + 1 / sqrt(2); atol = 1.0e-4)
            @test isapprox(grid[imax], 2cflv; atol = 1.0e-4)
            @test R(2cflv) ≈ cflv + 1 / sqrt(2) rtol = 1.0e-14
        end
        @test round(0.45 + 1 / sqrt(2); digits = 2) == 1.16
        # direct check in dimensional variables for random parameters: selected step / dynamic bound
        for _ in 1:10
            hv, sv, rv = 0.01 + rand(), 0.1 + rand(), 0.5 + 5rand()
            cs = sqrt(pi * sv / (rv * hv))
            cflv = 0.05 + 0.45rand()
            Uv = 2cflv * cs                                                 # equal bounds
            selected = min(cflv * hv / Uv, hv / (2cs))
            dynamic = hv / (sqrt(2) * cs + Uv)
            @test selected / dynamic ≈ cflv + 1 / sqrt(2) rtol = 1.0e-13
            @test cflv * hv / Uv ≈ hv / (2cs) rtol = 1.0e-13
            @test sqrt(rv * hv^3 / (4pi * sv)) ≈ hv / (2cs) rtol = 1.0e-13
        end
    end
    @testset "S4 gravity bound" begin
        Tgrav2 = period2(omega_grav2(2pi / (h / (2pi))))                 # wavelength h/(2 pi)
        @test zero_symbolic(Tgrav2 - dtgrav2_thesis)                      # thesis (7.4) = period
        @test zero_symbolic(dtgrav2_code - dtgrav2_thesis / 2)            # code = 2^(-1/2) (7.4): sqrt2 stricter
    end
    @testset "S3b interval where the code exceeds the dynamic bound" begin
        # R > 1 for 2 - sqrt2 < u < sqrt2 cfl/(1 - cfl) (cfl = 0.45: 0.586 < u < 1.157)
        @test zero_symbolic((s2 + (2 - s2)) / 2 - 1)                                  # R_cap(2 - sqrt2) = 1
        @test zero_symbolic(cfl * (s2 + s2 * cfl / (1 - cfl)) / (s2 * cfl / (1 - cfl)) - 1)   # R_adv(sqrt2 cfl/(1 - cfl)) = 1
        c = 0.45
        @test isapprox(2 - sqrt(2), 0.585786; atol = 1.0e-6) && isapprox(sqrt(2) * c / (1 - c), 1.157; atol = 1.0e-3)
    end
    @testset "S5b mesh widths where the bounds cross (fixed U)" begin
        hcap = 4pi * cfl^2 * sigma / (rho * Uv^2)        # dt_cap = dt_adv: capillary active for h < hcap
        hgrav = Uv^2 * rho / (2 * cfl^2 * drho * g)      # dt_grav = dt_adv: gravity active only for h > hgrav
        @test zero_symbolic(substitute(dtcap2 - (cfl * h / Uv)^2, h => hcap))
        @test zero_symbolic(substitute(dtgrav2_code - (cfl * h / Uv)^2, h => hgrav))
    end
    @testset "S5 scalings in h" begin
        dlog(f2) = Symbolics.simplify(h * Symbolics.derivative(f2, h) / f2) / 2   # d log f / d log h for f = sqrt(f2)
        @test isequal(dlog(dtcap2), 3 // 2)
        @test isequal(dlog(dtgrav2_code), 1 // 2)
        # advective bound cfl h/U: exponent 1; so cap/adv ~ h^(1/2), grav/adv ~ h^(-1/2), grav/cap ~ h^(-1)
        @test (3 // 2 - 1, 1 // 2 - 1, 1 // 2 - 3 // 2) == (1 // 2, -1 // 2, -1)
        # an error E = C dt^p with dt ~ h^a has order a p in h
        for (a, p, expected) in ((1, 1, 1), (3 // 2, 1, 3 // 2), (1 // 2, 1, 1 // 2))
            @test a * p == expected
        end
    end
end

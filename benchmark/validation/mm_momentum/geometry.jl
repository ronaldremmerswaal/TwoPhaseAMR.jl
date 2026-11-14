# Static geometry study of the Gram metric 𝓜 against the lumped metric W:
#  (1) Gram mass Zᵀ𝓜Z versus the VOF mass ZᵀWZ = ρV (the obstruction), as a
#      drop slides through the grid; with and without mass-matched points;
#  (2) λ_max(W, 𝓜) and ‖𝓜⁻¹W‖_∞ (bounds of the increment design), and
#      max |(𝓜 - W)Z| / mass (local GCL mismatch of a pure flux form),
#      over random drops and lines with tiny cut volumes.
include(joinpath(@__DIR__, "common.jl"))

function assemble(levelset; n, density = (1.0, 1.0e-3), floor = sqrt(eps()), match = false)
    TP._MIMETIC_MASS_MATCH[] = match
    mesh = BlockTreeMesh(block_size = (n, n), periodic = (true, true))
    interface = InterfaceField(mesh, levelset; lipschitz = 1, fraction_floor = floor)
    solver = TP.MimeticProjectionSolver(interface; density)
    work = TP.MimeticProjectionWorkspace(solver)
    assembly = TP._mimetic_assemble!(solver, work)
    M = TP._mimetic_metric_matrix(work)
    TP._MIMETIC_MASS_MATCH[] = false
    return (; interface, solver, work, assembly, M, mesh, density)
end

"Zᵀ𝓜Z (4×4) and ZᵀWZ, per phase blocks."
function mass_tensors(a)
    Z = TP._mimetic_constant_fields(a.assembly, a.mesh)
    W = a.assembly.dof_mass
    return Z' * (a.M * Z), Z' * (W .* Z)
end

function spectral(a)
    band = a.assembly.band
    isempty(band) && return (1.0, 1.0, 0.0, 0.0)
    Mb = Matrix(a.M[band, band])
    Wb = a.assembly.dof_mass[band]
    L = cholesky(Symmetric(Mb)).L
    S = L \ (Diagonal(Wb) / L')
    lmax = maximum(eigvals(Symmetric((S + S') / 2)))
    K = Mb \ Diagonal(Wb)
    inf = maximum(sum(abs, K; dims = 2))
    neg = minimum(K)
    Z = TP._mimetic_constant_fields(a.assembly, a.mesh)
    E = a.M * Z .- a.assembly.dof_mass .* Z
    rel = 0.0
    for dof in band
        m = max(a.assembly.dof_mass[dof], 1.0e-300)
        rel = max(rel, maximum(abs, E[dof, :]) / max(m, a.M[dof, dof]))
    end
    return (lmax, inf, neg, rel)
end

open(joinpath(OUT, "geometry.txt"), "w") do io
    say(s) = (println(io, s); flush(io); println(s); flush(stdout))
    say("# (1) Gram mass minus VOF mass, drop R=0.2 sliding by x0 + s h (s in [0,1]), density (1, 1e-3)")
    for n in (16, 32, 64), match in (false, true)
        h = 1 / n
        vals = Float64[]
        offd = Float64[]
        shifts = Float64[]
        for s in range(0, 1; length = 21)
            a = assemble((x, y) -> hypot(x - 0.5 - s * h, y - 0.47 - 0.37s * h) - 0.2; n, match)
            G, Wt = mass_tensors(a)
            push!(vals, (G[1, 1] - Wt[1, 1]) / Wt[1, 1])   # liquid x
            push!(offd, G[1, 2] / Wt[1, 1])
            match && push!(shifts, TP._MIMETIC_MASS_MATCH_LAST[].max_shift)
        end
        say(@sprintf("n=%3d match=%-5s liquid (G-V)/V xx: mean=%+.3e range=%.3e  xy/V max=%.3e  max shift=%s",
            n, match, mean(vals), maximum(vals) - minimum(vals), maximum(abs, offd),
            isempty(shifts) ? "-" : @sprintf("%.2e h", maximum(shifts))))
    end
    say("# (2) increment bound λmax(W,𝓜), ‖𝓜⁻¹W‖∞, min entry of 𝓜⁻¹W, max |(𝓜-W)Z|/max(m,𝓜ff) over the band")
    for kind in (:drop, :line), floor in (sqrt(eps()), 1.0e-12), n in (16, 32)
        worst = (0.0, 0.0, 0.0, 0.0)
        fmin = 1.0
        for trial in 1:40
            h = 1 / n
            eps_ = 10.0^(-rand(1:9))     # sliver depth (units of h)
            if kind === :drop
                r = 0.15 + 0.1rand()
                yc = round((0.5 + r) * n) / n - r + eps_ * h   # top just above a grid line
                ls = (x, y) -> hypot(x - 0.5 - rand() * 0 - 0.123h, y - yc) - r
            else
                p = rand(((1, 1), (1, 2), (2, 1), (1, 3), (3, 2)))
                w = 0.17
                # one of the two lines p⋅x = c ± w passes eps h above the grid point (0.5, 0.5)
                c = (p[1] + p[2]) * 0.5 - w + eps_ * h * hypot(p...)
                ls = (x, y) -> (abs(mod(p[1] * x + p[2] * y - c + 0.5, 1) - 0.5) - w) / hypot(p...)
            end
            local a
            try
                a = assemble(ls; n, floor)
            catch err
                say("  assembly error: " * first(sprint(showerror, err), 100)); continue
            end
            fmin = min(fmin, min_fraction(a.interface))
            s = spectral(a)
            worst = (max(worst[1], s[1]), max(worst[2], s[2]), min(worst[3], s[3]), max(worst[4], s[4]))
        end
        say(@sprintf("%-4s n=%2d floor=%.0e  λmax=%.3f  ‖𝓜⁻¹W‖∞=%.3f  min(𝓜⁻¹W)=%.3e  max|EZ|/m=%.3e  minC=%.1e",
            kind, n, floor, worst..., fmin))
    end
end

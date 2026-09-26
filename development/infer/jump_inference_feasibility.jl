# Cheap necessary-condition check for the jump-inference proposal.
# Use an oracle circle normal: incompatible targets with exact geometry and
# production K,D cannot be fixed by aperture reconstruction of E_T.
# No KKT system is assembled.
using LinearAlgebra
using Printf
using TwoPhaseAMR

include(joinpath(@__DIR__, "..", "vortex_sheet_commutativity",
    "commuting_normal_interpolant.jl"))
using .VortexSheetCommutativity: build_problem

function vector_value(j, x, y)
    j == 1 && return (1.0, 0.0)
    j == 2 && return (0.0, 1.0)
    j == 3 && return (x - 0.5, 0.0)
    j == 4 && return (y - 0.5, 0.0)
    j == 5 && return (0.0, x - 0.5)
    j == 6 && return (0.0, y - 0.5)
    error("unknown vector sample")
end

function pressure_value(j, x, y)
    j == 1 && return x - 0.5
    j == 2 && return y - 0.5
    j == 3 && return sinpi(2x)
    j == 4 && return cospi(2y)
    error("unknown pressure sample")
end

function pressure_gradient(j, x, y)
    j == 1 && return (1.0, 0.0)
    j == 2 && return (0.0, 1.0)
    j == 3 && return (2pi * cospi(2x), 0.0)
    j == 4 && return (0.0, -2pi * sinpi(2y))
    error("unknown pressure sample")
end

function samples(problem)
    mesh = problem.mesh
    active = problem.active_faces
    k = problem.assembly.relative_diagonal
    n = length(active)
    z = zeros(n, 6)
    zt = zeros(n, 6)
    xp = zeros(n, 4)
    yp = zeros(n, 4)
    centres = TwoPhaseAMR._cell_references(mesh)
    incidence = problem.assembly.incidence
    coupling = problem.assembly.data.coupling[active]

    for j in axes(xp, 2)
        p = [pressure_value(j, cellcenter(mesh, c.block, c.i, c.j)...)
             for c in centres]
        xp[:, j] .= coupling .* (incidence * p)[active] ./ k
    end

    for (r, face_index) in pairs(active)
        face = mesh.faces[face_index]
        x, y = face.center
        rx, ry = x - problem.centre[1], y - problem.centre[2]
        radius = hypot(rx, ry)
        tx, ty = -ry / radius, rx / radius
        t_axis = face.axis == 1 ? tx : ty
        for j in axes(z, 2)
            vx, vy = vector_value(j, x, y)
            z[r, j] = face.axis == 1 ? vx : vy
            zt[r, j] = t_axis * (tx * vx + ty * vy)
        end
        for j in axes(xp, 2)
            gx, gy = pressure_gradient(j, x, y)
            yp[r, j] = (inv(problem.density[2]) -
                inv(problem.density[1])) * t_axis * (tx * gx + ty * gy)
        end
    end
    return (; k, z, zt, xp, yp)
end

function errors(a, b, k)
    d = a - b
    weighted_l2 = sqrt(sum(k .* vec(sum(abs2, d; dims = 2))) /
        (size(d, 2) * sum(k)))
    linf = maximum(abs, d)
    return (; weighted_l2, linf)
end

function projector_error_lower_bound(x, y, k)
    # Orthogonality requires <P*x,x-P*x>_K=0, so P*x lies on the
    # K-sphere with centre x/2 and radius ||x||_K/2. The distance
    # from target y to this sphere is the minimum possible K error
    # for this single sample over *all* K-orthogonal projectors.
    lower = zeros(size(x, 2))
    for j in axes(x, 2)
        xj = view(x, :, j)
        yj = view(y, :, j)
        distance_to_centre = sqrt(sum(k .* abs2.(yj .- xj ./ 2)))
        sphere_radius = sqrt(sum(k .* abs2.(xj))) / 2
        lower[j] = abs(distance_to_centre - sphere_radius) / sqrt(sum(k))
    end
    return lower
end

function study_case(n, density)
    problem = build_problem(n; density)
    (; k, z, zt, xp, yp) = samples(problem)
    x = hcat(z, xp)
    y = hcat(zt, yp)
    r = x - y
    lower = projector_error_lower_bound(x, y, k)
    tangent_lower = maximum(lower[1:6])
    pressure_bounds = lower[7:10]
    pressure_lower, pressure_sample = findmax(pressure_bounds)

    # Necessary for any K-orthogonal projector satisfying T X = Y.
    # Normalize each moment by the kinetic sizes of its two arguments.
    ky = sqrt.(k) .* y
    kr = sqrt.(k) .* r
    compatibility = transpose(ky) * kr
    max_cosine = 0.0
    for i in axes(y, 2), j in axes(r, 2)
        denom = norm(view(ky, :, i)) * norm(view(kr, :, j))
        denom > 1e-14 * maximum(k) &&
            (max_cosine = max(max_cosine,
                abs(compatibility[i, j]) / denom))
    end

    # A global, nonlocal reference: the K-orthogonal projector onto span(Y).
    # It is not the optimal approximate projector among all possible ranges.
    f = svd(ky; full = false)
    rank_y = count(>(1e-11 * maximum(f.S)), f.S)
    q = f.U[:, 1:rank_y]
    tx = (q * (transpose(q) * (sqrt.(k) .* x))) ./ sqrt.(k)
    tangent = errors(tx[:, 1:6], zt, k)
    pressure = errors(tx[:, 7:10], yp, k)
    return (; n, density, faces = length(k), rank_y, max_cosine,
        tangent_lower, pressure_lower, pressure_sample,
        tangent, pressure)
end

function main()
    println("Oracle circle feasibility with production K and D")
    println("density          n   faces rank p# compat   T bound   p bound" *
            "    T L2      T Linf      p L2      p Linf")
    for density in ((1.0, 1.0), (10.0, 1.0),
                    (1000.0, 1.0), (1.0, 1000.0))
        for n in (16, 24, 32, 48, 64, 96)
            result = study_case(n, density)
            @printf("%-15s %3d %6d %4d %2d %8.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n", string(density), n,
                    result.faces, result.rank_y, result.pressure_sample,
                    result.max_cosine,
                    result.tangent_lower, result.pressure_lower,
                    result.tangent.weighted_l2, result.tangent.linf,
                    result.pressure.weighted_l2, result.pressure.linf)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

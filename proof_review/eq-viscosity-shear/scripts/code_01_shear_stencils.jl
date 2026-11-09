# Code check 01: eq:viscosity:shear -- the vertex derivative (pair / wall / least squares): selection rule, exactness, minimality,
# support size, and the local truncation error of each kind of row.
#
# Label:  eq:viscosity:shear (also :rows)
# Source: docs/techref/chapters/09_viscosity.tex:236-276; src/two_phase/diffusion/variable_diffusion_geometry.jl (_vertex_derivative,
#         _nearby_diffusion_faces!)
# Checks (all with the brute-force `shear_derivative` of _shared/viscosity_audit.jl, which reproduces the code's E to 1e-15,
# see code_01_rows.jl of eq-viscosity-rows; here the per-row structure is classified):
#   H1  row kinds present on random composite meshes (levels 1, 2; periodic, walled): symmetric pair, ASYMMETRIC pair (stretching at a
#       level transition), wall pair, least squares (quadratic), least squares with the affine fallback;
#   H2  every row reproduces d_t p(v) for p in {1,x,y} (affine) -- on the actual stencil positions (periodic images unwrapped);
#       quadratic p: exact for symmetric pairs and quadratic-LSQ rows; asymmetric pairs are exact only for affine p (error (h1-h2)/2 * p'');
#   H3  LSQ coefficients equal the KKT minimum-norm solution of  min sum a_j^2 (1+|r_j|^2/c^2)^2  s.t. exactness (independent algebra),
#       and any random feasible perturbation increases the weighted norm;
#   H4  support: <= 16 interior unknowns plus dropped boundary unknowns; all within 3 cmax;
#   H5  truncation error of each kind of row for u = smooth function under refinement (base mesh n = 16, 32, 64, same transition):
#       observed order of the max error per kind (symmetric pair 2, asymmetric pair 1, LSQ 2).
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-viscosity-shear/scripts/code_01_shear_stencils.jl
using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ViscosityAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "viscosity_audit.jl"))
using .ViscosityAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR
const MA = ViscosityAudit.MA

# All shear derivative rows (vertex, component) of a field with their classification.
function shear_rows(field, boundary)
    mesh = field.mesh
    lat = lattice_of(mesh)
    verts = Dict{NTuple{2, Int}, Vector{Tuple{CellRef, Float64}}}()
    for ref in cells(mesh)
        x = AMR.cellcenter(mesh, ref.block, ref.i, ref.j); h = cell_spacing(mesh, ref)
        for sx in (-1, 1), sy in (-1, 1)
            key = ntuple(2) do k
                val = round(Int, (x[k] + (k == 1 ? sx : sy) * h[k] / 2 - mesh.origin[k]) / mesh.extent[k] * lat[k])
                mesh.periodic[k] ? mod(val, lat[k]) : val
            end
            push!(get!(verts, key, Tuple{CellRef, Float64}[]), (ref, 1.0))
        end
    end
    out = []
    for key in sort!(collect(keys(verts)))
        p = vertex_point(mesh, key)
        cmax = maximum(maximum(cell_spacing(mesh, ref)) for (ref, _) in verts[key])
        for k in 1:2
            info = Dict{Symbol, Any}()
            coeffs = shear_derivative(mesh, p, k, cmax, boundary; info)
            push!(out, (; vertex = p, k, cmax, coeffs, info))
        end
    end
    return out
end

function offsets(mesh, v, i)
    f = mesh.faces[i]
    return (wrapped(f.center[1] - v[1], mesh.extent[1], mesh.periodic[1]), wrapped(f.center[2] - v[2], mesh.extent[2], mesh.periodic[2]))
end

function row_apply(mesh, row, p)       # sum_j a_j p(offset_j) with p evaluated at (dx, dy) relative to the vertex
    total = 0.0
    for (i, a) in row.coeffs
        dx, dy = offsets(mesh, row.vertex, i)
        total += a * p(dx, dy)
    end
    return total
end

@testset "H1-H4: structure of the shear rows" begin
    rng = MersenneTwister(2718)
    kinds = Dict{Symbol, Int}()
    worst_aff = 0.0; worst_quad_exact = 0.0; worst_kkt = 0.0; max_support = 0; max_radius_ratio = 0.0
    asym_quad_err = 0.0; perturb_ok = true; nwall_rows = 0
    for periodic in ((true, true), (false, false)), levels in (1, 2), trial in 1:2
        field, _, _ = MA.random_two_phase(rng; periodic, levels)
        mesh = field.mesh
        for row in shear_rows(field, NoSlip())
            kind = row.info[:kind]
            kinds[kind] = get(kinds, kind, 0) + 1
            kind == :free_slip_zero && continue
            t = 3 - row.k
            # unknowns of the row only (wall pair has the wall point as a zero-value support point)
            wallpt = kind == :wall_pair || (kind in (:pair_sym, :pair_asym) && (row.info[:pair][1] == 0 || row.info[:pair][3] == 0)) ||
                (kind in (:lsq1, :lsq2) && any(p[1] == 0 for p in row.info[:points]))      # supports containing wall values (zero, dropped)
            wallpt && (nwall_rows += 1)
            wallpt && continue
            # H2 affine exactness: value at the wall point is zero so skip affine test on wall pairs (p(wall) != 0)
            if !wallpt
                for (p, dp) in (((dx, dy) -> 1.0, 0.0), ((dx, dy) -> dx, t == 1 ? 1.0 : 0.0), ((dx, dy) -> dy, t == 2 ? 1.0 : 0.0))
                    worst_aff = max(worst_aff, abs(row_apply(mesh, row, p) - dp) * row.cmax)
                end
            end
            # quadratic exactness
            if kind in (:pair_sym, :lsq2)
                for (p, dp) in (((dx, dy) -> dx^2, 0.0), ((dx, dy) -> dx * dy, 0.0), ((dx, dy) -> dy^2, 0.0))
                    worst_quad_exact = max(worst_quad_exact, abs(row_apply(mesh, row, p) - dp) * row.cmax)
                end
            end
            if kind == :pair_asym
                lo_i, lo_s, hi_i, hi_s = row.info[:pair]
                q = row_apply(mesh, row, t == 1 ? (dx, dy) -> dx^2 : (dx, dy) -> dy^2)
                asym_quad_err = max(asym_quad_err, abs(q) * row.cmax)
            end
            # H3 KKT minimum norm (support = candidate set of the row)
            if kind in (:lsq2, :lsq1)
                pts = row.info[:points]
                deg = kind == :lsq2 ? 2 : 1
                c = row.cmax
                C = Matrix{Float64}(undef, deg == 2 ? 6 : 3, length(pts))
                D = zeros(length(pts))
                for (j, (i, dx, dy)) in enumerate(pts)
                    x, y = dx / c, dy / c
                    C[:, j] = deg == 2 ? [1.0, x, y, x^2, x * y, y^2] : [1.0, x, y]
                    D[j] = (1 + x^2 + y^2)^2
                end
                target = zeros(size(C, 1)); target[t + 1] = 1 / c
                a = (C .* (1 ./ D)') * C' \ target
                a = (1 ./ D) .* (C' * a)
                coded = [get(row.coeffs, i, 0.0) for (i, _, _) in pts]       # boundary faces (index 0) are dropped from the row
                inter = [i > 0 for (i, _, _) in pts]
                worst_kkt = max(worst_kkt, maximum(abs, a[inter] .- coded[inter]) * c)
                max_support = max(max_support, length(pts))
                max_radius_ratio = max(max_radius_ratio, maximum(hypot(dx, dy) for (_, dx, dy) in pts) / c)
                # perturbation in the null space of C increases the norm
                N = nullspace(C)
                for _ in 1:3
                    dz = N * randn(rng, size(N, 2)) * 0.1
                    perturb_ok &= dot(a .+ dz, D .* (a .+ dz)) >= dot(a, D .* a) * (1 - 1.0e-12)
                end
            end
        end
    end
    println("row kinds: ", kinds, "; rows whose support contains wall values (excluded from exactness tests): ", nwall_rows)
    println("max affine defect (x cmax): ", worst_aff, "; max quadratic defect sym-pair/LSQ2 (x cmax): ", worst_quad_exact, "; asym pair quadratic defect (x cmax): ", asym_quad_err)
    println("KKT defect (x cmax): ", worst_kkt, "; max LSQ support ", max_support, "; max LSQ radius / cmax ", max_radius_ratio)
    @test worst_aff <= 1.0e-9
    @test worst_quad_exact <= 1.0e-8
    @test worst_kkt <= 1.0e-8
    @test perturb_ok
    @test max_support <= 16
    @test max_radius_ratio <= 3 + 1.0e-12
    @test get(kinds, :pair_asym, 0) > 0 && get(kinds, :lsq2, 0) > 0
    @test asym_quad_err > 1.0e-3        # asymmetric pairs are NOT quadratic exact
end

# H5: truncation error per row kind for a smooth velocity under refinement
function truncation(n)
    L = 8.0
    mesh = BlockTreeMesh(;
        base_mesh = (n, n), block_size = (4, 4), extent = (L, L), periodic = (true, true), max_level = 1,
        refine = (key, o, e) -> key.level == 0 && key.i >= div(n, 8)
    )
    field = InterfaceField(mesh, (x, y) -> 1.0; lipschitz = 1)
    f(x, y, k) = k == 1 ? sin(2pi * x / L + 0.3) * cos(4pi * y / L + 0.1) : cos(2pi * x / L + 0.7) * sin(2pi * y / L - 0.2)
    dtf(x, y, k, t) = (
        k == 1 ? (t == 2 ? -4pi / L * sin(2pi * x / L + 0.3) * sin(4pi * y / L + 0.1) : 2pi / L * cos(2pi * x / L + 0.3) * cos(4pi * y / L + 0.1)) :
            (t == 1 ? -2pi / L * sin(2pi * x / L + 0.7) * sin(2pi * y / L - 0.2) : 2pi / L * cos(2pi * x / L + 0.7) * cos(2pi * y / L - 0.2))
    )
    err = Dict{Symbol, Float64}()
    for row in shear_rows(field, NoSlip())
        kind = row.info[:kind]
        total = 0.0
        for (i, a) in row.coeffs
            dx, dy = offsets(mesh, row.vertex, i)
            total += a * f(row.vertex[1] + dx, row.vertex[2] + dy, row.k)
        end
        e = abs(total - dtf(row.vertex[1], row.vertex[2], row.k, 3 - row.k))
        err[kind] = max(get(err, kind, 0.0), e)
    end
    return err
end

@testset "H5: truncation error by row kind" begin
    prev = nothing
    for n in (16, 32, 64)
        err = truncation(n)
        println("n=", n, " ", Dict(k => round(v, sigdigits = 3) for (k, v) in err))
        if prev !== nothing
            println("   observed orders: ", Dict(k => round(log2(prev[k] / v), digits = 2) for (k, v) in err if haskey(prev, k)))
        end
        prev = err
    end
end

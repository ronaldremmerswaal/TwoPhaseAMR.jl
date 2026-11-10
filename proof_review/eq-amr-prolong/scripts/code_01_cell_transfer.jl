# Code check 01: cell-field transfer (eq:amr:restrict, eq:amr:prolong, eq:amr:mc).
#
# Labels: eq:amr:restrict, eq:amr:prolong, eq:amr:mc
# Source: docs/techref/chapters/12_amr_criteria_transfer.tex:430-480; src/mesh/transfer.jl:_mc_slope, src/mesh/adaptivity/field_transfer.jl:_prolong_child!
# Checks:
#   P1  (algebra, 4e6 random draws)  MC(a,b): sign, 1/4|s| <= 1/2 min(|a|,|b|), MC(a,a) = a, MC(a,b) = 0 for ab <= 0; every child
#       v +- s_x/4 +- s_y/4 lies in [min, max] of {v, 4 neighbours}; the bound is attained (child = neighbour) so it is sharp;
#   P2  integral: sum |c| a_c unchanged to round-off for random-noise data under random regrids (walled, periodic, channel meshes);
#   P3  the package's children equal the chapter's formula evaluated on the halo-filled parent data (independent replica),
#       merged cells equal the mean of the four children, kept cells are untouched;
#   P4  local bound of every child by the parent and its four lattice neighbours (as seen through the halo), noise data;
#   P5  constants are reproduced exactly; affine data (re-initialised exactly before every regrid) are reproduced exactly at every child whose parent stencil is exact
#       for the data as read through the halo; stencils through the piecewise-constant coarse-to-fine halo, a zero-gradient wall halo or the
#       periodic seam are first order (counted and reported), measured on composite meshes;
#   P6  quadratic data: order of the error of the prolongation under uniform refinement (2 for smooth data) -- reported.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-amr-prolong/scripts/code_01_cell_transfer.jl

using Test, Random, LinearAlgebra
include(joinpath(@__DIR__, "..", "..", "_shared", "amr_audit.jl"))
using .AmrAudit
using TwoPhaseAMR
const AA = AmrAudit

mc(a, b) = a * b <= 0 ? 0.0 : sign(a + b) * min(abs(a + b) / 2, 2abs(a), 2abs(b))

@testset "P1 MC limiter algebra" begin
    rng = MersenneTwister(1)
    worst_bound = 0.0
    sharp = 0.0
    for _ in 1:2_000_000
        # neighbour differences of the centre value v = 0 in x and y: left/right
        ax, bx, ay, by = (randn(rng) * exp(3randn(rng)) for _ in 1:4)
        sx, sy = mc(ax, bx), mc(ay, by)
        @test abs(sx) <= 2min(abs(ax), abs(bx)) + 1.0e-300
        sx != 0 && @test sign(sx) == sign(ax) == sign(bx)
        # neighbours are v - ax, v + bx, v - ay, v + by with v = 0
        lo, hi = extrema((0.0, -ax, bx, -ay, by))
        for ex in (-1, 1), ey in (-1, 1)
            child = ex * sx / 4 + ey * sy / 4
            tol = 1.0e-13 * max(abs(lo), abs(hi), 1.0)
            (lo - tol <= child <= hi + tol) || (worst_bound = max(worst_bound, max(lo - child, child - hi)))
            # how close can the child get to the extreme neighbour: (child - 0)/(hi - 0) when positive etc.
            child > 0 && hi > 0 && (sharp = max(sharp, child / hi))
            child < 0 && lo < 0 && (sharp = max(sharp, child / lo))
        end
    end
    @test worst_bound == 0.0
    println("P1 child outside the neighbour range: ", worst_bound, ";  sup child/neighbour-extreme over draws = ", sharp)
    # explicit extremiser: x neighbours (-3, +1), y neighbours (-3, +1): s = 2 in both, child (+,+) = 1 = the largest neighbour
    @test mc(3.0, 1.0) == 2.0 && 0.25 * mc(3.0, 1.0) + 0.25 * mc(3.0, 1.0) == 1.0
    @test mc(2.0, 2.0) == 2.0 && mc(-1.5, -1.5) == -1.5 && mc(1.0, -1.0) == 0.0 && mc(0.0, 1.0) == 0.0
end

function snapshot_with_halo(field, mesh)
    fill_halo!(field, ZeroGradient())
    return Dict(block.key => copy(field.blocks[b]) for (b, block) in pairs(mesh.blocks))
end

integral(mesh, field) = sum(sum(@view field.blocks[b][(mesh.halo + 1):(mesh.halo + mesh.block_size[1]), (mesh.halo + 1):(mesh.halo + mesh.block_size[2])]) * AA.cell_volume(mesh, b) for b in 1:nblocks(mesh))
absintegral(mesh, field) = sum(sum(abs, @view field.blocks[b][(mesh.halo + 1):(mesh.halo + mesh.block_size[1]), (mesh.halo + 1):(mesh.halo + mesh.block_size[2])]) * AA.cell_volume(mesh, b) for b in 1:nblocks(mesh))

function run_case(name, mesh, seed)
    rng = MersenneTwister(seed)
    h = mesh.halo
    nx, ny = mesh.block_size
    noise = CellField(mesh, (x, y) -> 0.0)
    for b in 1:nblocks(mesh)
        noise.blocks[b] .= randn(rng, size(noise.blocks[b]))
    end
    const_field = CellField(mesh, (x, y) -> 3.25)
    affine = CellField(mesh, (x, y) -> 1.0 + 2.0x - 3.0y)
    workspace = AMRWorkspace(mesh, noise, const_field, affine)
    worst = Dict(:int => 0.0, :replica => 0.0, :merge => 0.0, :kept => 0.0, :bound => 0.0, :const => 0.0, :affine => 0.0, :affine_wall => 0.0)
    nchild = nbound = naff = naffwall = 0
    for round in 1:6
        labels = AA.random_labels(rng, mesh; pr = round <= 2 ? 0.6 : 0.3, pc = round <= 2 ? 0.1 : 0.6)
        old = snapshot_with_halo(noise, mesh)
        I0, A0 = integral(mesh, noise), absintegral(mesh, noise)
        for (b, block) in pairs(mesh.blocks), j in 1:ny, i in 1:nx       # exact affine / constant cell averages before every regrid
            x, y = cellcenter(mesh, b, i, j)
            affine.blocks[b][i + h, j + h] = 1.0 + 2.0x - 3.0y
            const_field.blocks[b][i + h, j + h] = 3.25
        end
        fill_halo!(const_field, ZeroGradient())
        old_affine = snapshot_with_halo(affine, mesh)
        old_affine_keys = Set(block.key for block in mesh.blocks)
        stats = adapt!(workspace, labels; step = round, force = true)
        stats.changed || continue
        worst[:int] = max(worst[:int], abs(integral(mesh, noise) - I0) / A0)
        for (b, block) in pairs(mesh.blocks)
            key = block.key
            vals = noise.blocks[b]
            if haskey(old, key)
                worst[:kept] = max(worst[:kept], maximum(abs, vals[(h + 1):(h + nx), (h + 1):(h + ny)] .- old[key][(h + 1):(h + nx), (h + 1):(h + ny)]))
            elseif key.level > 0 && haskey(old, AA.parent_key(key))
                P = old[AA.parent_key(key)]
                ci, cj = mod(key.i, 2), mod(key.j, 2)
                for j in 1:ny, i in 1:nx
                    fi, fj = ci * nx + i, cj * ny + j
                    pi, pj = (fi + 1) ÷ 2 + h, (fj + 1) ÷ 2 + h
                    v = P[pi, pj]
                    sx, sy = mc(v - P[pi - 1, pj], P[pi + 1, pj] - v), mc(v - P[pi, pj - 1], P[pi, pj + 1] - v)
                    child = v + (isodd(fi) ? -0.25 : 0.25) * sx + (isodd(fj) ? -0.25 : 0.25) * sy
                    worst[:replica] = max(worst[:replica], abs(vals[i + h, j + h] - child))
                    lo, hi = extrema((v, P[pi - 1, pj], P[pi + 1, pj], P[pi, pj - 1], P[pi, pj + 1]))
                    worst[:bound] = max(worst[:bound], max(lo - vals[i + h, j + h], vals[i + h, j + h] - hi, 0.0))
                    nbound += 1
                    nchild += 1
                    worst[:const] = max(worst[:const], abs(const_field.blocks[b][i + h, j + h] - 3.25))
                    # affine
                    x, y = AA_cell_center(mesh, b, i, j)
                    err = abs(affine.blocks[b][i + h, j + h] - (1.0 + 2.0x - 3.0y))
                    # is the parent stencil exact for the affine data as read through the halo (uniform neighbourhood)?
                    hc = block.spacing
                    xp = isodd(fi) ? x + hc[1] / 2 : x - hc[1] / 2
                    yp = isodd(fj) ? y + hc[2] / 2 : y - hc[2] / 2
                    PA = old_affine[AA.parent_key(key)]
                    exact_stencil = all(abs(PA[pi + di, pj + dj] - (1.0 + 2.0 * (xp + di * 2hc[1]) - 3.0 * (yp + dj * 2hc[2]))) < 1.0e-12 for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1)))
                    if exact_stencil
                        worst[:affine] = max(worst[:affine], err); naff += 1
                    else
                        worst[:affine_wall] = max(worst[:affine_wall], err); naffwall += 1
                    end
                end
            else
                kids = AA.child_keys(key)
                for j in 1:ny, i in 1:nx
                    total = 0.0
                    for fj in (2j - 1):2j, fi in (2i - 1):2i
                        child = kids[(fi - 1) ÷ nx + 2((fj - 1) ÷ ny) + 1]
                        total += old[child][mod(fi - 1, nx) + 1 + h, mod(fj - 1, ny) + 1 + h]
                    end
                    worst[:merge] = max(worst[:merge], abs(vals[i + h, j + h] - total / 4))
                    x, y = AA_cell_center(mesh, b, i, j)
                    worst[:affine] = max(worst[:affine], abs(affine.blocks[b][i + h, j + h] - (1.0 + 2.0x - 3.0y)))
                end
            end
        end
    end
    println(
        rpad(name * " seed $seed", 20), " children=", nchild, "  ∫ change/∫|a| = ", worst[:int], "  replica = ", worst[:replica], "  merge = ", worst[:merge],
        "  kept = ", worst[:kept], "  bound violation = ", worst[:bound], "  const = ", worst[:const], "  affine (exact stencil, n=", naff, ") = ", worst[:affine],
        "  affine (stencil through an injected halo / wall / seam, n=", naffwall, ") = ", worst[:affine_wall]
    )
    @test worst[:int] <= 1.0e-14 && worst[:replica] == 0.0 && worst[:merge] <= 1.0e-15 && worst[:kept] == 0.0 && worst[:bound] == 0.0
    @test worst[:const] <= 1.0e-15 && worst[:affine] <= 1.0e-13 && naff > 0
    return worst
end

AA_cell_center(mesh, b, i, j) = cellcenter(mesh, b, i, j)

@testset "P2-P5 cell transfer (composite random meshes)" begin
    wall_errors = Float64[]
    for seed in 1:3
        run_case("walled", AA.walled_mesh(), seed)
        w = run_case("periodic", AA.periodic_mesh(), 10 + seed)
        w2 = run_case("channel", AA.channel_mesh(), 20 + seed)
        push!(wall_errors, w2[:affine_wall])
    end
    # parents whose stencil reads a piecewise-constant halo (coarser neighbour), a zero-gradient wall halo or the periodic seam: first order, error O(gradient * h)
    @test maximum(wall_errors) > 1.0e-3
end

# P6: order of the prolongation of smooth data on a uniform mesh (every block refined once, then the error of the child against the exact cell average)
@testset "P6 order of limited prolongation (smooth data, exact cell averages)" begin
    # periodic smooth data f = sin(2 pi x + 0.3) cos(4 pi y - 0.2) + 2 (the periodic seam must not break smoothness)
    # cell average of f over [x-a,x+a]x[y-b,y+b], separable closed form
    avg(x, y, hx, hy) = (cos(2pi * (x - hx / 2) + 0.3) - cos(2pi * (x + hx / 2) + 0.3)) / (2pi * hx) * (sin(4pi * (y + hy / 2) - 0.2) - sin(4pi * (y - hy / 2) - 0.2)) / (4pi * hy) + 2
    errs = Float64[]; maxerrs = Float64[]
    for n in (16, 32, 64, 128)
        mesh = BlockTreeMesh(; base_mesh = (n, n), block_size = (8, 8), extent = (1.0, 1.0), periodic = (true, true), max_level = 1)
        field = CellField(mesh, (x, y) -> 0.0)
        for (b, block) in pairs(mesh.blocks), j in 1:8, i in 1:8
            x, y = cellcenter(mesh, b, i, j)
            field.blocks[b][i + mesh.halo, j + mesh.halo] = avg(x, y, block.spacing[1], block.spacing[2])
        end
        workspace = AMRWorkspace(mesh, field)
        adapt!(workspace, fill(:refine, nblocks(mesh)); step = 1, force = true)
        err = sumsq = 0.0
        ncell = 0
        for (b, block) in pairs(mesh.blocks), j in 1:8, i in 1:8
            x, y = cellcenter(mesh, b, i, j)
            e = abs(field.blocks[b][i + mesh.halo, j + mesh.halo] - avg(x, y, block.spacing[1], block.spacing[2]))
            err = max(err, e); sumsq += e^2; ncell += 1
        end
        push!(errs, sqrt(sumsq / ncell)); push!(maxerrs, err)
    end
    rates = log2.(errs[1:(end - 1)] ./ errs[2:end])
    println("P6 RMS error of the limited prolongation of smooth periodic data at n = 16, 32, 64, 128: ", errs, "; observed order: ", rates, "\n   max-norm error (depends on whether cell centres align with the extrema, no clean rate): ", maxerrs)
    overall = log2(errs[1] / errs[end]) / (length(errs) - 1)
    println("   overall observed order n = 16 -> 128: ", overall)
    @test overall > 1.8   # limited linear: second order on average (individual steps scatter: at extremum cells the slope is zero, error h^2 u''/8, alignment-dependent)
end

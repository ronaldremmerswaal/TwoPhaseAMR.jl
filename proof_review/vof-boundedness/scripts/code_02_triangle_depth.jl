# Code check 02: the cut-donor example with donating depth 1/sqrt(2) at the bound (Section 5.5), its realisability in the package,
# its sharpness (the depth can approach 1), and the depth for consistent straight walls.
#
# Label:  none (example inside the paragraph "Boundedness." of Section 5.5)
# Source: docs/techref/chapters/05_vof_transport.tex:315-322: "in cut donors d can exceed 1/2: if the open part of the donor is the
#         triangle below its anti-diagonal and the neighbour behind the face is uncut, a_f = 1/2 and the open area of the full-width
#         rectangle is d^2/2, so nu_out = nu_ext = 1/2 gives d = 1/sqrt(2)"; docs/techref/figures/vof.jl `triangle_depth_example() = sqrt(0.5)`
#         (a closed form only, not a check against the package).
# Claims: (i)   area of the strip of depth d over the open triangle is d^2/2; the shared face has trace 0 (donor) and 1 (neighbour),
#               so the package's aperture is 1/2 (rule of `_interior_open_aperture`);
#         (ii)  at stable_timestep (nu_out = nu_ext = 1/2 simultaneously) the package's flux for this face equals the independent
#               evaluation with depth 1/sqrt(2); the package step is then within bounds;
#         (iii) 1/sqrt(2) is not extremal: the same construction with the open triangle in the far corner (legs eps) gives
#               depth 1 - eps + eps/sqrt(2) -> 1 as eps -> 0 (apertures 1/2 again);
#         (iv)  for a geometry that is consistent across cells (one straight wall through donor and neighbours) a random search reports how
#               often and how deep the rectangle at the stable step exceeds 1/2.
# Method: a 4x4 mesh whose cell [1,2]x[1,2] gets the stored solid plane by hand (the package's own aperture rule and area functions
#         are then used, nothing else is edited); for (iv) random straight walls on a 6x6 mesh with random velocities.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/vof-boundedness/scripts/code_02_triangle_depth.jl
#         (from the repository root)

using TwoPhaseAMR
using Test, Random, Printf
const AMR = TwoPhaseAMR
isdefined(Main, :VofAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "vof_audit.jl"))
using .VofAudit
const D = VofAudit.D
const G = VofAudit.Docs.Geo

"""
4x4 unit mesh (one block), no solid except the cell [1,2]x[1,2], whose stored solid plane is `solid = {n . xi <= a}` (local coordinates) with
solid fraction `alpha`; liquid level set `liquid`; x-velocity 1 on the high-x face of the centre cell only.
"""
function hand_made_donor(n, a, alpha; liquid)
    mesh = BlockTreeMesh(; base_mesh = (4, 4), block_size = (4, 4), extent = (4.0, 4.0), periodic = (false, false))
    geometry = GeometryField(mesh, (x, y) -> y + 10.0; lipschitz = 1)          # no solid anywhere (phi > 0 is fluid) ...
    donor = CellRef(1, 2, 2)
    AMR._set!(geometry.solid.fraction, alpha, donor)                             # ... except the centre cell, by hand
    AMR._set!(geometry.solid.normal_x, n[1], donor); AMR._set!(geometry.solid.normal_y, n[2], donor)
    AMR._set!(geometry.solid.intercept, a, donor)
    AMR._set!(geometry.open_fraction, 1 - alpha, donor)
    for (index, face) in pairs(mesh.faces)                                       # the package's own aperture rule
        geometry.aperture[index] = AMR._interior_open_aperture(geometry.solid, face)
    end
    field = InterfaceField(mesh, liquid; geometry, lipschitz = 1)
    reconstruct!(field)
    velocity = FaceVelocity(mesh)
    index = findfirst(f -> f.minus == donor && f.axis == 1, mesh.faces)
    velocity.interior[index] = 1.0
    return field, velocity, donor, index
end

function analyse(field, velocity, donor, index)
    mesh = field.mesh
    face = mesh.faces[index]
    work = WYAdvectionWorkspace(field)
    dt = stable_timestep(velocity, work)
    open_volume = D.open_volume_of(field, donor)
    aperture = D.aperture_of(field, index)
    volume = dt * velocity.interior[index] * face.area * aperture
    nu_out = abs(volume) / open_volume                                           # donor has no other flux
    nu_ext = dt * abs(velocity.interior[index]) / mesh.blocks[donor.block].spacing[1]
    region = D.independent_donating_region(field, face, velocity.interior[index], dt, volume)
    AMR._compute_directional_wy_fluxes!(field, velocity, dt, work, 1)
    # the package's own area function at the independent depth
    lo, hi = D.tangent_interval(mesh, donor, face)
    area_code = AMR._open_subcell_area(field.geometry.solid, donor, 0.5 - region.depth, 0.5, lo, hi)
    return (;
        dt, aperture, nu_out, nu_ext, depth = region.depth, area_code, target = abs(volume) / D.cell_volume(mesh, donor),
        flux_code = work.liquid_flux[index], flux_independent = region.liquid, open_fraction = D.open_fraction_of(field, donor),
        cond = D.cond(field, donor), work, volume,
    )
end

@testset "triangle example (claim 2)" begin
    @testset "(i) strip area d^2/2 by polygon clipping" begin
        worst = 0.0
        for d in range(0.0, 1.0; length = 101)
            strip = D.donating_rectangle(1, +1, d, -0.5, 0.5)
            open = G.clip_halfplane(strip, (1 / sqrt(2), 1 / sqrt(2)), 0.0; keep_less = true)   # open = {xi + eta <= 0}
            worst = max(worst, abs(G.shoelace(open) - d^2 / 2))
        end
        @test worst <= 1.0e-15
        # the closed form of the example
        @test isapprox(sqrt(2 * 0.25), 1 / sqrt(2); atol = 1.0e-15)          # d^2/2 = a_f nu = 1/4
    end
    @testset "(ii) the package on the hand-made triangle donor" begin
        field, velocity, donor, index = hand_made_donor((-1 / sqrt(2), -1 / sqrt(2)), 0.0, 0.5; liquid = (x, y) -> x - 1.45)
        r = analyse(field, velocity, donor, index)
        println(
            "(ii) aperture ", r.aperture, ", open fraction ", r.open_fraction, ", liquid fraction ", r.cond, ", dt ", r.dt,
            ", nu_out ", r.nu_out, ", nu_ext ", r.nu_ext, ", depth ", r.depth, " (1/sqrt 2 = ", 1 / sqrt(2), ")"
        )
        println(
            "     strip open area (package) at that depth ", r.area_code, " vs target ", r.target,
            "; liquid flux package ", r.flux_code, " independent ", r.flux_independent
        )
        @test r.aperture == 0.5
        @test isapprox(r.open_fraction, 0.5; atol = 1.0e-15) && 0 < r.cond < 1
        @test isapprox(r.nu_out, 0.5; atol = 1.0e-12) && isapprox(r.nu_ext, 0.5; atol = 1.0e-12)
        @test isapprox(r.depth, 1 / sqrt(2); atol = 1.0e-12)
        @test isapprox(r.area_code, r.target; atol = 1.0e-12)
        @test isapprox(r.flux_code, r.flux_independent; atol = 1.0e-14)
        # the full step stays within bounds (the package would raise an error otherwise)
        advect!(field, velocity, r.dt, r.work)
        @test all(0 <= D.cond(field, ref) <= 1 for ref in D.all_cells(field.mesh))
    end
    @testset "(iii) the depth can approach 1: open triangle in the far corner" begin
        depths = Float64[]
        for eps in (0.5, 0.3, 0.2, 0.1, 0.05, 0.02)
            # open = {xi + eta <= -1 + eps}  <=>  solid = {-(xi + eta)/sqrt2 <= (1 - eps)/sqrt2}
            field, velocity, donor, index = hand_made_donor(
                (-1 / sqrt(2), -1 / sqrt(2)), (1 - eps) / sqrt(2), 1 - eps^2 / 2;
                liquid = (x, y) -> x - (1.0 + 0.55 * eps)
            )
            r = analyse(field, velocity, donor, index)
            expected = 1 - eps + eps / sqrt(2)
            push!(depths, r.depth)
            println(
                "(iii) eps = ", eps, ": aperture ", r.aperture, ", open ", r.open_fraction, ", nu_out ", r.nu_out,
                ", nu_ext ", r.nu_ext, ", depth ", r.depth, " expected ", expected
            )
            @test r.aperture == 0.5
            @test isapprox(r.nu_out, 0.5; atol = 1.0e-9)
            @test isapprox(r.depth, expected; atol = 1.0e-9)
            @test isapprox(r.area_code, r.target; atol = 1.0e-12)
            @test isapprox(r.flux_code, r.flux_independent; atol = 1.0e-14)
        end
        @test issorted(depths) && depths[end] > 0.97
    end
    @testset "(iv) straight walls (consistent geometry): depth at the bound" begin
        rng = MersenneTwister(2718)
        mesh = BlockTreeMesh(; base_mesh = (6, 6), block_size = (6, 6), extent = (6.0, 6.0), periodic = (false, false))
        deepest = 0.0; best = nothing; donors = 0; beyond_half = 0
        depth_histogram = zeros(Int, 10)
        for trial in 1:3000
            # half of the walls are nearly grid-aligned (|angle - k pi/2| < 0.05): that is where the traces of two cells can differ most
            theta = trial % 2 == 0 ? 2pi * rand(rng) : (pi / 2) * rand(rng, 0:3) + 0.1 * (rand(rng) - 0.5)
            n = (cos(theta), sin(theta))
            offset = 3.0 * n[1] + 3.0 * n[2] + (rand(rng) - 0.5) * 3.0
            geometry = GeometryField(mesh, (x, y) -> n[1] * x + n[2] * y - offset; lipschitz = 1)
            field = InterfaceField(mesh, (x, y) -> hypot(x - 3, y - 3) - 1.7; geometry, lipschitz = 1)
            reconstruct!(field)
            velocity = VofAudit.random_velocity(rng, field; noise = 0.7)
            work = WYAdvectionWorkspace(field)
            dt = stable_timestep(velocity, work)
            isfinite(dt) || continue
            for (index, face) in pairs(mesh.faces)
                u = velocity.interior[index]
                iszero(u) && continue
                donor = u >= 0 ? face.minus : face.plus
                0 < D.solid_fraction_of(field, donor) < 1 || continue
                volume = dt * u * face.area * D.aperture_of(field, index)
                volume == 0 && continue
                lo, hi = D.tangent_interval(mesh, donor, face)
                depth = D.independent_depth(field, donor, face.axis, u >= 0 ? 1 : -1, lo, hi, abs(volume) / D.cell_volume(mesh, donor))
                isfinite(depth) || continue
                donors += 1
                depth > 0.5 && (beyond_half += 1)
                depth_histogram[min(10, floor(Int, 10 * depth) + 1)] += 1
                if depth > deepest
                    deepest = depth
                    other = u >= 0 ? face.plus : face.minus
                    side = u >= 0 ? 1 : -1
                    trace(ref, sd) = AMR._cell_face_open_fraction(field.geometry.solid, ref, face, sd)
                    best = (;
                        trial, theta, offset, aperture = D.aperture_of(field, index), donor_open = D.open_fraction_of(field, donor),
                        other_open = D.open_fraction_of(field, other), trace_donor = trace(donor, side), trace_other = trace(other, -side),
                        depth, nu_ext = dt * abs(u) / D.cell_spacing(mesh, donor)[face.axis],
                        liquid_donor = D.cond(field, donor), solid_alpha_donor = D.solid_fraction_of(field, donor),
                        solid_alpha_other = D.solid_fraction_of(field, other),
                    )
                end
            end
        end
        println(
            "(iv) ", donors, " cut-donor faces from random straight walls at the stable step; depth > 1/2 on ", beyond_half,
            " of them; histogram of the depth in tenths: ", depth_histogram
        )
        println("     deepest rectangle: ", best)
        @test donors > 3000
        @test deepest < 1            # (whether a depth above 1/2 occurs for straight walls is reported above)
    end
end

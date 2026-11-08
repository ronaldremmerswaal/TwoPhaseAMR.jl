# Step 01 (symbolic): the depth of a subface rectangle without a solid does not depend on the phase aperture, and the two
# subface rectangles of equal velocity tile the face rectangle.
#
# Label:  eq:vof:frozen-flux  (with eq:vof:subface, eq:vof:depth)
# Source: docs/techref/chapters/05_vof_transport.tex:600-634 ("otherwise d = dt |u^phi|/h"; "the two subface rectangles tile
#         the one-velocity rectangle"); code src/two_phase/two_velocity/interface_transport.jl, _frozen_subface_swept_fraction
#         (distance = target / transverse_width) and _frozen_subface_interval.
# Checks (Symbolics.jl): with the face segment of length |f| = w h_t (donor tangential cell width h_t, local width w), the donor
#         volume V = h_k h_t, phase aperture a, phase velocity u:
#   S1  target = dt |u| |f| a / V  and  transverse width = a w  give  distance = target / (a w) = dt |u| / h_k  (any a > 0);
#   S2  the phase volumes add up to the one-velocity carrier: dt |f| (a_l + a_g) u = dt |f| u  iff  a_l + a_g = 1;
#   S3  the subface widths a_l w and a_g w with a_l + a_g = 1 tile the segment [lo, lo + w] without gap or overlap,
#       whichever phase sits at the lower end;
#   S4  the rectangle areas (distance * width, in local units) add to the one-velocity rectangle area w * dt|u|/h_k.
#   S5  if a_l + a_g = 1 + e (rounding), the tiles overlap (e < 0: gap) by e w : the reduction holds up to round-off only.
# Run:    julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/eq-vof-frozen-flux/scripts/step01_tiling_symbolic.jl

using Symbolics, Test

@variables dt u f_len hk ht a al ag w lo e
expanded(x) = Symbolics.simplify(Symbolics.expand(x))
iszero_sym(x) = isequal(expanded(x), 0)

V = hk * ht
wloc = f_len / ht                                       # local width of the segment, w = |f| / h_t

@testset "eq:vof:frozen-flux tiling (symbolic)" begin
    @testset "S1 depth independent of the phase aperture" begin
        target = dt * u * f_len * a / V
        distance = target / (a * wloc)
        @test iszero_sym(distance - dt * u / hk)
    end
    @testset "S2 phase volumes add to the one-velocity carrier iff a_l + a_g = 1" begin
        total = dt * f_len * (al + ag) * u
        @test iszero_sym(total - dt * f_len * u - dt * f_len * u * ((al + ag) - 1))
        @test iszero_sym(substitute(total, Dict(ag => 1 - al)) - dt * f_len * u)
    end
    @testset "S3 the subface intervals tile the segment" begin
        # liquid at the lower end: [lo, lo + al w], gas [lo + al w, lo + w] when ag = 1 - al
        agv = 1 - al
        @test iszero_sym((lo + al * w) - (lo + al * w))                       # shared end point
        @test iszero_sym((lo + w) - (lo + al * w) - agv * w)                  # gas width = w - al w
        # gas at the lower end: [lo, lo + ag w], liquid [lo + w - al w, lo + w]
        @test iszero_sym((lo + w - al * w) - (lo + agv * w))                  # the two ends meet
    end
    @testset "S4 areas add" begin
        depth = dt * u / hk
        @test iszero_sym(depth * (al * w) + depth * ((1 - al) * w) - w * depth)
    end
    @testset "S5 rounding: a_l + a_g = 1 + e leaves a gap/overlap of e w" begin
        agv = 1 + e - al
        gap = (lo + w) - (lo + al * w) - agv * w       # remaining length after the two tiles
        @test iszero_sym(gap + e * w)
    end
end

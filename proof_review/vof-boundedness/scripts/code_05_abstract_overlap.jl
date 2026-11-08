# Code check 05: can donating rectangles overlap in the open part of a cut donor, using ONLY convexity of the open part and the outgoing
# bound?  (abstract single-cell model, no mesh, no aperture rule; polygon clipping only)
#
# Label:  none (paragraph "Boundedness." of Section 5.5)
# Source: docs/techref/chapters/05_vof_transport.tex:357-361 ("For a cut coarse donor with two segments on a side neither argument applies")
# Model:  one donor cell [-1/2,1/2]^2 with open part O = cell cap {n . xi >= a} (one solid half-plane); x sweep; rectangles
#         R_i(d) = strip of depth d from the side of segment i over its tangential range; carrier volumes tau_i >= 0 free, subject only to
#         the outgoing bound  sum_i tau_i <= (1/2)|O|  and to the existence of the depth (tau_i <= |O cap R_i(1)|); depth by bisection of
#         |O cap R_i(d)| = tau_i.  Configurations: (A) right side two half-width segments + left single face; (B) both sides two segments;
#         (C) single faces on both sides (control, proven disjoint).
# Question: does an overlap |O cap R_i cap R_j| > 0 of rectangles of opposite sides occur?  And, if so, can a half-plane liquid region
#         inside O be double counted so that its outflow exceeds its volume (violation of C >= 0 for chi = 0)?
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/vof-boundedness/scripts/code_05_abstract_overlap.jl
#         (from the repository root; uses only the polygon helpers of proof_review/_shared/techref_geometry.jl)

using Test, Random, Printf
isdefined(Main, :Geo) || include(joinpath(@__DIR__, "..", "..", "_shared", "techref_geometry.jl"))
using .Geo

const SQ = rect_poly(-0.5, 0.5, -0.5, 0.5)

open_polygon(n, a) = clip_halfplane(SQ, n, a; keep_less = false)        # open part = {n . xi >= a}

"""Rectangle of depth `d` on side `s` (+1: x = +1/2, -1: x = -1/2) over the tangential range [lo, hi]."""
rect(s, d, lo, hi) = s > 0 ? rect_poly(0.5 - d, 0.5, lo, hi) : rect_poly(-0.5, -0.5 + d, lo, hi)

open_area(O, s, d, lo, hi) = begin
    poly = O
    for (xlo, xhi, ylo, yhi) in ((s > 0 ? 0.5 - d : -0.5, s > 0 ? 0.5 : -0.5 + d, lo, hi),)
        poly = clip_halfplane(poly, (1.0, 0.0), xlo; keep_less = false)
        poly = clip_halfplane(poly, (1.0, 0.0), xhi; keep_less = true)
        poly = clip_halfplane(poly, (0.0, 1.0), ylo; keep_less = false)
        poly = clip_halfplane(poly, (0.0, 1.0), yhi; keep_less = true)
    end
    shoelace(poly)
end

function depth_for(O, s, lo, hi, tau)
    a1 = open_area(O, s, 1.0, lo, hi)
    tau <= a1 * (1 - 1.0e-12) || return Inf
    lo_d, hi_d = 0.0, 1.0
    for _ in 1:60
        mid = (lo_d + hi_d) / 2
        open_area(O, s, mid, lo, hi) < tau ? (lo_d = mid) : (hi_d = mid)
    end
    return (lo_d + hi_d) / 2
end

"""Segments (side, lo, hi) of a configuration."""
function segments(config)
    config == :A && return [(+1, -0.5, 0.0), (+1, 0.0, 0.5), (-1, -0.5, 0.5)]
    config == :B && return [(+1, -0.5, 0.0), (+1, 0.0, 0.5), (-1, -0.5, 0.0), (-1, 0.0, 0.5)]
    return [(+1, -0.5, 0.5), (-1, -0.5, 0.5)]                              # :C
end

function overlap_open(O, segs, depths)
    best = 0.0
    for i in eachindex(segs), j in (i + 1):length(segs)
        si, loi, hii = segs[i]; sj, loj, hij = segs[j]
        si == sj && continue
        lo, hi = max(loi, loj), min(hii, hij)
        hi > lo || continue
        (isfinite(depths[i]) && isfinite(depths[j])) || continue
        # intersection of the two strips: x in [0.5 - d_right, -0.5 + d_left]
        dr = si > 0 ? depths[i] : depths[j]; dl = si > 0 ? depths[j] : depths[i]
        xlo, xhi = 0.5 - dr, -0.5 + dl
        xhi > xlo || continue
        poly = O
        poly = clip_halfplane(poly, (1.0, 0.0), xlo; keep_less = false); poly = clip_halfplane(poly, (1.0, 0.0), xhi; keep_less = true)
        poly = clip_halfplane(poly, (0.0, 1.0), lo; keep_less = false); poly = clip_halfplane(poly, (0.0, 1.0), hi; keep_less = true)
        best = max(best, shoelace(poly))
    end
    return best
end

@testset "abstract overlap search (convex open part, outgoing bound only)" begin
    rng = MersenneTwister(31337)
    for config in (:C, :A, :B)
        samples = overlaps = refused = violations = 0
        max_overlap = 0.0
        worst_violation = 0.0
        example = nothing
        for trial in 1:60000
            theta = 2pi * rand(rng)
            n = (cos(theta), sin(theta))
            a = (rand(rng) - 0.5) * 1.2
            O = open_polygon(n, a)
            length(O) >= 3 || continue
            area = shoelace(O)
            0.02 < area < 0.98 || continue
            segs = segments(config)
            w = -log.(rand(rng, length(segs)))
            w ./= sum(w)
            # a random subset of the faces donate
            active = [rand(rng) < 0.85 for _ in segs]
            any(active) || continue
            w = w .* active
            tau = w ./ sum(w) .* (0.5 * area) .* rand(rng)
            depths = [active[k] ? depth_for(O, segs[k][1], segs[k][2], segs[k][3], tau[k]) : 0.0 for k in eachindex(segs)]
            if any(!isfinite, depths)
                refused += 1
                continue
            end
            samples += 1
            ov = overlap_open(O, segs, depths)
            if ov > 1.0e-10
                overlaps += 1
                max_overlap = max(max_overlap, ov)
                example === nothing && (example = (; theta, a, tau, depths, ov))
                # liquid half-plane regions inside O: try to double count (chi = 0: liquid area <= area / 2)
                for _ in 1:40
                    th = 2pi * rand(rng); nl = (cos(th), sin(th))
                    lo_, hi_ = extrema(nl[1] * p[1] + nl[2] * p[2] for p in O)
                    al = lo_ + rand(rng) * (hi_ - lo_)
                    L = clip_halfplane(O, nl, al; keep_less = true)
                    length(L) >= 3 || continue
                    vol = shoelace(L)
                    vol <= 0.5 * area || continue
                    out = 0.0
                    for k in eachindex(segs)
                        active[k] || continue
                        s, lo, hi = segs[k]
                        d = depths[k]
                        poly = clip_halfplane(L, (1.0, 0.0), s > 0 ? 0.5 - d : -0.5; keep_less = false)
                        poly = clip_halfplane(poly, (1.0, 0.0), s > 0 ? 0.5 : -0.5 + d; keep_less = true)
                        poly = clip_halfplane(poly, (0.0, 1.0), lo; keep_less = false)
                        poly = clip_halfplane(poly, (0.0, 1.0), hi; keep_less = true)
                        out += shoelace(poly)
                    end
                    if out - vol > 1.0e-10
                        violations += 1
                        worst_violation = max(worst_violation, (out - vol) / area)
                    end
                end
            end
        end
        @printf("config %s: %6d feasible samples (%d refused by the strip condition), overlap in the open part in %d (max area %.3e), double-counting violations %d (worst %.3e of |O|)\n",
            config, samples, refused, overlaps, max_overlap, violations, worst_violation)
        example === nothing || println("   first overlapping example: ", example)
        if config == :C
            @test overlaps == 0                       # proven: single-face sides, full-width rectangles, outgoing bound
        end
        @test samples > 10000
    end
end

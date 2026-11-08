# Step 02: the telescoping identity behind the symmetry (Symbolics.jl).
#
# Label:  eq:setting:patch-metric
# Claim:  03_discrete_setting.tex:304-305; mechanism of the proof.
# Statement proved here (with hx = r*hy, s = cell HEIGHT, g(s) = s^2/6):
#   (i)   for two cells that share a corner at v and have sides (a, b) with b/a in {1/2, 1, 2},
#         the face segment between them (length min(a,b), tangential centre offset (a-b)/2) has
#             |f| * d_tangential = min(a,b)(a-b)/2 = g(a) - g(b);
#   (ii)  the four half-lines from v give  Sx = sum_{axis 2} |f| d_x = r^2 [g(NE)-g(SE) + g(SW)-g(NW)],
#                                          Sy = sum_{axis 1} |f| d_y =     [g(NE)-g(NW) + g(SW)-g(SE)],
#         hence 2(T12 - T21) = Sx - Sy = (r^2 - 1) (g(NE) + g(SW) - g(NW) - g(SE)),
#         which vanishes for r = 1 and for every choice of the four sides (telescoping);
#   (iii) every one of the 4 half-line faces of all 16 corner patterns satisfies (i)/(ii)
#         when read off the face construction of model.jl (hx, hy symbolic).
# Run:    julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/eq-setting-patch-metric/scripts/step02_telescoping_symbolic.jl

using Symbolics
using Test
isdefined(Main, :PatchModel) || include(joinpath(@__DIR__, "model.jl"))
using .PatchModel

@variables a r hy gNE gNW gSW gSE

is_zero(x) = (v = Symbolics.value(Symbolics.simplify(Symbolics.expand(Num(x)))); v isa Number ? iszero(v) : false)

g(s) = s^2 / 6

@testset "step02: telescoping identity" begin
    @testset "(i) min(a,b)(a-b)/2 = (a^2-b^2)/6 for b/a in {1/2, 1, 2}" begin
        for (b, m) in ((a, a), (2a, a), (a / 2, a / 2))        # (b, min(a, b))
            @test is_zero(m * (a - b) / 2 - (a^2 - b^2) / 6)
        end
        # and it FAILS for a 4:1 contact (min = b, b = a/4): the identity needs the 2:1 grading
        b4, m4 = a / 4, a / 4
        @test !is_zero(m4 * (a - b4) / 2 - (a^2 - b4^2) / 6)
    end

    @testset "(ii) the cyclic sum" begin
        Sx = r^2 * ((gNE - gSE) + (gSW - gNW))
        Sy = (gNE - gNW) + (gSW - gSE)
        @test is_zero(Sx - Sy - (r^2 - 1) * (gNE + gSW - gNW - gSE))
        @test is_zero(Symbolics.substitute(Sx - Sy, Dict(r => 1)))       # r = 1: always zero
    end

    @testset "(iii) each half-line face of the 16 corner patterns, read off model.jl" begin
        n_faces = 0
        for sNE in (1, 2), sNW in (1, 2), sSW in (1, 2), sSE in (1, 2)
            sizes = Dict(:NE => sNE, :NW => sNW, :SW => sSW, :SE => sSE)
            cells = corner_cells(sNE, sNW, sSW, sSE)
            faces = faces_at_vertex(cells, 0, 0)
            @test length(faces) == 4
            for f in faces
                hl = half_line(f, 0, 0)
                # the two cells adjacent to this half-line, ordered (plus-side quadrant first
                # in the direction of increasing tangential coordinate, as in (ii))
                pair = hl == :E ? (:NE, :SE) : hl == :W ? (:SW, :NW) : hl == :N ? (:NE, :NW) : (:SW, :SE)
                sa, sb = sizes[pair[1]] * hy, sizes[pair[2]] * hy        # cell heights
                # physical |f| d_t with cell widths (r*hy, hy)
                wx, wy = r * hy, hy
                area = f.len * (f.axis == 1 ? wy : wx)
                d = (f.d2[1] * wx / 2, f.d2[2] * wy / 2)
                moment_t = f.axis == 2 ? area * d[1] : area * d[2]
                expected = (f.axis == 2 ? r^2 : 1) * (g(sa) - g(sb))
                @test is_zero(moment_t - expected)
                n_faces += 1
            end
        end
        @test n_faces == 64
    end
end

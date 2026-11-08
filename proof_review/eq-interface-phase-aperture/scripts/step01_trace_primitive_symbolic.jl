# Step 01: the segment-clipping primitive behind the one-sided traces (4.21), (7.4), and the partition lemma.
#
# Label:  eq:interface:phase-aperture / eq:interface:liquid-trace
# Claim:  04_interface.tex:563-590 and 07_cutcells.tex:90-104: the traces are lengths of the face segment inside the
#         open polygon (solid half-plane) and inside the liquid polygon; the phase apertures are 1/2 of the sums over the
#         two cells; liquid + gas = open.
# Code:   src/two_phase/forces.jl:313-324 (_clip_line_interval): on the segment s in [lo, hi] the condition
#         m s + k <= 0 (keep_less) or >= 0 gives the interval (lo, crossing) / (crossing, hi) with
#         crossing = lo + (hi - lo) v_lo / (v_lo - v_hi), v = m s + k.
# Checks (symbolic):
#   S1  crossing = -k/m, the root of the linear function (all cases of the code's branches);
#   S2  the length of the kept interval equals the measure of {s in [lo,hi] : m s + k <= 0}  in the four endpoint cases;
#   S3  partition lemma: with the liquid interval inside the open interval, open = liquid + gas (gas = open minus liquid):
#       intervals [a, hi] ⊃ [a, b] give hi - a = (b - a) + (hi - b), and the average over two cells keeps the identity;
#   S4  a covered neighbour removes the cell's term: the thesis' sum is (1/2) trace of the open cell (= 1/2 for a
#       pure open cell), the covered rule gives 0.
# Run:    julia --project=@claude-symbolics proof_review/eq-interface-phase-aperture/scripts/step01_trace_primitive_symbolic.jl

using Symbolics, Test

@variables lo hi m k tA tB
isz(e) = isequal(Symbolics.simplify(Symbolics.expand(e)), 0)

vlo = m * lo + k
vhi = m * hi + k

@testset "S1 the crossing of the code is the root of m s + k" begin
    crossing = lo + (hi - lo) * vlo / (vlo - vhi)
    @test isz(crossing + k / m)
end

@testset "S2 lengths of the kept intervals" begin
    crossing = -k / m
    # keep_less (m s + k <= 0):
    #   inside at lo only (v_lo <= 0 < v_hi, m > 0): interval (lo, crossing), length crossing - lo = -(k + m lo)/m = -v_lo/m
    @test isz((crossing - lo) + vlo / m)
    #   inside at hi only (v_hi <= 0 < v_lo, m < 0): interval (crossing, hi), length hi - crossing = v_hi/m (>= 0 for m < 0)
    @test isz((hi - crossing) - vhi / m)
    #   both inside: length hi - lo; neither inside: 0 (interval (lo, lo)).
    # keep_greater is the complement on the segment: lengths add up to hi - lo
    @test isz(((crossing - lo) + (hi - crossing)) - (hi - lo))
end

@testset "S3 partition lemma on one face segment: open = liquid + gas" begin
    # open interval [a, hi], liquid sub-interval [a, b] (a <= b <= hi), gas [b, hi]; traces divided by the face length hi - lo
    a, b = tA, tB
    open_len = hi - a; liquid_len = b - a; gas_len = hi - b
    @test isz(open_len - liquid_len - gas_len)
    # two cells (minus: a1, b1; plus: a2, b2): the half-sum of liquid and gas traces is the half-sum of open traces
    @variables a1 b1 a2 b2
    w = hi - lo
    half_liq = ((b1 - a1) + (b2 - a2)) / (2w)
    half_gas = ((hi - b1) + (hi - b2)) / (2w)
    half_open = ((hi - a1) + (hi - a2)) / (2w)
    @test isz(half_liq + half_gas - half_open)
end

@testset "S4 covered neighbour: thesis sum vs the covered rule" begin
    @variables t_open
    thesis = (0 + t_open) / 2           # covered cell contributes 0, the open cell its open trace
    code = 0                            # covered rule: a_f = 0
    @test isz(thesis - t_open / 2)
    @test !isz(thesis - code)
    # at a grid-aligned solid boundary the open cell is pure: trace 1, so the thesis' open aperture is 1/2
    @test isz(Symbolics.substitute(thesis, Dict(t_open => 1)) - 1 // 2)
end

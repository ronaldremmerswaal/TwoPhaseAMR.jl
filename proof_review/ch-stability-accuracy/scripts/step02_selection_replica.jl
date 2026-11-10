# Steps 5-6 of the audit of ch:stability-accuracy: the step-selection rule eq:stab:selection and the reference-step bookkeeping,
# on a literal replica of the loop of src/navier_stokes/simulate.jl (lines 90-166, 195-197), Float64.
#
# Label:  eq:stab:selection (ch:stability-accuracy)
# Source: docs/techref/chapters/13_stability_accuracy.tex:146-233 (selection, "last step", reference step);
#         src/navier_stokes/simulate.jl:90-91 (time_tolerance), 122-149 (proposal, stretched step, reference), 166-172 (time update)
# Checks:
#   R1  every step of a call but the last equals its proposal, the proposal never increases (retained cumulative minimum);
#   R2  the last step of a call that follows a step of the same call lies in [1e-6, 1 + 1e-6) times its proposal (random bounds, random targets,
#       remainders placed on both sides of the 1e-6 threshold); both ends are attained (extremal targets); the hypothesis "an earlier step of the
#       same call" is necessary (first step of a call with a retained proposal: ratio 1e-9);
#   R3  reference step ref = max(proposal, previous step, step) >= step always; the third term is the maximum only for a stretched last step
#       (step > proposal); a shortened last step has ref = max(proposal, previous step);
#   R4  prescribed dt: the proposal is dt at every step; same rules.
# Run:    julia proof_review/ch-stability-accuracy/scripts/step02_selection_replica.jl

using Test, Random

include(joinpath(@__DIR__, "_replica.jl"))

Random.seed!(1306)

@testset "step selection replica" begin
    @testset "R1-R3 random calls" begin
        stretched_third = 0
        shortened_last = 0
        lower_hits = 0
        for trial in 1:40_000
            t0 = rand() < 0.5 ? 0.0 : 10rand()
            scale = 10.0^(-3 + 3rand())
            # decreasing-or-noisy bounds: the proposal is their running minimum
            start = scale * (0.5 + rand())
            bounds = [start * max(0.2, 1 + 0.3randn()) for _ in 1:2000]
            if rand() < 0.2
                bounds[1:3] .= Inf
            end
            p_first = isfinite(bounds[1]) ? bounds[1] : NaN
            nsteps = rand(1:6)
            remainder = rand((0.0, 1.0e-7 * rand(), 1.0e-6 * (1 - 1.0e-3 * rand()), 1.0e-6 * (1 + 1.0e-3 * rand()), 2.0e-6 * rand(), rand(), 1 - 1.0e-9 * rand()))
            # target after `nsteps` proposals (computed with the running minimum) plus a fraction of the next one
            running = Inf
            acc = t0
            for n in 1:nsteps
                running = min(running, bounds[n])
                isfinite(running) || (running = (n == 1 ? scale : running))
                acc += running
            end
            nextp = min(running, bounds[nsteps + 1])
            isfinite(nextp) || (nextp = scale)
            T = acc + remainder * nextp
            T > t0 || continue
            call = replica_call(t0, T, n -> bounds[min(n, end)])
            rec = call.rec
            isempty(rec) && continue
            last = length(rec)
            # proposals: running minimum, never increasing
            props = getfield.(rec, :proposed)
            @test issorted(props; rev = true)
            # R1: every step but the last equals its proposal
            for n in 1:(last - 1)
                @test rec[n].step == rec[n].proposed
            end
            r = rec[last]
            ratio = r.step / r.proposed
            @test ratio < 1 + 1.0e-6 + 1.0e-9
            if last >= 2
                # R2 (tolerance for the rounding of the times: rel. 1e-9 of the proposal at times <= 10)
                @test ratio >= 1.0e-6 * (1 - 1.0e-3)
                ratio < 1.0e-6 * 1.001 && (lower_hits += 1)
            end
            # R3
            for x in rec
                @test x.reference >= x.step
                if x.step > max(x.proposed, x.defect)
                    @test x === r && x.step > x.proposed          # only a stretched last step
                    @test x.reference == x.step
                end
            end
            if r.step < r.proposed
                shortened_last += 1
                @test r.reference == max(r.proposed, r.defect)
            end
            if r.step > r.proposed && r.step > r.defect
                stretched_third += 1
                @test r.reference == r.step > r.proposed
            end
        end
        @test stretched_third > 0 && shortened_last > 0
        println(
            "random calls: stretched last steps with active third term ", stretched_third, ", shortened last steps ", shortened_last,
            ", last steps within 0.1 % of the lower end ", lower_hits
        )
    end

    @testset "R2 extremal ends and necessity of the hypothesis" begin
        p = 0.0125
        # remainder exactly 1e-6 p: a step of its own, ratio 1e-6 (lower end attained, up to the rounding of the times)
        call = replica_call(0.0, 3p + 1.0e-6 * p, n -> p)
        @test length(call.rec) == 4 || length(call.rec) == 3          # rounding of 3p + 1e-6 p decides which side of the threshold
        # remainder just below 1e-6 p: stretched, ratio 1 + 1e-6 (1 - 1e-3)
        call = replica_call(0.0, 3p + 0.999e-6 * p, n -> p)
        @test length(call.rec) == 3
        @test isapprox(call.rec[end].step / p, 1 + 0.999e-6; rtol = 1.0e-9)
        # remainder just above: a fourth step of ratio ~1e-6
        call = replica_call(0.0, 3p + 1.001e-6 * p, n -> p)
        @test length(call.rec) == 4
        @test isapprox(call.rec[end].step / p, 1.001e-6; rtol = 1.0e-6)
        # first step of a call with a retained proposal: the ratio can be arbitrarily small (hypothesis "earlier step of the same call" necessary)
        call = replica_call(0.0, 1.0e-9 * p, n -> Inf; retained = p)
        @test length(call.rec) == 1
        @test isapprox(call.rec[1].step / p, 1.0e-9; rtol = 1.0e-9)
        # a proposal that drops sharply inside the call: still >= 1e-6 (the argument uses p_n <= p_{n-1})
        bounds = n -> n == 1 ? p : 1.0e-3 * p
        call = replica_call(0.0, p + 2.0e-6 * p, bounds)
        @test call.rec[1].step == p
        @test call.rec[end].step / call.rec[end].proposed >= 1.0e-6 * (1 - 1.0e-9)
    end

    @testset "R4 prescribed dt and the 128 cap" begin
        dt = 0.01
        call = replica_call(0.0, 0.0345, n -> Inf; dt)
        @test all(x -> x.proposed == dt, call.rec) && length(call.rec) == 4
        @test isapprox(call.rec[end].step, 0.0045; rtol = 1.0e-9)
        # infinite bound: interval/128 and exactly 128 steps
        call = replica_call(0.0, 1.0, n -> Inf)
        @test length(call.rec) == 128 && all(x -> x.proposed == 1 / 128, call.rec)
        # infinite bound at a call that has a retained finite proposal: no cap
        call = replica_call(0.0, 1.0, n -> Inf; retained = 0.3)
        @test all(x -> x.proposed == 0.3, call.rec) && length(call.rec) == 4
    end

    @testset "R3 regrid and carried defects" begin
        p = 0.1
        # previous call's last step (0.03) is the defect of the first step; reference = max(proposal, defect, step)
        call = replica_call(0.0, 0.25, n -> p; defect0 = 0.03)
        @test call.rec[1].defect == 0.03 && call.rec[1].reference == p
        # a larger carried defect (previous call's last step 0.4 with a smaller dt now) is the reference
        call = replica_call(0.0, 0.25, n -> Inf; dt = 0.1, defect0 = 0.4)
        @test call.rec[1].reference == 0.4 && call.rec[2].reference == 0.1
        # after a regrid the defect becomes the proposal of the step just taken (untruncated)
        call = replica_call(0.0, 0.3, n -> p; regrid = n -> n == 1)
        @test call.rec[2].defect == p
        # a shortened step before a regrid: the defect is the (untruncated) proposal
        call = replica_call(0.0, 0.25, n -> p)
        @test isapprox(call.rec[end].step, 0.05; rtol = 1.0e-9) && call.rec[end].reference == p
    end
end

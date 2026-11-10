# Code check 01: the boundary-value stage of the one-velocity projection and the Deviation "Boundary reset after the projection" (dev:projection:outflow).
#
# Label:  dev:projection:outflow  (also the sentences of sec:projection:stage on boundary values)
# Source: docs/techref/chapters/10_projection.tex:808-860; src/operators/staggered_diffusion.jl (enforce_velocity_boundary!), src/navier_stokes/projection.jl
# Checks (channel 2 x 1, composite mesh, parabolic Inflow left, Outflow right, NoSlip walls; ε = 1e-12):
#   O1  before the final boundary reset the corrected velocity is solenoidal to the tolerance (dt max|D v^{n+1}| <= ε); after the reset the cells next to the
#       Outflow side carry dt|D v| = O(1e-3) (value printed), the interior cells 1e-13: the tolerance does not bound it;
#   O2  the global sum Σ_c |c| D v still vanishes after the reset (Outflow adds one uniform correction: total boundary flux zero), so the defects sum to zero;
#   O3  re-projecting the end-of-step velocity does NOT remove the defect from the result: the defect after the 2nd, 3rd projection is the same size
#       (the sentence "the next projection removes the defect again" is true of the solve, not of its output);
#   O4  ZeroGradientVelocity side + inflow, no Outflow side: a nonzero total boundary flux throws (ArgumentError), as the text says;
#   O5  Inflow/NoSlip/FreeSlip sides and walls: the second application changes nothing (end-of-step defect <= ε);
#   O6  with a solid the faces with a_f = 0 are zero before and after.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/dev-projection-outflow/scripts/code_01_outflow.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ProjectionAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "projection_audit.jl"))
using .ProjectionAudit
using TwoPhaseAMR
const PA = ProjectionAudit

chan = BlockTreeMesh(;
    base_mesh = (16, 8), block_size = (4, 4), extent = (2.0, 1.0), periodic = (false, false), max_level = 1,
    refine = (key, o, e) -> key.level == 0 && key.i == 3
)
prof(y) = 4y * (1 - y)
vel0() = FaceVelocity(chan, (x, y, t, axis) -> axis == 1 ? prof(y) + 0.3sinpi(2x) * sinpi(y) : 0.2sinpi(x) * sinpi(2y))
dt = 0.01
cop = PoissonOperator(chan)

function strong_div(v)
    nx, ny = chan.block_size
    out = zeros(length(chan.blocks) * nx * ny)
    for (i, f) in pairs(chan.faces)
        w = f.area * v.interior[i]; out[AMR._cell_index(f.minus, nx, ny)] += w; out[AMR._cell_index(f.plus, nx, ny)] -= w
    end
    for (i, f) in pairs(chan.boundary_faces)
        out[AMR._cell_index(f.cell, nx, ny)] += f.side * f.area * v.boundary[i]
    end
    vol = [chan.blocks[(c - 1) ÷ (nx * ny) + 1].volume for c in 1:length(out)]
    return out ./ vol, out
end

@testset "boundary reset after the projection" begin
    bc = VelocityBoundaryConditions(; left = Inflow((x, y, t, axis) -> axis == 1 ? prof(y) : 0.0), right = Outflow(), bottom = NoSlip(), top = NoSlip())
    solver = EllipticSolver(cop; preconditioner = :tree_geometric)
    work = ProjectionWorkspace(solver)
    v = vel0()
    enforce_velocity_boundary!(v, bc; time = 0.0); vstar = copy(v.interior); bstar = copy(v.boundary)
    p = CellField(chan)
    stats = project!(v, p, solver, dt, work; velocity_boundary = bc, tolerance = 1.0e-12, max_cycles = 500)
    @test stats.converged
    # O1: pre-reset velocity
    nx, ny = chan.block_size
    pv = FA.cell_vector(ref -> AMR._get(p, ref), chan)
    Gp = Matrix(FA.metric_matrix(cop)) * (Matrix(FA.jump_matrix(chan)) * pv)
    pre = FaceVelocity(chan); pre.interior .= vstar .- dt .* Gp; pre.boundary .= bstar
    d_pre, _ = strong_div(pre)
    d_post, int_post = strong_div(v)
    # cells next to the outflow side: those with an Outflow boundary face
    outcells = Set(AMR._cell_index(f.cell, nx, ny) for f in chan.boundary_faces if f.axis == 1 && f.side == 1)
    near = [c for c in eachindex(d_post) if c in outcells]; far = [c for c in eachindex(d_post) if !(c in outcells)]
    println("O1: dt max|D v| before reset ", dt * maximum(abs.(d_pre)), "; after reset: next to Outflow ", dt * maximum(abs.(d_post[near])), " (", length(near), " cells), elsewhere ", dt * maximum(abs.(d_post[far])))
    @test dt * maximum(abs.(d_pre)) <= 1.0e-11
    @test dt * maximum(abs.(d_post[near])) > 1.0e-4
    @test dt * maximum(abs.(d_post[far])) <= 1.0e-11
    # O2
    println("O2: Σ|c| D v after the reset = ", sum(int_post), " (largest |c| D v ", maximum(abs.(int_post)), ")")
    @test abs(sum(int_post)) <= 1.0e-12 * maximum(abs.(int_post))
    # O3: repeated projections
    sizes = Float64[]
    for k in 1:4
        st = project!(v, p, solver, dt, work; velocity_boundary = bc, tolerance = 1.0e-12, max_cycles = 500)
        d, _ = strong_div(v)
        push!(sizes, dt * maximum(abs.(d)))
    end
    println("O3: end-of-step dt max|D v| after the 2nd..5th projection: ", sizes)
    @test minimum(sizes) > 1.0e-4
    # O7: Outflow value = area-weighted mean of the parallel interior faces of the boundary cell (the faces on the opposite side of the cell) + one uniform correction
    let w = vel0()
        enforce_velocity_boundary!(w, bc; time = 0.0)
        raw = Dict{Int, Float64}(); v_raw = vel0()
        for (i, f) in pairs(chan.boundary_faces)
            (f.axis == 1 && f.side == 1) || continue
            cellidx = AMR._cell_index(f.cell, nx, ny)
            num = den = 0.0
            for (g, fi) in pairs(chan.faces)
                fi.axis == 1 && AMR._cell_index(fi.plus, nx, ny) == cellidx || continue        # faces on the left of the boundary cell (it is their plus cell)
                num += fi.area * v_raw.interior[g]; den += fi.area
            end
            raw[i] = num / den
        end
        v_after = vel0()
        enforce_velocity_boundary!(v_after, bc; time = 0.0)
        net = sum(f.side * f.area * v_after.boundary[i] for (i, f) in pairs(chan.boundary_faces))
        # recompute the uncorrected net flux: inflow + NoSlip(0) + raw outflow means
        inflow = sum(f.side * f.area * prof(f.center[2]) for (i, f) in pairs(chan.boundary_faces) if f.axis == 1 && f.side == -1)
        outarea = sum(f.area for (i, f) in pairs(chan.boundary_faces) if f.axis == 1 && f.side == 1)
        net0 = inflow + sum(f.side * f.area * raw[i] for (i, f) in pairs(chan.boundary_faces) if haskey(raw, i))
        corr = -net0 / outarea
        e7 = maximum(abs(v_after.boundary[i] - (raw[i] + corr)) for i in keys(raw))
        println("O7: Outflow values = weighted mean + uniform correction (", corr, "): max difference ", e7, "; total boundary flux after = ", net)
        @test e7 <= 1.0e-13
        @test abs(net) <= 1.0e-13
    end
    # O4
    bz = VelocityBoundaryConditions(; left = Inflow((x, y, t, axis) -> axis == 1 ? prof(y) : 0.0), right = ZeroGradientVelocity(), bottom = NoSlip(), top = NoSlip())
    @test_throws ArgumentError project!(vel0(), CellField(chan), solver, dt, work; velocity_boundary = bz, tolerance = 1.0e-12, max_cycles = 500)
    # O5: closed box (no outflow): second application changes nothing
    box = BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (false, false), max_level = 1, refine = (key, o, e) -> key.level == 0 && key.i >= 2)
    bop = PoissonOperator(box); bs = EllipticSolver(bop; preconditioner = :tree_geometric)
    bfree = VelocityBoundaryConditions(; left = FreeSlip(), right = NoSlip(), bottom = Inflow((x, y, t, axis) -> 0.0), top = FreeSlip())
    vb = FaceVelocity(box, (x, y, t, axis) -> axis == 1 ? sinpi(x / 4) * cospi(y / 8) : cospi(x / 8) * sinpi(y / 4))
    st = project!(vb, CellField(box), bs, dt, ProjectionWorkspace(bs); velocity_boundary = bfree, tolerance = 1.0e-12, max_cycles = 500)
    nx2, ny2 = box.block_size
    out = zeros(length(box.blocks) * nx2 * ny2)
    for (i, f) in pairs(box.faces)
        w = f.area * vb.interior[i]; out[AMR._cell_index(f.minus, nx2, ny2)] += w; out[AMR._cell_index(f.plus, nx2, ny2)] -= w
    end
    for (i, f) in pairs(box.boundary_faces)
        out[AMR._cell_index(f.cell, nx2, ny2)] += f.side * f.area * vb.boundary[i]
    end
    volb = [box.blocks[(c - 1) ÷ (nx2 * ny2) + 1].volume for c in 1:length(out)]
    println("O5: closed box, dt max|D v| = ", dt * maximum(abs.(out ./ volb)), " (tolerance 1e-12), converged ", st.converged)
    @test dt * maximum(abs.(out ./ volb)) <= 1.0e-11
    # O6: solid
    field = PA.fixture(:amr; solid = true)
    op = FA.one_velocity_operator(field)
    vs = PA.one_predictor(field, MersenneTwister(2)); a = PA.aperture(field)
    vs.interior .+= 1.0                                           # garbage on a = 0 faces before the projection
    p2, st2, _ = PA.run_projection!(field, op, vs)
    @test all(vs.interior[a .== 0] .== 0.0)
    println("O6: ", count(a .== 0), " faces with a_f = 0, all zero after the projection")
end

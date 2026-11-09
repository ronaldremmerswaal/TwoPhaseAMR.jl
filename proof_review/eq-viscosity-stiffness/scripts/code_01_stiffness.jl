# Code check 01: eq:viscosity:stiffness -- the single-phase integrated stiffness: symmetric, positive semidefinite, conservative core, kernel,
# cut-cell variant symmetric up to the volumes.  (The independent assembly S = sym(sum meas d a'a + sum cond e e') and the identity
# u'Su = sum meas d (grad u)^2 + sum cond u^2 are in eq-viscosity-gradient/scripts/code_01_gradient_stiffness.jl.)
#
# Label:  eq:viscosity:stiffness
# Source: docs/techref/chapters/09_viscosity.tex:70-92, 178-200; src/operators/staggered_diffusion.jl (_diffusion_stiffness),
#         src/embedded/operators.jl (_cut_cell_diffusion_component)
# Checks:
#   T1  S == S' exactly (the code symmetrises), lambda_min >= -1e-13 lambda_max, for no-slip / free-slip walls, periodic, composite, uniform;
#   T2  kernel dimension: periodic meshes 1 per component (constants); walled NoSlip/FreeSlip 0; x-periodic channel with FreeSlip: 1 (u_1 = const),
#       NoSlip: 0; 1' S = 0 (columns sum to 0) on periodic meshes;
#   T3  S = sum of PSD terms: removing the wall terms leaves S_core PSD with the constant in its kernel (walled mesh);
#   T4  cut cells (uniform mesh, disc): S symmetric PSD, V * L = -S is symmetric ("symmetric up to the volumes"), covered rows -1;
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-viscosity-stiffness/scripts/code_01_stiffness.jl
using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ViscosityAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "viscosity_audit.jl"))
using .ViscosityAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR

composite(periodic) = BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic, max_level = 1, refine = (key, o, e) -> key.level == 0 && key.i >= 2)
uniform(periodic) = BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic)

@testset "T1/T2: symmetry, PSD, kernel" begin
    cases = (
        ("walled composite NoSlip", composite((false, false)), NoSlip(), 0),
        ("walled composite FreeSlip", composite((false, false)), FreeSlip(), 0),
        ("periodic composite", composite((true, true)), NoSlip(), 1),
        ("periodic uniform", uniform((true, true)), NoSlip(), 1),
        ("x-periodic composite NoSlip", composite((true, false)), NoSlip(), 0),
        ("x-periodic composite FreeSlip", composite((true, false)), FreeSlip(), :component),
    )
    for (name, mesh, boundary, kernel) in cases
        op = StaggeredDiffusionOperator(mesh; viscosity = 1.0, boundary)
        dims = Int[]
        for (component, axis) in ((op.x, 1), (op.y, 2))
            S = Matrix(component.stiffness)
            ev = eigvals(Symmetric(S))
            @test S == S'
            @test ev[1] >= -1.0e-13 * ev[end]
            push!(dims, count(<(1.0e-10 * ev[end]), ev))
            all(mesh.periodic) && @test maximum(abs, vec(sum(S, dims = 1))) <= 1.0e-12 * maximum(abs, S)
        end
        println(rpad(name, 32), "kernel dimension per component: ", dims)
        if kernel isa Int
            @test all(==(kernel), dims)
        else
            @test dims == [1, 0]               # u_1 = const (tangential, free slip, x-periodic) survives; u_2 does not
        end
    end
end

@testset "T3: core without wall terms" begin
    mesh = composite((false, false))
    op = StaggeredDiffusionOperator(mesh; viscosity = 1.0)
    for (component, axis) in ((op.x, 1), (op.y, 2))
        S = copy(component.stiffness)
        for c in component.wall_connections
            S[c.degree_of_freedom, c.degree_of_freedom] -= c.conductance
        end
        S = Matrix(S)
        ev = eigvals(Symmetric(S))
        @test ev[1] >= -1.0e-13 * ev[end]
        @test maximum(abs, S * ones(size(S, 1))) <= 1.0e-12 * maximum(abs, S)       # constants are in the kernel of the wall-free core
    end
end

@testset "T4: cut-cell stiffness" begin
    mesh = uniform((false, false))
    geometry = GeometryField(mesh, (x, y) -> hypot(x - 4.0, y - 3.2) - 1.6; lipschitz = 1)
    op = StaggeredDiffusionOperator(geometry; viscosity = 1.0)
    for (component, axis) in ((op.x, 1), (op.y, 2))
        S = Matrix(component.stiffness)
        VL = Diagonal(component.volumes) * Matrix(component.laplacian)
        ev = eigvals(Symmetric(S))
        @test S == S'
        @test ev[1] >= -1.0e-13 * ev[end]
        @test maximum(abs, VL + S) <= 1.0e-13 * maximum(abs, S)
        @test maximum(abs, VL - VL') <= 1.0e-13 * maximum(abs, S)
        println("cut component ", axis, ": lambda_min/lambda_max = ", ev[1] / ev[end], ", |V L + S| = ", maximum(abs, VL + S))
    end
end

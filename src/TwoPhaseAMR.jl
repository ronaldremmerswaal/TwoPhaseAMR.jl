module TwoPhaseAMR

import RegionTrees
using LinearAlgebra: I, Symmetric, diag, dot, isposdef, ldiv!, lu, mul!, tr
using SparseArrays: SparseMatrixCSC, dropzeros!, nonzeros, nzrange, rowvals,
                    sparse, spdiagm, spzeros
using StaticArrays: SVector

export AbstractPlacement, Cell, Face, placement
export BlockKey, LeafBlock, BlockTreeMesh, nblocks, ncells, isbalanced
export CellField, FaceField, initialize!, interior, blockdata, cellcenter, integrate
export AbstractBoundaryCondition, ZeroGradient, Dirichlet
export fill_halo!, restrict!, prolong!
export FaceVelocity, set_velocities!
export AdvectionWorkspace, advect!, stable_timestep
export AbstractStaggeredAdvectionScheme, Upwind, Centred, Centered, LaxWendroff, LW
export StaggeredAdvectionWorkspace, StaggeredVelocityAdvectionWorkspace
export convection!, staggered_inner_product
export divergence!
export GradientOperator, gradient!
export EllipticOperator, PoissonOperator, EllipticWorkspace, apply!
export harmonic_face_coefficients, weighted_gradient!
export GeometricMultigridSolver, MultigridStats, solve!, residual_norms
export ProjectionWorkspace, project!

include("placements.jl")
include("mesh.jl")
include("fields.jl")
include("transfer.jl")
include("advection.jl")
include("gradient.jl")
include("staggered_advection.jl")
include("elliptic.jl")
include("multigrid.jl")

end

module TwoPhaseAMR

import RegionTrees
using LinearAlgebra: Diagonal, I, Symmetric, diag, dot, isposdef, ldiv!, lu, mul!, tr
using SparseArrays: SparseMatrixCSC, dropzeros!, nonzeros, nzrange, rowvals,
                    sparse, spdiagm, spzeros
using StaticArrays: SVector

abstract type AbstractEllipticOperator{T} end

export AbstractPlacement, Cell, Face, placement
export BlockKey, LeafBlock, CellRef, BlockTreeMesh, nblocks, ncells, isbalanced
export CellField, FaceField, initialize!, interior, blockdata, cellcenter, integrate
export AbstractBoundaryCondition, ZeroGradient, Dirichlet
export fill_halo!, restrict!, prolong!
export FaceVelocity, set_velocities!, enforce_no_permeation!
export AdvectionWorkspace, advect!, stable_timestep
export AbstractStaggeredAdvectionScheme, Upwind, Centred, Centered, LaxWendroff, LW
export StaggeredAdvectionWorkspace, StaggeredVelocityAdvectionWorkspace
export convection!, staggered_inner_product
export AbstractWallBoundaryCondition, NoSlip, FreeSlip, Slip
export StaggeredDiffusionOperator, diffusion!
export IncompressibleNavierStokesSolver, NavierStokesSolver
export NavierStokesWorkspace, step!, simulate!
export divergence!
export GradientOperator, gradient!
export AbstractEllipticOperator, EllipticOperator, PoissonOperator, EllipticWorkspace, apply!
export StaggeredEllipticOperator, StaggeredEllipticSolver
export StaggeredGeometricMultigridSolver
export implicit_midpoint_diffusion!
export harmonic_face_coefficients, weighted_gradient!
export GeometricMultigridSolver, MultigridStats, solve!, residual_norms
export ProjectionWorkspace, project!
export AbstractVisualization, NoVisualization, AMRCellGeometry
export visualization_geometry, visualization_values, cell_for_vertex
export visualize, update_visualization!, selected_cell, visualization_figure

include("placements.jl")
include("mesh.jl")
include("fields.jl")
include("transfer.jl")
include("advection.jl")
include("gradient.jl")
include("staggered_advection.jl")
include("staggered_diffusion.jl")
include("elliptic.jl")
include("multigrid.jl")
include("implicit_diffusion.jl")
include("navier_stokes.jl")
include("visualization.jl")

end

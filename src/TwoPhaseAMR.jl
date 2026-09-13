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
export FaceRef, AMRFaceGeometry, CentredMagnitude, CenteredMagnitude, Vorticity
export VisualizationFieldRegistry, register_field!, register_default_fields!, field_names
export register_diagnostic!, diagnostic_names
export visualization_geometry, visualization_values, cell_for_vertex, face_for_vertex
export visualize, initialize_visualization!, update_visualization!
export select_visualization_field!
export selected_cell, selected_face, visualization_figure
export VisualizationFrame, VisualizationCheckpoint, VisualizationDiagnosticSample
export VisualizationSession
export visualization_checkpoints, latest_frame, select_checkpoint!, live_mode!
export visualization_diagnostics
export wait_visualization!, dropped_visualization_frames
export simulation_state, request_pause!, resume_simulation!, wait_until_paused

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
include("visualization.jl")
include("visualization_session.jl")
include("navier_stokes.jl")

end

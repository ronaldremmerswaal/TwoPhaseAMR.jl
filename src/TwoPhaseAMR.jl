"""
Tree-native block-adaptive finite-volume operators for single-phase and
incompressible one-velocity two-phase Navier--Stokes flow.
"""
module TwoPhaseAMR

import RegionTrees
import Krylov
import Polyester
import LinearAlgebra: issymmetric, ldiv!, mul!, transpose!
import SparseArrays: nnz
using LinearAlgebra: Diagonal, I, Symmetric, diag, dot, isposdef,
    lu, tr, svdvals
using SparseArrays: SparseMatrixCSC, dropzeros!, nonzeros, nzrange, rowvals,
    sparse, spdiagm, spzeros
using StaticArrays: SVector

"""Interface for assembled scalar or staggered elliptic operators."""
abstract type AbstractEllipticOperator{T} end

export AbstractPlacement, Cell, Face, placement, AbstractMeshField
export BlockKey, LeafBlock, CellRef, BlockTreeMesh, nblocks, ncells, isbalanced
export MeshAdaptationOptions
export AbstractRefinementIndicator, InterpolationErrorIndicator
export AMRWorkspace, AMRStats, refinement_labels!, register_adaptive_field!
export adapt!
export CellField, FaceField, initialize!, interior, blockdata, cellcenter, integrate
export AbstractBoundaryCondition, ZeroGradient, Dirichlet
export fill_halo!, restrict!, prolong!
export FaceVelocity, TwoPhaseVelocity, set_velocities!, enforce_no_permeation!
export AdvectionWorkspace, advect!, stable_timestep
export InterfaceField, reconstruct!, WYAdvectionWorkspace, PhaseScalar
export GeometryField, refresh_geometry!, enforce_geometry_no_slip!, plic_segments
export GeometryRefinementIndicator
export PLICSolidGeometry, plic_solid_geometry
export PhaseFaceScalar, MomentumAdvectionWorkspace, InterfaceRefinementIndicator
export CurvatureRefinementIndicator, VorticityRefinementIndicator
export CSF, GFM, surface_tension!, gravity!, mixture_density
export VariableStaggeredDiffusionOperator, dynamic_viscosity
export AbstractStaggeredAdvectionScheme, Upwind, Centred, Centered, LaxWendroff, LW
export Fromm, Minmod, MonotonizedCentral, MC, VanLeer, Superbee
export default_amr_interface_scheme
export StaggeredAdvectionWorkspace, StaggeredVelocityAdvectionWorkspace
export convection!, staggered_inner_product
export AbstractWallBoundaryCondition, NoSlip, FreeSlip, Slip
export VelocityBoundaryConditions, Inflow, Outflow, ZeroGradientVelocity
export enforce_velocity_boundary!
export StaggeredDiffusionOperator, diffusion!
export IncompressibleNavierStokesSolver, NavierStokesSolver
export NavierStokesWorkspace, step!, simulate!
export FlowSimulation, NavierStokesSimulation
export divergence!
export GradientOperator, gradient!
export AbstractEllipticOperator, EllipticOperator, PoissonOperator, EllipticWorkspace, apply!
export StaggeredEllipticOperator, StaggeredEllipticSolver
export TreeStaggeredGeometricMultigridPreconditioner
export StaggeredGeometricMultigridPreconditioner
export StaggeredGeometricMultigridSolver
export implicit_midpoint_diffusion!
export harmonic_face_coefficients, weighted_gradient!
export GeometricMultigridPreconditioner, GeometricMultigridSolver
export TreeGeometricMultigridPreconditioner
export EllipticSolver
export LinearSolverStats, MultigridStats, solve!, residual_norms
export algebraic_multigrid_preconditioner, incomplete_lu_preconditioner
export incomplete_cholesky_preconditioner
export ProjectionWorkspace, project!
export AbstractVisualization, NoVisualization, AMRCellGeometry
export FaceRef, AMRFaceGeometry, CentredMagnitude, CenteredMagnitude, Vorticity
export VisualizationFieldRegistry, register_field!, register_default_fields!, field_names
export register_interface!, register_geometry!
export register_diagnostic!, diagnostic_names
export visualization_geometry, visualization_values, cell_for_vertex, face_for_vertex
export visualize, initialize_visualization!, update_visualization!
export select_visualization_field!
export selected_cell, selected_face, visualization_figure
export VisualizationTopology, VisualizationFrame, VisualizationCheckpoint
export VisualizationDiagnosticSample
export VisualizationSession
export visualization_checkpoints, latest_frame, select_checkpoint!, live_mode!
export visualization_diagnostics
export wait_visualization!, dropped_visualization_frames
export simulation_state, request_pause!, resume_simulation!, wait_until_paused

include("placements.jl")
include("threading.jl")
include("mesh.jl")
include("fields.jl")
include("transfer.jl")
include("advection.jl")
include("adaptivity.jl")
include("gradient.jl")
include("staggered_advection.jl")
include("staggered_diffusion.jl")
include("elliptic.jl")
include("multigrid.jl")
include("implicit_diffusion.jl")
include("height_functions.jl")
include("interface.jl")
include("levelset_initialization.jl")
include("geometry.jl")
include("vof_advection.jl")
include("phase_staggered_advection.jl")
include("two_velocity.jl")
include("two_phase_forces.jl")
include("variable_staggered_diffusion.jl")
include("two_phase_model.jl")
include("visualization.jl")
include("visualization_session.jl")
include("navier_stokes.jl")
include("two_phase_step.jl")
include("flow_simulation.jl")

end

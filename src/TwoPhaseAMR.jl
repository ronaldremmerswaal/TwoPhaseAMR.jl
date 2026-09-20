"""
Tree-native block-adaptive finite-volume operators for single-phase and
incompressible one- and two-velocity two-phase Navier--Stokes flow.
"""
module TwoPhaseAMR

import RegionTrees
import Krylov
import Polyester
import IncompleteLU
import LinearAlgebra: issymmetric, ldiv!, mul!, transpose!
import SparseArrays: nnz
using LinearAlgebra: Diagonal, I, Symmetric, diag, dot, isposdef,
    cholesky, eigen, lu, norm, nullspace, qr, rank, tr, svdvals
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
export InterfaceField, PolygonIndicator, reconstruct!, interface_height
export WYAdvectionWorkspace, PhaseScalar
export AbstractInterfaceTransportMethod
export ProjectedLiquidVelocity, FrozenInterfaceFluxes
export phase_flux_diagnostics
export GeometryField, refresh_geometry!, enforce_geometry_no_slip!, plic_segments
export GeometryRefinementIndicator
export PLICSolidGeometry, plic_solid_geometry
export PhaseFaceScalar, MomentumAdvectionWorkspace, InterfaceRefinementIndicator
export CurvatureRefinementIndicator, VorticityRefinementIndicator
export AbstractSurfaceTensionMethod, CSF, GFM, CGFM
export surface_tension!, gravity!, mixture_density
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
export LinearSolverOptions, MultigridOptions
export AdvectionOptions, NavierStokesOptions, Adaptivity
export SinglePhase, TwoPhase, SurfaceTension
export AbstractVelocityModel, OneVelocityModel, TwoVelocityModel
export IncompressibleNavierStokesSolver, NavierStokesSolver
export NavierStokesSystem
export NavierStokesWorkspace, step!, simulate!
export FlowSimulation, NavierStokesSimulation
export SimulationState, simulation_state
export AbstractDiagnostic, Diagnostic, diagnostic_name, diagnostic_value
export KineticEnergy, PotentialEnergy, SurfaceEnergy, TotalEnergy
export LinearMomentum, PhaseVolume, PhaseMass, InterfaceMeasure
export MaximumVelocity, CFLTimeStep, DivergenceNorm
export LeafBlockCount, CellCount
export AbstractDiagnosticSchedule, EveryStep, EveryNSteps, EveryTimeInterval
export DiagnosticCollection, DiagnosticHistory
export add_diagnostic!, record_diagnostics!, diagnostic_history
export print_simulation_summary
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
export MimeticProjectionSolver, MimeticProjectionWorkspace
export MimeticProjectionStats, mimetic_gradient!
export AbstractTwoVelocityProjection, AbstractTwoVelocityPressureProjection
export MimeticProjection
export CMDGFMProjection, CMDGFMProjectionSolver, CMDGFMProjectionWorkspace
export CMDGFMProjectionStats, cmdgfm_gradient!, cmdgfm_jump_interpolant
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

# Core and mesh topology
include("mesh/topology/placements.jl")
include("threading.jl")
include("mesh/topology/mesh.jl")
include("solvers/linear_options.jl")
include("solvers/multigrid/options.jl")

# Variables, transfer, and mesh adaptation
include("mesh/variables.jl")
include("mesh/transfer.jl")
include("operators/advection/cell_advection.jl")
include("mesh/adaptivity/indicators.jl")
include("mesh/adaptivity/workspace.jl")
include("mesh/adaptivity/field_transfer.jl")
include("mesh/adaptivity/adapt.jl")
include("mesh/adaptivity/configuration.jl")

# Generic discrete operators and linear solvers
include("operators/gradient.jl")
include("operators/advection/staggered_schemes.jl")
include("operators/advection/staggered_schedules.jl")
include("operators/advection/staggered_fluxes.jl")
include("operators/advection/staggered_update.jl")
include("navier_stokes/options.jl")
include("operators/staggered_diffusion.jl")
include("operators/elliptic.jl")
include("solvers/multigrid/hierarchy.jl")
include("solvers/multigrid/geometric_preconditioner.jl")
include("solvers/multigrid/cycles.jl")
include("solvers/multigrid/elliptic_solver.jl")
include("navier_stokes/projection.jl")
include("navier_stokes/implicit_diffusion.jl")

# Physical indicator geometry and transport
include("indicator/phase/height_functions.jl")
include("indicator/phase/interface_field.jl")
include("indicator/phase/levelset_initialization.jl")
include("indicator/embedded_geometry.jl")
include("indicator/phase/polygon_initialization.jl")
include("indicator/embedded_operators.jl")
include("indicator/plic_geometry.jl")
# These mesh policies extend the physical fields defined immediately above.
include("mesh/adaptivity/phase_criteria.jl")
include("mesh/adaptivity/geometry_criteria.jl")
include("indicator/phase/scalars.jl")
include("two_phase/two_velocity/transport_methods.jl")
include("indicator/phase/vof_workspace.jl")
include("indicator/phase/vof_fluxes.jl")
include("indicator/phase/scalar_transfer.jl")

# One- and two-velocity two-phase models
include("two_phase/phase_face_fields.jl")
include("two_phase/momentum_workspace.jl")
include("two_phase/phase_face_transport.jl")
include("two_phase/phase_face_transfer.jl")
include("two_phase/two_velocity/momentum_workspace.jl")
include("two_phase/two_velocity/transport_workspace.jl")
include("two_phase/two_velocity/velocity_fields.jl")
include("two_phase/two_velocity/interface_transport.jl")
include("two_phase/two_velocity/transport_geometry.jl")
include("two_phase/two_velocity/projection/methods.jl")
include("two_phase/two_velocity/projection/mimetic_foundation.jl")
include("two_phase/two_velocity/projection/cmdgfm_projection.jl")
include("two_phase/two_velocity/projection/mimetic_projection.jl")
include("two_phase/forces.jl")
include("single_phase/physics.jl")
include("two_phase/physics.jl")
include("two_phase/one_velocity.jl")
include("two_phase/two_velocity/model.jl")
include("two_phase/variable_diffusion_geometry.jl")
include("two_phase/variable_diffusion_operator.jl")
include("two_phase/variable_diffusion_preconditioner.jl")
include("two_phase/variable_diffusion_solver.jl")
include("two_phase/model.jl")
include("two_phase/two_velocity/flow_workspace.jl")

# Simulation state and diagnostics (independent of visualization)
include("simulation/state.jl")
include("simulation/diagnostics/schedules.jl")
include("simulation/diagnostics/interface.jl")
include("simulation/diagnostics/history.jl")

# Visualization and equation-system orchestration
include("visualization/types.jl")
include("visualization/registry.jl")
include("visualization/phase_geometry.jl")
include("visualization/sampling.jl")
include("visualization/interface.jl")
include("visualization/checkpoints.jl")
include("visualization/session_types.jl")
include("visualization/frame_capture.jl")
include("visualization/updates.jl")
include("visualization/controls.jl")
include("navier_stokes/system.jl")
include("navier_stokes/workspace.jl")
include("navier_stokes/stages.jl")
include("single_phase/stepping.jl")
include("two_phase/stepping.jl")
include("two_phase/two_velocity/stepping.jl")
include("navier_stokes/independent_step.jl")
include("navier_stokes/simulate.jl")
include("navier_stokes/flow_simulation.jl")
include("navier_stokes/flow_diagnostics.jl")

end

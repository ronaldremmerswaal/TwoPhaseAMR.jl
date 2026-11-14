"""
Tree-native block-adaptive finite-volume operators for single-phase and
incompressible one- and two-velocity two-phase Navier--Stokes flow.
"""
module TwoPhaseAMR

import RegionTrees
import Krylov
import Polyester
import IncompleteLU
import LinearAlgebra: issymmetric, ldiv!, mul!
import SparseArrays: nnz
using LinearAlgebra: ColumnNorm, Diagonal, I, Symmetric, diag, dot, isposdef,
    cholesky, eigen, lu, norm, nullspace, qr, qr!, rank, tr, svdvals
using SparseArrays: SparseMatrixCSC, dropzeros!, nonzeros, nzrange, rowvals,
    sparse, spdiagm, spzeros
using StaticArrays: SVector

const _SupportedReal = Union{Float32, Float64}

@inline function _check_real_type(T, message = "element type must be Float32 or Float64"; exact = false)
    (exact ? T in (Float32, Float64) : T <: _SupportedReal) ||
        throw(ArgumentError(message))
    return nothing
end

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
export phase_flux_diagnostics
export GeometryField, refresh_geometry!, enforce_geometry_no_slip!, plic_segments
export GeometryRefinementIndicator
export PLICSolidGeometry, plic_solid_geometry
export PhaseFaceScalar, MomentumAdvectionWorkspace, InterfaceRefinementIndicator
export CurvatureRefinementIndicator, VorticityRefinementIndicator
export AbstractSurfaceTensionMethod, CSF, CGFM
export surface_tension!, gravity!, mixture_density
export VariableStaggeredDiffusionOperator, dynamic_viscosity
export AbstractStaggeredAdvectionScheme, Upwind, Centred, LaxWendroff
export Fromm, Minmod, MonotonizedCentral, VanLeer, Superbee
export default_amr_interface_scheme
export StaggeredAdvectionWorkspace, StaggeredVelocityAdvectionWorkspace
export convection!, staggered_inner_product
export AbstractWallBoundaryCondition, NoSlip, FreeSlip
export VelocityBoundaryConditions, Inflow, Outflow, ZeroGradientVelocity
export enforce_velocity_boundary!
export StaggeredDiffusionOperator, diffusion!
export AbstractViscosityAveraging, HarmonicViscosityAveraging, GeometricViscosityAveraging
export LinearSolverOptions, MultigridOptions
export AdvectionOptions, NavierStokesOptions, Adaptivity
export SinglePhase, TwoPhase, SurfaceTension
export AbstractVelocityModel, OneVelocityModel, TwoVelocityModel
export NavierStokesSystem
export NavierStokesWorkspace, step!, simulate!
export FlowSimulation
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
export EllipticOperator, PoissonOperator, EllipticWorkspace, apply!
export StaggeredEllipticOperator, StaggeredEllipticSolver
export TreeStaggeredGeometricMultigridPreconditioner
export implicit_midpoint_diffusion!
export harmonic_face_coefficients, weighted_gradient!
export TreeGeometricMultigridPreconditioner
export EllipticSolver
export LinearSolverStats, solve!, residual_norms
export ProjectionRunStats
export algebraic_multigrid_preconditioner, incomplete_lu_preconditioner
export incomplete_cholesky_preconditioner
export ProjectionWorkspace, project!
export AbstractTwoVelocityProjection
export CMDGFMProjection, CMDGFMProjectionSolver, CMDGFMProjectionWorkspace
export CMDGFMProjectionStats, cmdgfm_gradient!, cmdgfm_jump_interpolant
export MimeticProjection
export MimeticProjectionSolver, MimeticProjectionWorkspace, MimeticProjectionStats
export PhysicalDimension, DiagnosticInfo, diagnostic_info
export DiagnosticGroup, diagnostic_groups, diagnostic_names
export AbstractObserver, NoObserver, StepInfo, observe!, stop_requested
export Recorder, AtTimes, EquispacedTimes, EveryWallInterval, Retention
export run_state, pause!, wait_paused, resume!, request_stop!
export request_keyframe!, pause_when!
export SnapshotStore, snapshot_store, frames, keyframes, frame_at, newest_frame
export frame_stats, store_usage, step_log, run_events, subscribe
export VideoSchedule, video_frame_count, video_frame_schedule, MAX_VIDEO_FRAMES
export LiveVideoSampler, push_live_frame!, flush_live_frames!
export Frame, FrameInfo, FieldStats, TopologySnapshot, FrameSubscription
export StepLog, RunEvent, DiagnosticColumns
export ViewCamera, fit_camera, zoom, zoom_to_box, visible_rect, rasterize, locate, locate_face, FaceRef
export CentredMagnitude, Vorticity
export VisualizationFieldRegistry, register_field!, register_default_fields!, field_names
export register_interface!, register_geometry!, visualization_values
export fieldaxis, amrfield, amrfield!, diagnosticsgrid!, export_animation, export_figure

# mesh/topology
include("mesh/topology/placements.jl")
# src
include("threading.jl")
# mesh/topology
include("mesh/topology/mesh.jl")
# solvers
include("solvers/linear_options.jl")
# solvers/multigrid
include("solvers/multigrid/options.jl")
# mesh
include("mesh/variables.jl")
include("mesh/transfer.jl")
# operators/advection
include("operators/advection/cell_advection.jl")
# mesh/adaptivity
include("mesh/adaptivity/indicators.jl")
include("mesh/adaptivity/workspace.jl")
include("mesh/adaptivity/field_transfer.jl")
include("mesh/adaptivity/adapt.jl")
include("mesh/adaptivity/configuration.jl")
# operators
include("operators/gradient.jl")
# operators/advection
include("operators/advection/staggered_schemes.jl")
include("operators/advection/staggered_schedules.jl")
include("operators/advection/staggered_fluxes.jl")
include("operators/advection/staggered_update.jl")
# operators
include("operators/viscosity_averaging.jl")
# navier_stokes
include("navier_stokes/options.jl")
# operators
include("operators/staggered_diffusion.jl")
include("operators/elliptic.jl")
# solvers/multigrid
include("solvers/multigrid/hierarchy.jl")
include("solvers/multigrid/stats.jl")
include("solvers/multigrid/tree_coarsening.jl")
include("solvers/multigrid/geometric_preconditioner.jl")
include("solvers/multigrid/cycles.jl")
include("solvers/multigrid/elliptic_solver.jl")
# navier_stokes
include("navier_stokes/projection.jl")
include("navier_stokes/projection_stats.jl")
include("navier_stokes/implicit_diffusion.jl")
# interface
include("interface/height_functions.jl")
include("interface/interface_field.jl")
# embedded
include("embedded/geometry_utils.jl")
# interface
include("interface/levelset_initialization.jl")
# embedded
include("embedded/geometry.jl")
# interface
include("interface/polygon_initialization.jl")
# embedded
include("embedded/operators.jl")
# visualization
include("visualization/plic_geometry.jl")
# mesh/adaptivity
include("mesh/adaptivity/phase_criteria.jl")
include("mesh/adaptivity/geometry_criteria.jl")
# vof
include("vof/scalars.jl")
include("vof/transport_methods.jl")
include("vof/workspace.jl")
include("vof/fluxes.jl")
include("vof/scalar_transfer.jl")
# two_phase
include("two_phase/phase_face_fields.jl")
include("two_phase/momentum_workspace.jl")
include("two_phase/phase_face_transport.jl")
include("two_phase/phase_face_transfer.jl")
# two_phase/two_velocity
include("two_phase/two_velocity/momentum_workspace.jl")
include("two_phase/two_velocity/velocity_fields.jl")
include("two_phase/two_velocity/interface_transport.jl")
include("two_phase/two_velocity/transport_geometry.jl")
# two_phase/projection
include("two_phase/projection/methods.jl")
include("two_phase/projection/common.jl")
include("two_phase/projection/cmdgfm_cache.jl")
include("two_phase/projection/cmdgfm_projection.jl")
include("two_phase/projection/mimetic_metric.jl")
include("two_phase/projection/mimetic_projection.jl")
# two_phase
include("two_phase/forces.jl")
# single_phase
include("single_phase/physics.jl")
# two_phase
include("two_phase/physics.jl")
include("two_phase/one_velocity.jl")
# two_phase/two_velocity
include("two_phase/two_velocity/model.jl")
# two_phase/diffusion
include("two_phase/diffusion/variable_diffusion_geometry.jl")
include("two_phase/diffusion/variable_diffusion_operator.jl")
include("two_phase/diffusion/variable_diffusion_preconditioner.jl")
include("two_phase/diffusion/variable_diffusion_solver.jl")
# two_phase
include("two_phase/model.jl")
# two_phase/two_velocity
include("two_phase/two_velocity/flow_workspace.jl")
# simulation
include("simulation/state.jl")
# simulation/diagnostics
include("simulation/diagnostics/schedules.jl")
# visualization
include("visualization/diagnostic_metadata.jl")
# simulation/diagnostics
include("simulation/diagnostics/interface.jl")
include("simulation/diagnostics/history.jl")
# visualization
include("visualization/types.jl")
include("visualization/registry.jl")
include("visualization/sampling.jl")
include("visualization/interface.jl")
include("visualization/diagnostic_groups.jl")
# observation
include("observation/observer.jl")
include("observation/schedules.jl")
include("observation/topology.jl")
include("observation/frames.jl")
include("observation/video.jl")
include("observation/capture.jl")
include("observation/store.jl")
include("observation/recorder.jl")
# visualization
include("visualization/camera.jl")
include("visualization/rasterize.jl")
# navier_stokes
include("navier_stokes/system.jl")
include("navier_stokes/workspace.jl")
include("navier_stokes/stages.jl")
# single_phase
include("single_phase/stepping.jl")
# two_phase
include("two_phase/stepping.jl")
# two_phase/two_velocity
include("two_phase/two_velocity/stepping.jl")
# navier_stokes
include("navier_stokes/independent_step.jl")
include("navier_stokes/flow_simulation.jl")
include("navier_stokes/simulate.jl")
include("navier_stokes/flow_diagnostics.jl")
# src
include("precompile.jl")

end

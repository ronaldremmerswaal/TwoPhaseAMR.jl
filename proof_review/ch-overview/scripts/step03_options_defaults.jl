# Check 03: the options table tab:overview:options -- defaults (first value), the admitted values, and the solver choices of the Krylov row;
# the step limits and defaults quoted in the implementation boxes of sec:overview:models / step / run / options.
#
# Label:  tab:overview:options
# Source: docs/techref/chapters/01_overview.tex:585-690 (table), 160-166, 262-280 (boxes); src/two_phase/two_velocity/model.jl,
#         src/two_phase/projection/methods.jl, src/two_phase/physics.jl, src/navier_stokes/options.jl, src/interface/interface_field.jl,
#         src/solvers/linear_options.jl, src/navier_stokes/flow_simulation.jl
# Method: construct the objects of the INSTRUMENTED package (identical code) with and without keywords and compare with the table.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/ch-overview/scripts/step03_options_defaults.jl

include(joinpath(@__DIR__, "_fixtures.jl"))

throws(f, T = ArgumentError) = try
    f(); false
catch e
    e isa T
end

println("== velocity formulation, carrier, projection")
flow = drop_flow()
check("FlowSimulation: velocity_model defaults to OneVelocityModel (shared FaceVelocity)", flow.velocity isa A.FaceVelocity)
tv = A.TwoVelocityModel()
check("TwoVelocityModel(): FrozenInterfaceFluxes carrier, CMDGFMProjection", tv.interface_transport isa A.FrozenInterfaceFluxes && tv.pressure_projection isa A.CMDGFMProjection)
check("TwoVelocityModel accepts exactly the two carriers and the two projections", A.TwoVelocityModel(interface_transport = A.ProjectedLiquidVelocity()).interface_transport isa A.ProjectedLiquidVelocity && A.TwoVelocityModel(pressure_projection = A.MimeticProjection()).pressure_projection isa A.MimeticProjection && throws(() -> A.TwoVelocityModel(interface_transport = 1)) && throws(() -> A.TwoVelocityModel(pressure_projection = :cmdgfm)))
check("a shared velocity initializer with a two-velocity model is copied; a pair with one velocity is rejected (never inferred)", drop_flow(model = tv, velocity = (0.0, 0.0)).velocity isa A.TwoPhaseVelocity && throws(() -> drop_flow(velocity = ((x, y, t, a) -> 0.0, (x, y, t, a) -> 0.0))))

println("\n== CMDGFM preconditioner, mimetic variant")
check("CMDGFMProjection default :block; :ilu and nothing accepted; others rejected", A.CMDGFMProjection().preconditioner === :block && A.CMDGFMProjection(preconditioner = :ilu).preconditioner === :ilu && A.CMDGFMProjection(preconditioner = nothing).preconditioner === nothing && throws(() -> A.CMDGFMProjection(preconditioner = :amg)))
check("MimeticProjection default variant :joint_smoothed; :joint_face accepted; others rejected", A._mimetic_variant_symbol(A.MimeticProjection().variant) === :joint_smoothed && A._mimetic_variant_symbol(A.MimeticProjection(variant = :joint_face).variant) === :joint_face && throws(() -> A.MimeticProjection(variant = :joint)))

println("\n== surface tension")
check("SurfaceTension default method CGFM; CSF accepted; TwoPhase(surface_tension = number) is CGFM", A.SurfaceTension(0.1).method isa A.CGFM && A.SurfaceTension(0.1; method = A.CSF()).method isa A.CSF && drop_flow().system.two_phase.formulation isa A.CGFM && drop_flow(sigma = 0.0).system.two_phase.formulation isa A.CGFM)
check("SinglePhase has no gravity parameter; TwoPhase defaults gravity (0,0), no scalars", !(:gravity in fieldnames(A.SinglePhase)) && A.TwoPhase(A.InterfaceField(unit_mesh(), drop_levelset; lipschitz = 1); density = (1.0, 2.0), dynamic_viscosity = (0.0, 0.0)).gravity == (0.0, 0.0))

println("\n== momentum face states")
schemes = [A.LaxWendroff(), A.Upwind(), A.Centred(), A.Fromm(), A.Minmod(), A.MonotonizedCentral(), A.VanLeer(), A.Superbee()]
check("AdvectionOptions default scheme LaxWendroff", A.AdvectionOptions().scheme isa A.LaxWendroff)
expected = [A.MonotonizedCentral, A.Upwind, A.MonotonizedCentral, A.Fromm, A.Minmod, A.MonotonizedCentral, A.VanLeer, A.Superbee]
got = [typeof(A.AdvectionOptions(scheme = s).amr_interface_scheme) for s in schemes]
check("states at level transitions: MonotonizedCentral for LaxWendroff and Centred, otherwise the scheme itself (all 8 schemes)", got == expected, string(got))
check("an upwind transition scheme with a non-upwind scheme is rejected", throws(() -> A.AdvectionOptions(scheme = A.LaxWendroff(), amr_interface_scheme = A.Upwind())))

println("\n== reconstruction")
mesh = unit_mesh()
check("InterfaceField reconstruction_method default :lhf; :lhf_ghf and :wy accepted; others rejected", A.InterfaceField(mesh, drop_levelset; lipschitz = 1).reconstruction_method === :lhf && all(m -> A.InterfaceField(mesh, drop_levelset; lipschitz = 1, reconstruction_method = m).reconstruction_method === m, (:lhf_ghf, :wy)) && throws(() -> A.InterfaceField(mesh, drop_levelset; lipschitz = 1, reconstruction_method = :ghf)))

println("\n== Krylov method and preconditioner")
check("LinearSolverOptions default: krylov = nothing, preconditioner = :auto", A.LinearSolverOptions().krylov === nothing && A.LinearSolverOptions().preconditioner === :auto)
sys = single_flow().system
check("one-phase pressure: CG with the tree geometric multigrid (:auto)", sys.pressure_solver.krylov isa Main.Krylov.CgWorkspace && sys.pressure_solver.stats.preconditioner in (:tree_geometric_multigrid, :tree_geometric) || (sys.pressure_solver.krylov isa Main.Krylov.CgWorkspace && sys.pressure_solver.preconditioner isa A.TreeGeometricMultigridPreconditioner))
f1 = single_flow()
A.simulate!(f1, 1.0e-3; dt = 1.0e-3)
check("one-phase viscous step on a uniform mesh: CG (symmetric)", f1.workspace.elliptic.krylov_factory === Main.Krylov.CgWorkspace, string(f1.workspace.elliptic.krylov_factory))
amr_mesh = A.BlockTreeMesh(
    base_mesh = (16, 16), block_size = (4, 4), max_level = 1, periodic = (false, false),
    refine = (key, origin, extent) -> key.level == 0 && key.i >= 2,
)
f2 = single_flow(mesh = amr_mesh)
A.simulate!(f2, 1.0e-3; dt = 1.0e-3)
check("one-phase viscous step on a composite mesh: BiCGSTAB (nonsymmetric rows at level transitions, issue #80)", f2.workspace.elliptic.krylov_factory === Main.Krylov.BicgstabWorkspace, string(f2.workspace.elliptic.krylov_factory))
f3 = drop_flow()
check("two-phase viscous step: CG with tree geometric multigrid", f3.workspace.two_phase.diffusion_solver.krylov isa Main.Krylov.CgWorkspace)
A.simulate!(f3, 1.0e-3; dt = 1.0e-3)
f4 = drop_flow(model = tv)
A.simulate!(f4, 1.0e-3; dt = 1.0e-3)
st = A._two_velocity_flow_workspace!(f4.workspace.two_phase, f4.system.two_phase)
check("CMDGFM system: BiCGSTAB (not CG) with its own block preconditioner", st.projection_workspace.krylov isa Main.Krylov.BicgstabWorkspace, string(typeof(st.projection_workspace.krylov)))
f5 = drop_flow(model = A.TwoVelocityModel(pressure_projection = A.MimeticProjection()))
A.simulate!(f5, 1.0e-3; dt = 1.0e-3)
st5 = A._two_velocity_flow_workspace!(f5.workspace.two_phase, f5.system.two_phase)
check("mimetic system: MINRES (not BiCGSTAB), block-diagonal Cholesky", st5.projection_workspace.cache.krylov isa Main.Krylov.MinresWorkspace, string(typeof(st5.projection_workspace.cache.krylov)))

println("\n== iteration caps and defaults quoted in the implementation boxes")
check("default max_cycles: 100 (one phase), 500 (two phases)", A._navier_stokes_cycle_limit(single_flow().system, nothing) == 100 && A._navier_stokes_cycle_limit(drop_flow().system, nothing) == 500)
check("adaptivity default: no criteria", isempty(single_flow().system.adaptivity.criteria))
check("MeshAdaptationOptions defaults: update_interval 1, coarsen_delay 4", A.MeshAdaptationOptions().update_interval == 1 && A.MeshAdaptationOptions().coarsen_delay == 4)
check("default block size 16 x 16 cells", A.BlockTreeMesh().block_size == (16, 16))

finish("step03_options_defaults")

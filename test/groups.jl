# Paths are relative to test/. Integration entries include their prerequisites.
const TEST_GROUPS = [
    "mesh" => ["mesh/topology.jl", "mesh/variables.jl", "mesh/adaptivity.jl", "mesh/amr_connectivity.jl"],
    "threading" => ["threading.jl"],
    "indicators" => ["indicator/phase/anisotropic_cells.jl", "indicator/phase/levelset_initialization.jl", "indicator/phase/height_functions.jl", "indicator/geometry.jl"],
    "operators" => ["operators/advection/cell_advection.jl", "operators/gradient.jl", "operators/advection/staggered_advection.jl", "operators/staggered_diffusion.jl", "operators/poisson.jl"],
    "two_phase" => ["two_phase/transport.jl", "two_phase/phase_face_lifecycle.jl", "two_phase/two_velocity/inactive_velocity.jl", "two_phase/two_velocity/transport/two_velocity.jl", "two_phase/two_velocity/projection/mimetic_projection.jl", "two_phase/two_velocity/projection/cmdgfm_projection.jl", "two_phase/two_velocity/projection/cmdgfm_pressure_reuse.jl", "two_phase/two_velocity/projection/mimetic_projection_variants.jl", "two_phase/two_velocity/transport/two_velocity_wy_reference.jl", "two_phase/variable_staggered_diffusion.jl", "two_phase/forces.jl", "two_phase/two_velocity/diffusion.jl"],
    "navier_stokes" => ["navier_stokes/body_force.jl", "navier_stokes/two_phase_navier_stokes.jl", "navier_stokes/navier_stokes.jl", "navier_stokes/high_level_api.jl", "navier_stokes/configuration.jl", "readme.jl"],
    "diagnostics" => ["diagnostics.jl"],
    "examples" => ["examples/helpers.jl", "examples/basic_examples.jl", "examples/cylinder_example.jl", "examples/stokes3_example.jl", "examples/lgpi_example.jl", "examples/two_phase_examples.jl"],
    "solvers" => ["solvers.jl"],
    "recording" => ["recording/fields.jl", "recording/rasterize.jl", "recording/recorder.jl", "recording/diagnostics.jl"],
    "aqua" => ["aqua.jl"],
    "integration/wave_equation" => ["integration/wave_equation.jl"],
    "integration/staggered_wave_equation" => ["integration/staggered_wave_equation.jl"],
    "integration/amr_manufactured_flow" => ["integration/amr_manufactured_problem.jl", "integration/amr_manufactured_flow.jl"],
    "integration/two_phase_vortex" => ["integration/vortex_problem.jl", "integration/two_phase_vortex.jl"],
    "integration/two_velocity_transport" => ["integration/two_velocity_transport.jl"],
    "integration/two_velocity_rotation" => ["integration/two_velocity_rotation.jl"],
    "integration/two_velocity_vortex" => ["integration/vortex_problem.jl", "integration/two_velocity_vortex.jl"],
    "integration/two_velocity_instabilities" => ["integration/interfacial_instability_problem.jl", "integration/two_velocity_instabilities.jl"],
    "integration/two_phase_waves" => ["integration/two_phase_waves.jl"],
    "integration/two_phase_droplet" => ["integration/two_phase_droplet.jl"],
    "integration/two_phase_sessile_droplet" => ["integration/two_phase_sessile_droplet.jl"],
    "integration/anisotropic_cells" => ["integration/anisotropic_cells.jl"],
    "integration/single_phase_cylinder" => ["integration/cylinder_reference_problem.jl", "integration/single_phase_cylinder.jl"],
    "performance" => ["performance/performance.jl", "performance/two_phase_performance.jl"],
]

const TEST_HELPERS = [
    "test_utils.jl", "two_phase/two_velocity/helpers.jl",
    "two_phase/two_velocity/projection/cmdgfm_reference_helpers.jl",
    "embedded_operator_reference_helpers.jl",
    "integration/circle_geometry.jl",
]
# Entrypoints and the separately provisioned Makie suite are not included as tests.
const TEST_EXCLUSIONS = [
    "runtests.jl", "groups.jl", "integration/runtests.jl", "performance/runtests.jl",
    "integration/mimeticfv_reference.jl", "makie_visualization.jl",
]

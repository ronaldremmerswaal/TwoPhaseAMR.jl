# LS-STAG method
Add a geometry-presence AMR criterion (like the interface one)

For a more accurate discretisation of diffusion, consider the LS-STAG method, where we use PLIC instead of an explicit LevelSet (the level set may only be used for initializing the geometry fractions). The paper title is: "The LS-STAG method: A new immersed boundary/level-set method for the computation of incompressible viscous flows in complex moving geometries with good conservation properties". Implement this as an optional method, and determine using accuracy tests and benchmarks whether we should use it by default

# Wave equation on disk (using cut-cells)
Implement a wave equation solver on a disk using cut-cells. The initial condition should be an eigenmode of the disk, and the solution should be compared to the analytical solution. Start with a centred scalar wave equation with Dirichlet boundary conditions, and then implement a staggered wave equation with Neumann boundary conditions. Add an example to the examples folder, and an integration test/integration to the tests folder.

# Turbulence modelling
Implement QR model.

# Illustrative examples
Develop a series of illustrative examples that demonstrate the capabilities of the framework. Each results additionally in a .md file with a description of the problem, the setup, and the results.
- Lid-driven cavity flow
- Single-phase HIT turbulence
- Single-phase channel flow
- Capillary wave
- Stokes-3 wave
# Code refactor
Refactor the code to improve its structure and maintainability. 
Emphasize modularity, readability, and adherence to best practices. 
Ensure that the code is well-documented and includes comprehensive unit tests to validate functionality. 
Ensure that the high-level API is clear and intuitive for users.
Update the examples to be more concise.

# Documentation rewrite
Rewrite the documentation: 
- High-level API description
- Tutorials
  - Grid generation
  - Single-phase flow
  - Two-phase flow
  - Two-velocity modelling
- Add a markdown file to each example explaining the setup and showing some expected results.

# Technical report
Write a detailed technical report describing the numerical methods used in the code, using the style and notation of the PhD thesis.

# LS-STAG method
Add a geometry-presence AMR criterion (like the interface one)

For a more accurate discretisation of diffusion, consider the LS-STAG method, where we use PLIC instead of an explicit LevelSet (the level set may only be used for initializing the geometry fractions). The paper title is: "The LS-STAG method: A new immersed boundary/level-set method for the computation of incompressible viscous flows in complex moving geometries with good conservation properties". Implement this as an optional method, and determine using accuracy tests and benchmarks whether we should use it by default

Research: use LS-STAG for two-phase diffusion, one- and two-velocity

# Relative fractions
Investigate the role of relative fractions and relative apertures. Currently the mimetic gravity model seems incorrect at cut-cell boundaries.

# Wave equation on disk (using cut-cells)
Implement a wave equation solver on a disk using cut-cells. The initial condition should be an eigenmode of the disk, and the solution should be compared to the analytical solution. Start with a centred scalar wave equation with Dirichlet boundary conditions, and then implement a staggered wave equation with Neumann boundary conditions. Add an example to the examples folder, and an integration test/integration to the tests folder.

# Turbulence modelling
Implement QR model.

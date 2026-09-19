# CI failure
Currently the CI fails for several reasons

# Code refactor
Refactor the code to improve its structure and maintainability. 
Emphasize modularity, readability, and adherence to best practices. 

In particular, improve the file/folder structure: 
- src/mesh/...
- src/solvers/...
- src/variables/...
- src/single_phase/...
- src/indicator/... (interface, geometry, height functions, etc) 
- src/two_phase/...
- src/two_velocity/...
Think about this, I might have overlooked some important folders. Perhaps we can do the same for the tests folder?

Ensure that the code is well-documented and includes comprehensive unit tests to validate functionality. 
Ensure that the high-level API is clear and intuitive for users (currently adding diagnostics for kinetic energy/momentum/etc. is a bit cumbersome).
Make sure options are passed in a consistent way to the Navier--Stokes solver (e.g., CSF vs GFM, Frozen vs Projected, CMDGFM vs MimeticGradient, etc).
Update the examples to be more concise: they should be an inviting entry point for new users, right now they are unnecessarily long and complex.
Also, make sure that all two-phase examples can be run using both the one-velocity and two-velocity models. The instability example should use two-velocity by default.

# Conditional AMR criteria
It should be easily possible to combine multiple AMR criteria such that one can, for example, do vorticity-based refinement only at the interface, or the other way around: do vorticity-based refinement everywhere except at the interface. This should be possible without having to write a new AMR criterion.

# Example setup time
Starting an example currently takes a long time: why?

# Visualisation
Allow for more flexibility in visualisation
- multiple panels
- if single panel then zooming in/out should not restrict to the aspect ratio of the domain
- switch logarithmic scale on/off per axis for graphs
- Web-based visualisation?

# Documentation rewrite
Rewrite the documentation: 
- The main README is a mess, make it concise, linking to the documentation and examples. Show some example visualisations.
- High-level API description
- Tutorials
  - Grid generation
  - Single-phase flow
  - Two-phase flow
  - Two-velocity modelling
- Add a markdown file to each example explaining the setup and showing some expected results.

# Technical report
Write a detailed technical report describing the numerical methods used in the code, using the style and notation of the PhD thesis.
The key addition compared to the thesis is AMR: the thesis essentially assumes a uniform grid. 
Think about a concise notation for the AMR grid, and how to describe the AMR algorithm in a concise way. 
Moreover, the report should make a connection between the numerical methods described mathematically, and the actual implementation in the code. 

Make figures whenever appropriate, especially for AMR, cut-cells, interface reconstruction, advection, two-velocity, etc.
Add rigorous convergence tests and benchmarks to validate the numerical methods.
Before writing the report, make a detailed plan and write it to a new markdown file. The plan should include (at least) an overview of the report sections, and a list of figures, as well as a list of convergence tests and benchmarks to be included.

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

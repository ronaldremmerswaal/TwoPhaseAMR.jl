# 
# Mesh not coarsening Stokes-3 2-velocity
# CI failure
Currently the CI fails for several reasons

# Conditional AMR criteria
It should be easily possible to combine multiple AMR criteria such that one can, for example, do vorticity-based refinement only at the interface, or the other way around: do vorticity-based refinement everywhere except at the interface. This should be possible without having to write a new AMR criterion.
Hence, we need a AND (&&) and NOT (!) operator for AMR criteria

Moreover, the AMR initialisation is still quite verbose, can this be improved?

# Example setup time
Starting an example currently takes a long time: why?

# Visualisation
Allow for more flexibility in visualisation
- multiple panels
- if single panel then zooming in/out should not restrict to the aspect ratio of the domain
- switch logarithmic scale on/off per axis for graphs
- Web-based visualisation?

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

# AMR
I want to implement a framework for adaptive refinement: use indicators to label each active block for: :refine, :coarsen, :keep; and to then adapt the mesh keeping a 2:1 ratio in mind (graded mesh), while satisfying the user-requested modifications (with :refine taking top highest precedence, then :keep, and finally :coarsen). Any number of cell/face fields should then be updated along with the new mesh using the prolongation/restriction operators. This should avoid allocations by having a pre-allocated buffer of blocks for the mesh and the fields (which grows only if needed). Moreover, we need some ad-hoc strategy that ensures a block is not repeatedly refined and then coarsened in a short time span, so we should keep track of the last time a block was refined, and refuse coarsening it until a certain number of time steps have passed. Moreover, if the Navier--Stokes solver uses this, the number of times the mesh is updated should be limited to once every so many time steps
We will add specific refinement indicators at a later stage.

# Cut-cell
Implement a symmetry-preserving cut-cell method: "A symmetry-preserving Cartesian grid method for computing a viscous flow past a circular cylinder". The paper also provides a testcase that can be used to verify the implementation. 

# Turbulence modelling
Implement QR model.

# Illustrative examples
Develop a series of illustrative examples that demonstrate the capabilities of the framework. Each results additionally in a .md file with a description of the problem, the setup, and the results.
- Lid-driven cavity flow
- Single-phase HIT turbulence
- Single-phase channel flow
- Capillary wave
- Stokes-3 wave
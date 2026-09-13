# AMR
I want to implement a framework for adaptive refinement: use indicators to label each active block for: :refine, :coarsen, :keep; and to then adapt the mesh keeping a 2:1 ratio in mind (graded mesh), while satisfying the user-requested modifications (with :refine taking top highest precedence, then :keep, and finally :coarsen). Any number of cell/face fields should then be updated along with the new mesh using the prolongation/restriction operators. This should avoid allocations by having a pre-allocated buffer of blocks for the mesh and the fields (which grows only if needed). Moreover, we need some ad-hoc strategy that ensures a block is not repeatedly refined and then coarsened in a short time span, so we should keep track of the last time a block was refined, and refuse coarsening it until a certain number of time steps have passed. Moreover, if the Navier--Stokes solver uses this, the number of times the mesh is updated should be limited to once every so many time steps
We will add specific refinement indicators at a later stage.

# Cut-cell
Implement a symmetry-preserving cut-cell method: "A symmetry-preserving Cartesian grid method for computing a viscous flow past a circular cylinder". The paper also provides a testcase that can be used to verify the implementation. 

# Visualization
I want to implement a visualization framework that can handle the adaptive mesh and the associated fields in an interactive way in a GUI
- I'm not sure what is the best backend for this, perhaps it should be rendered in a browser? investigate the available options suitable for my requirements
- during simulation, the user should be able to visualize the mesh and the fields, and interact with them (e.g., zoom, pan, rotate, select blocks, etc.)
- the simulation should be pausable and resumable
- a fixed number of checkpoints should be saved during the simulation, allowing the user to visualize the state of the simulation at those points, while the simulation is still running
- the user should be able to switch back to live mode, while the simulation is still running
- for future developments, multiple cell-based fields and face-based fields (should be able to plot the magnitude or vorticity; no vector-valued plots) should be supported, and the user should be able to select which fields to visualize
- scalar quantities should be optionally visualized as function of time (e.g., the total momentum, energy, maximum velocity, etc.) in a single plot, at a higher frequency than the mesh and the fields. the checkpoints should also be visible in this plot, and the user should be able to select a checkpoint by clicking on it in the plot
- the visualization should be an object that can optionally be passed to the Navier--Stokes solver, this object is responsible for storing the checkpoints and the data associated with them, and for providing the visualization functionality. the solver merely calls an update function on the visualization object every step, the object determines whether to store a checkpoint or not, and the GUI is updated with the new data (always, even if the checkpoint is not stored)
Please evaluate my plan before starting implementation: perhaps I am being overly ambitious

# Turbulence modelling
Implement QR model.

# Illustrative examples
Develop a series of illustrative examples that demonstrate the capabilities of the framework. Each results additionally in a .md file with a description of the problem, the setup, and the results.
- Lid-driven cavity flow
- Single-phase HIT turbulence
- Single-phase channel flow
- Capillary wave
- Stokes-3 wave
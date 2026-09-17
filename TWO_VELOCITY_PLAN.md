# Two-velocity model
We will implement the two-velocity model described in the thesis (/home/ronald/Documents/phd_project/thesis/thesis.tex), this is a big effort, so I first want to discuss the implementation plan. Each subsection should be treated as a milestone and separate task. Propose a detailed implementation/research plan, with milestones, tasks, and gates

## Data structures
A velocity is active if the corresponding staggered phase fraction is greater than zero. Hence, there are faces with both velocities. So we need a data structure that can store two velocities on the same face. Probably the easiest is to define new data types that simply wrap two single-phase data types. Both for staggered and centred data

## Interface advection
This is a topic that requires some research. The thesis proposes two approaches: 
1. extend the liquid velocity field into the phase domain using a projection step, sufficiently far such that advection can be performed. this implies that only the liquid momentum is conserved, and the gas momentum is not. 
2. use the frozen-in-time cut-cells to define per-phase fluxes, a justification and discussion is found in the thesis. the tests in the thesis, however, show convergence issues with this approach. moreover, it was tested only with directionally unsplit advection methods (I want to stick with directionally split methods for now)
Try both approaches, and validate them with passive advection tests

## Momentum transport
This should be straightforward, provided with the currently available WY staggered scheme. The only difference is that we do not take the mass weighted average of the two velocities, but instead keep both separate.
Again, this should be validated with passive advection tests (see vortex reverse).
Details can be found in the thesis

## Pressure projection
We will discuss this later. Focus on passive transport first




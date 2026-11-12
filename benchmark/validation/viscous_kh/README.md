# Viscous Kelvin–Helmholtz instability: one- vs two-velocity model under mesh refinement

This folder contains an independent linear reference solver (two-fluid Orr–Sommerfeld
eigenvalue and initial-value problem), the TwoPhaseAMR study driver, the analysis
scripts, the result tables and the figures. Conclusions and discussion are in
[REPORT.md](REPORT.md).

## 1. Governing equations

Two immiscible incompressible fluids, liquid ($\ell$, below) and gas ($g$, above),
with constant densities $\rho_\pi$ and dynamic viscosities $\mu_\pi$
($\nu_\pi=\mu_\pi/\rho_\pi$), separated by an interface $\Gamma(t)$ with surface tension
$\sigma$ and no gravity:

$$
\rho_\pi(\partial_t\mathbf u+\mathbf u\cdot\nabla\mathbf u)=-\nabla p+\nabla\cdot(2\mu_\pi\mathbf D(\mathbf u)),\qquad
\nabla\cdot\mathbf u=0\quad\text{in phase }\pi,
$$

$$
[\![\mathbf u]\!]=0,\qquad [\![-p\,\mathbf I+2\mu\mathbf D(\mathbf u)]\!]\,\mathbf n=\sigma\kappa\,\mathbf n\quad\text{on }\Gamma,
$$

and $\Gamma$ moves with the flow. TwoPhaseAMR discretises these on a quadtree
(staggered velocities, geometric VOF/PLIC interface, conservative ghost-fluid surface
tension, implicit-midpoint viscous step; see `docs/techref`):

* **One-velocity model** (`OneVelocityModel()`): one face velocity per face, mixture
  density (arithmetic in the staggered liquid fraction), mixture viscosity (geometric
  mean of the phase viscosities), Poisson projection. The velocity is continuous by
  construction; an unresolved shear layer is smeared over at least one cell.
* **Two-velocity model** (`TwoVelocityModel()`): one velocity per phase on faces
  where that phase is present; phase momentum transport; the CMDGFM projection
  (Remmerswaal, PhD thesis, ch. 7) imposes continuity of the *normal* velocity only, so
  a tangential velocity jump (a vortex sheet that models an unresolved shear layer) is
  allowed at the interface. The viscous step diffuses the mass-weighted mixture
  velocity with the same variable-viscosity operator and adds the common increment to
  both phases (so it reduces to the one-velocity step when the jump vanishes).

## 2. Reference solution

**Problem.** Domain $(0,\lambda)\times(-H,H)$, $\lambda=1$ ($k=2\pi$), $H=\lambda/2$,
periodic in $x$, free-slip walls at $y=\pm H$. The base state is a flat interface
$y=0$ with the diffusing composite error-function shear layer (Rayleigh's
impulsively-started layer between two fluids; Yecko et al. 2002, Boeck & Zaleski
2005, Bagué et al. 2010)

$$
U_\pi(y,t)=U^\infty_\pi\,\operatorname{erf}\!\big(y/\delta_\pi(t)\big),\qquad
\delta_\pi(t)^2=\delta_{\pi,0}^2+4\nu_\pi t,\qquad
\frac{\delta_{\ell,0}}{\delta_{g,0}}=\sqrt{\frac{\nu_\ell}{\nu_g}},\qquad
\sqrt{\rho_\ell\mu_\ell}\,U^\infty_\ell=\sqrt{\rho_g\mu_g}\,U^\infty_g ,
$$

with $U^\infty_\ell+U^\infty_g=\Delta U=1$ (liquid moving in $-x$, gas in $+x$,
interface at rest). Velocity and shear stress are continuous at $y=0$ and the profile
is an exact solution of the two-fluid diffusion equation (the walls are $\ge 16\,\delta$
away, so they do not affect it to double precision). Surface tension is set by the
vortex-sheet Weber number $We=\rho_\ell\rho_g\Delta U^2/(k\sigma(\rho_\ell+\rho_g))=2$,
which makes the harmonics $2k,3k,\dots$ of a vortex sheet stable.

**Linear problem.** A perturbation $\psi=\phi_\pi(y,t)e^{ikx}$ (stream function,
$u=\partial_y\psi$, $v=-\partial_x\psi$) and interface displacement
$\eta=\hat\eta(t)e^{ikx}$ satisfy in each layer

$$
\partial_t(D^2-k^2)\phi=-ikU(D^2-k^2)\phi+ikU''\phi+\nu(D^2-k^2)^2\phi ,
$$

with interface conditions linearised at $y=0$ (Boomkamp et al. 1997, Bagué et al.
2010): continuity of $\phi$, of $D\phi+U'\hat\eta$ (tangential velocity) and of
$\mu(D^2\phi+k^2\phi+U''\hat\eta)$ (tangential stress), the normal-stress balance

$$
\Big[\rho\,\partial_tD\phi+ik\rho\,(UD\phi-U'\phi)-\mu(D^3-3k^2D)\phi\Big]_\ell^g-ik\sigma k^2\hat\eta=0,
$$

the kinematic condition $\partial_t\hat\eta=-ik\phi(0)$, and free-slip walls
$\phi=D^2\phi=0$. For a frozen base flow, $e^{-i\omega t}$ modes give the
generalised (Orr–Sommerfeld) eigenproblem for $\omega=\omega_r+i\omega_i$ (growth rate
$\omega_i$, phase speed $\omega_r/k$); for $\kappa\delta\to0$, $\nu\to0$ it tends to
Lamb's vortex-sheet relation
$\rho_\ell(\omega-kU_\ell)^2+\rho_g(\omega-kU_g)^2=\sigma k^3\tanh kH$.

**Base-flow diffusion.** The shear layer thickens by up to a factor 2.5 during the
runs; quasi-steady growth rates change by 2–40 % (table in
`results/reference_summary.txt`). Instead of maintaining the base flow by a body
force (which could not be represented when $\delta\ll h$) the reference is the
**linear initial-value problem**: the $t=0$ eigenmode evolved with the diffusing base
flow. The simulation is initialised with exactly the same eigenmode, so the complex
amplitude $\hat\eta(t)$ can be compared directly at every time.

**Discretisation and verification** (`os_reference.jl`, `verify_reference.jl`).
Each layer is a Chebyshev–Gauss–Lobatto collocation grid (optionally stretched
towards the interface by $|y|=H\sinh(\beta(1+\xi)/2)/\sinh\beta$ with exact
chain-rule derivative matrices); boundary and interface conditions replace the two
collocation rows next to each end (Boomkamp et al. 1997). The descriptor system
$M\dot X=A(t)X$ is solved by QZ (eigenproblem) or integrated with the three-stage
Radau IIA method (order 5, stiffly accurate). Verification
(`results/verify_reference.txt`):

| check | published | this solver |
|---|---|---|
| plane Poiseuille, $Re=10^4$, $\alpha=1$, as two identical layers (Orszag 1971) | $c=0.23752649+0.00373967i$ | $0.23752649+0.00373967i$ |
| Bagué et al. (2010) Table II, case A ($r=1$, $m=0.1$, $\sigma=0$), $\alpha c_i$ at $\alpha\delta_g=\pi/2$ | 0.1087018 | 0.1087019 |
| case B (as A, $We_g=10$) | 0.0937734 | 0.0937737 |
| case C ($r=0.1$, $m=0.99$, $\sigma=0$) | 0.200692 | 0.20065–0.20074 |
| case D (as C, $We_g=10$) | 0.188538 | 0.18849–0.18859 |

(Cases C/D: the spread over $N=100$–$130$ and stretching reflects collocation
round-off, as also reported by Bagué et al.) The study references are converged to
better than the tolerances in `results/reference_summary.txt` (time-step and
collocation checks, and an exact-exponential check of the time integrator with a
frozen base flow).

## 3. Comparison approach

**Cases** (`kh_cases.jl`):

| case | $\rho_\ell/\rho_g$ | $\nu_g/\nu_\ell$ | $\delta_{g,0}$ | $Re=\Delta U\delta_{g,0}/\nu_g$ | $T$ |
|---|---|---|---|---|---|
| `r1_thin` | 1 | 1 | $\lambda/128$ | 1000 | 1.5 |
| `r1_thick` | 1 | 1 | $\lambda/32$ | 100 | 3 |
| `r10_thin` | 10 | 4 | $\lambda/128$ | 1000 | 2.5 |
| `r10_thick` | 10 | 4 | $\lambda/32$ | 100 | 4 |

initial amplitude $|\hat\eta(0)|=2\cdot10^{-4}\lambda$.

**Initial data.** Interface $y=|\hat\eta_0|\cos kx$ (exact VOF initialisation) and the
eigenmode velocity added to the base flow. Staggered velocities are face-control-volume
averages of the exact field (essential when $\delta\ll h$): the one-velocity model
gets the mass-weighted mixture average, the two-velocity model the phase averages over
the part of the control volume occupied by each phase.

**Meshes.** Static graded quadtree: interface resolution $h=\lambda/n$,
$n\in\{16,\dots,512\}$, uniform within $|y|\le\lambda/8$, one level coarser per
doubling of the distance to the interface (base $16\times16$). The ladder spans
$\delta_{\ell,0}/h$ from $1/16$ (clearly unresolved) to $8$–$16$ (clearly resolved).

**Measures** (`analyze.jl`).
* $\hat\eta(t)$ = first Fourier coefficient of the interface height computed from the
  volume fractions every step ($\eta=\mathrm{Re}(\hat\eta e^{ikx})$, exact for cell
  columns of width $h$ after division by $\mathrm{sinc}^2(kh/2)$).
* growth rate and phase speed: least-squares slopes of $\log|\hat\eta|$ and of the
  unwrapped phase over $t\in[T/3,T]$, the same fit applied to the reference;
  errors relative (growth) and absolute in units of $\Delta U$ (phase speed);
* interface state: $|\hat\eta(T)-\hat\eta_{\rm ref}(T)|/|\hat\eta_{\rm ref}(T)|$;
* shear-layer state: RMS deviation over $|y|\le0.1$ of the $x$-averaged tangential
  velocity at $T$ from the exact base profile $U(y,T)$ (each phase's own velocity on
  its own side for the two-velocity model) and the phase velocity jump at the interface.
* cost: cells, time steps, wall-clock time (single thread).

## 4. Rerunning

```sh
cd benchmark/validation/viscous_kh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. verify_reference.jl                     # solver verification
julia --project=. make_reference.jl                       # results/reference_*.csv
julia --project=. kh_run.jl r1_thin,r10_thin both 16,32,64,128,256,512
julia --project=. kh_run.jl r1_thick,r10_thick both 16,32,64,128,256
julia --project=. kh_run.jl r1_thin both 32,512 field     # fields for the 2×2 figure
julia --project=. analyze.jl                              # results/tables.md, errors.csv
julia --project=. plots.jl r1_thin 32 512                 # figures/*.png
```

Use `OPENBLAS_NUM_THREADS=1 julia -t 1` on a shared machine; the $n=512$ runs take
the longest (see the wall-clock column of the tables).

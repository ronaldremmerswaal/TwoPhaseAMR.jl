# Instability accuracy: CMDGFM and inferred jumps

Generated 2026-09-26T13:05:17.376 UTC by `development/infer/instability_accuracy_overview.jl` with Julia 1.12.7.

All runs use a uniform mesh, frozen interface fluxes, the same aperture-based divergence, CFL 0.35, pressure tolerance 1e-7 and the case's linear-theory final time. The interface starts with amplitude 0.01 wavelength. `:infer_aperture` is the paired aperture normal; `:infer_plic` changes only that normal to the reconstructed PLIC normal. Both retain the existing diagonal K and symmetric KKT projection. The mimetic solver uses separate exact Cholesky factors for its independent preconditioner blocks. CMDGFM uses its own pressure-dependent jump inference.

CMDGFM rows already measured with the same Julia 1.12.7 and unchanged CMDGFM implementation were carried into the post-fix data file. Every inferred-method row in this report is recomputed with the separated exact-block preconditioner; pre-fix inferred rows are not used.

Amplitude error is the time-integrated complex first-mode L2 error relative to linear theory, normalized by the theory's time-integrated norm. A mesh rate is log2(error at n/2 / error at n); a time rate is log2(error at 2Δt / error at Δt). These are *analytical-error* rates; a non-increasing rate is not required for temporal boundedness. Ratios compare to CMDGFM at the same mesh and time-step factor.

Energy loss is `100 (E(0)-E(t))/S` in percent, where `E=kinetic+gravity potential+surface` and `S=kinetic(0)+|gravity potential(0)|+surface(0)`. Positive means loss, negative means gain. The measure includes errors from transport, body force, curvature, and projection; it is not solely the projection energy identity. Time-history samples are linearly interpolated at fractions of final time. The `max |drift|` column uses every recorded step.

| Case | ρₗ/ρg | We | Bo | Orientation | T |
|---|---:|---:|---:|---|---:|
| KH: density 1:1, Weber 0.5 | 1 | 0.5 | 0 | liquid below | 8.8858 |
| KH: density 1000:1, Weber 0.5 | 1e+03 | 0.5 | 0 | liquid below | 8.8858 |
| KH: density 1:1, Weber 5 | 1 | 5 | 0 | liquid below | 0.65848 |
| KH: density 1000:1, Weber 5 | 1e+03 | 5 | 0 | liquid below | 0.65848 |
| RT: density 3:1, Bond 0.5 | 3 | 0 | 0.5 | liquid above | 8.8858 |
| RT: density 1000:1, Bond 2 | 1e+03 | 0 | 2 | liquid above | 1.317 |
| Combined KH/RT: density 3:1, Weber 0.5, Bond 0.75 | 3 | 0.5 | 0.75 | liquid above | 2.6339 |
| Combined KH/RT: density 1000:1, Weber 0.5, Bond 0.75 | 1e+03 | 0.5 | 0.75 | liquid above | 2.6339 |

## KH: density 1:1, Weber 0.5

### Mesh refinement (time-step factor 1)

| Method | n | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 8 | 0.71 | — | 1 | -3.87 | 22.3 | 46 |
| `cmdgfm` | 16 | 0.414 | 0.778 | 1 | -0.898 | 14.6 | 128 |
| `cmdgfm` | 32 | 0.204 | 1.02 | 1 | -0.291 | 3.41 | 363 |
| `infer_aperture` | 8 | 0.774 | — | 1.09 | 0.481 | 17.7 | 46 |
| `infer_aperture` | 16 | 0.474 | 0.709 | 1.14 | -0.739 | 13.5 | 128 |
| `infer_aperture` | 32 | 0.282 | 0.75 | 1.38 | -0.287 | 4.37 | 363 |
| `infer_plic` | 8 | 0.729 | — | 1.03 | -3.1 | 21.5 | 46 |
| `infer_plic` | 16 | 0.462 | 0.66 | 1.12 | -0.856 | 17.8 | 128 |
| `infer_plic` | 32 | 0.279 | 0.726 | 1.37 | -0.28 | 6.1 | 363 |

### Time-step refinement (n=16)

| Method | Δt factor | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 1 | 0.414 | — | 1 | -0.898 | 14.6 | 128 |
| `cmdgfm` | 0.5 | 0.378 | 0.129 | 1 | -0.473 | 18.6 | 256 |
| `cmdgfm` | 0.25 | 0.359 | 0.0744 | 1 | -0.308 | 9.81 | 512 |
| `infer_aperture` | 1 | 0.474 | — | 1.14 | -0.739 | 13.5 | 128 |
| `infer_aperture` | 0.5 | 0.434 | 0.124 | 1.15 | -0.364 | 11 | 256 |
| `infer_aperture` | 0.25 | 0.424 | 0.0345 | 1.18 | -0.188 | 6.89 | 512 |
| `infer_plic` | 1 | 0.462 | — | 1.12 | -0.856 | 17.8 | 128 |
| `infer_plic` | 0.5 | 0.428 | 0.11 | 1.13 | -0.431 | 14 | 256 |
| `infer_plic` | 0.25 | 0.413 | 0.0505 | 1.15 | -0.273 | 10.1 | 512 |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -2.41 | -0.91 | -1.81 | -0.898 | 0.0259 | 14.6 | — |
| `infer_aperture` | 0 | -1.69 | -0.761 | -1.17 | -0.739 | 0.673 | 13.5 | 9.17e-14 |
| `infer_plic` | 0 | -1.71 | -0.869 | -1.26 | -0.856 | 0.651 | 17.8 | 1.21e-13 |

## KH: density 1000:1, Weber 0.5

### Mesh refinement (time-step factor 1)

| Method | n | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 8 | 1.75 | — | 1 | -2.79 | 8.32 | 758 |
| `cmdgfm` | 16 | 0.465 | 1.91 | 1 | -0.673 | 4.61 | 1508 |
| `cmdgfm` | 32 | 0.331 | 0.489 | 1 | -0.235 | 4.33 | 3026 |
| `infer_aperture` | 8 | 1.66 | — | 0.947 | 10.5 | 11.9 | 727 |
| `infer_aperture` | 16 | 1.58 | 0.0651 | 3.4 | 4.79 | 7.75 | 1463 |
| `infer_aperture` | 32 | 1.79 | -0.176 | 5.4 | 0.815 | 8.8 | 3049 |
| `infer_plic` | 8 | 1.72 | — | 0.984 | 2.1 | 9.26 | 725 |
| `infer_plic` | 16 | 1.01 | 0.765 | 2.18 | 2.42 | 6.61 | 1456 |
| `infer_plic` | 32 | 0.877 | 0.207 | 2.65 | 4.88 | 4.88 | 2960 |

### Time-step refinement (n=16)

| Method | Δt factor | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 1 | 0.465 | — | 1 | -0.673 | 4.61 | 1508 |
| `cmdgfm` | 0.5 | 0.505 | -0.121 | 1 | -0.404 | 2.58 | 3029 |
| `cmdgfm` | 0.25 | 0.497 | 0.0246 | 1 | 0.048 | 1.85 | 6051 |
| `infer_aperture` | 1 | 1.58 | — | 3.4 | 4.79 | 7.75 | 1463 |
| `infer_aperture` | 0.5 | 1.35 | 0.23 | 2.67 | 6.91 | 7.17 | 2920 |
| `infer_aperture` | 0.25 | 2.02 | -0.584 | 4.07 | -1.92 | 7.1 | 5849 |
| `infer_plic` | 1 | 1.01 | — | 2.18 | 2.42 | 6.61 | 1456 |
| `infer_plic` | 0.5 | 1.03 | -0.0201 | 2.03 | 2.8 | 6.42 | 2914 |
| `infer_plic` | 0.25 | 1.06 | -0.0454 | 2.13 | 4.98 | 5.84 | 5825 |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -4.6 | -1.3 | -1.8 | -0.673 | 0 | 4.61 | — |
| `infer_aperture` | 0 | 3.02 | 3.81 | 6.08 | 4.79 | 7.75 | 4.19 | 3.83e-12 |
| `infer_plic` | 0 | -0.77 | 2.85 | 4.11 | 2.42 | 5 | 6.61 | 4.42e-12 |

## KH: density 1:1, Weber 5

### Mesh refinement (time-step factor 1)

| Method | n | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 8 | 0.255 | — | 1 | -0.602 | 0.863 | 6 |
| `cmdgfm` | 16 | 0.13 | 0.97 | 1 | -0.233 | 0.391 | 12 |
| `cmdgfm` | 32 | 0.0666 | 0.966 | 1 | -0.124 | 0.161 | 27 |
| `infer_aperture` | 8 | 0.229 | — | 0.9 | -0.545 | 0.857 | 6 |
| `infer_aperture` | 16 | 0.121 | 0.92 | 0.932 | -0.246 | 0.393 | 12 |
| `infer_aperture` | 32 | 0.0631 | 0.941 | 0.948 | -0.115 | 0.162 | 27 |
| `infer_plic` | 8 | 0.239 | — | 0.939 | -0.561 | 0.849 | 6 |
| `infer_plic` | 16 | 0.124 | 0.95 | 0.953 | -0.225 | 0.391 | 12 |
| `infer_plic` | 32 | 0.0637 | 0.961 | 0.956 | -0.109 | 0.162 | 27 |

### Time-step refinement (n=16)

| Method | Δt factor | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 1 | 0.13 | — | 1 | -0.233 | 0.391 | 12 |
| `cmdgfm` | 0.5 | 0.0905 | 0.524 | 1 | -0.119 | 0.192 | 24 |
| `cmdgfm` | 0.25 | 0.0698 | 0.375 | 1 | -0.068 | 0.097 | 48 |
| `infer_aperture` | 1 | 0.121 | — | 0.932 | -0.246 | 0.393 | 12 |
| `infer_aperture` | 0.5 | 0.0794 | 0.611 | 0.877 | -0.132 | 0.193 | 25 |
| `infer_aperture` | 0.25 | 0.0639 | 0.312 | 0.916 | -0.0491 | 0.0977 | 49 |
| `infer_plic` | 1 | 0.124 | — | 0.953 | -0.225 | 0.391 | 12 |
| `infer_plic` | 0.5 | 0.0828 | 0.582 | 0.915 | -0.119 | 0.192 | 24 |
| `infer_plic` | 0.25 | 0.0599 | 0.467 | 0.858 | -0.0682 | 0.0981 | 48 |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -0.336 | -0.314 | -0.307 | -0.233 | 0 | 0.391 | — |
| `infer_aperture` | 0 | -0.344 | -0.31 | -0.297 | -0.246 | 0 | 0.393 | 1.34e-12 |
| `infer_plic` | 0 | -0.334 | -0.306 | -0.297 | -0.225 | 0 | 0.391 | 2.36e-12 |

## KH: density 1000:1, Weber 5

### Mesh refinement (time-step factor 1)

| Method | n | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 8 | 0.624 | — | 1 | -0.915 | 1.65 | 176 |
| `cmdgfm` | 16 | 0.169 | 1.89 | 1 | 0.626 | 0.626 | 376 |
| `cmdgfm` | 32 | 0.0725 | 1.22 | 1 | -1.56 | 3.03 | 803 |
| `infer_aperture` | 8 | 0.724 | — | 1.16 | 11 | 11 | 173 |
| `infer_aperture` | 16 | 0.564 | 0.361 | 3.35 | 5.8 | 5.81 | 348 |
| `infer_aperture` | 32 | 0.445 | 0.34 | 6.14 | 3.67 | 3.67 | 703 |
| `infer_plic` | 8 | 1.24 | — | 1.99 | 6.08 | 6.08 | 172 |
| `infer_plic` | 16 | 0.552 | 1.17 | 3.27 | 4.69 | 4.72 | 345 |
| `infer_plic` | 32 | 0.343 | 0.685 | 4.73 | 3.11 | 3.11 | 699 |

### Time-step refinement (n=16)

| Method | Δt factor | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 1 | 0.169 | — | 1 | 0.626 | 0.626 | 376 |
| `cmdgfm` | 0.5 | 0.126 | 0.425 | 1 | 0.338 | 0.405 | 741 |
| `cmdgfm` | 0.25 | 0.123 | 0.0327 | 1 | 0.244 | 0.343 | 1484 |
| `infer_aperture` | 1 | 0.564 | — | 3.35 | 5.8 | 5.81 | 348 |
| `infer_aperture` | 0.5 | 0.484 | 0.22 | 3.86 | 6.39 | 6.4 | 696 |
| `infer_aperture` | 0.25 | 0.515 | -0.0895 | 4.2 | 6.48 | 6.49 | 1389 |
| `infer_plic` | 1 | 0.552 | — | 3.27 | 4.69 | 4.72 | 345 |
| `infer_plic` | 0.5 | 0.505 | 0.128 | 4.02 | 4.56 | 4.57 | 692 |
| `infer_plic` | 0.25 | 0.429 | 0.236 | 3.49 | 5.7 | 5.7 | 1390 |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -0.166 | -0.245 | 0.371 | 0.626 | 0.626 | 0.307 | — |
| `infer_aperture` | 0 | 2.64 | 3.84 | 5.2 | 5.8 | 5.81 | -0 | 4.35e-11 |
| `infer_plic` | 0 | 1.42 | 3.11 | 4.44 | 4.69 | 4.72 | -0 | 3.72e-11 |

## RT: density 3:1, Bond 0.5

### Mesh refinement (time-step factor 1)

| Method | n | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 8 | 0.386 | — | 1 | -0.0198 | 3.51 | 46 |
| `cmdgfm` | 16 | 0.0816 | 2.24 | 1 | -0.0045 | 4.12 | 128 |
| `cmdgfm` | 32 | 0.00863 | 3.24 | 1 | 0.00198 | 0.915 | 363 |
| `infer_aperture` | 8 | 0.505 | — | 1.31 | 0.118 | 0.886 | 46 |
| `infer_aperture` | 16 | 0.221 | 1.19 | 2.71 | 0.0359 | 0.384 | 128 |
| `infer_aperture` | 32 | 0.124 | 0.835 | 14.3 | -0.0284 | 0.368 | 363 |
| `infer_plic` | 8 | 0.442 | — | 1.14 | 0.0595 | 1.28 | 46 |
| `infer_plic` | 16 | 0.215 | 1.04 | 2.63 | 0.0182 | 0.191 | 128 |
| `infer_plic` | 32 | 0.124 | 0.788 | 14.4 | -0.0279 | 0.312 | 363 |

### Time-step refinement (n=16)

| Method | Δt factor | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 1 | 0.0816 | — | 1 | -0.0045 | 4.12 | 128 |
| `cmdgfm` | 0.5 | 0.0711 | 0.197 | 1 | -0.00273 | 0.874 | 256 |
| `cmdgfm` | 0.25 | 0.0661 | 0.106 | 1 | -0.00178 | 0.951 | 512 |
| `infer_aperture` | 1 | 0.221 | — | 2.71 | 0.0359 | 0.384 | 128 |
| `infer_aperture` | 0.5 | 0.212 | 0.0554 | 2.99 | 0.0339 | 0.335 | 256 |
| `infer_aperture` | 0.25 | 0.212 | 0.00267 | 3.21 | 0.0371 | 0.212 | 512 |
| `infer_plic` | 1 | 0.215 | — | 2.63 | 0.0182 | 0.191 | 128 |
| `infer_plic` | 0.5 | 0.207 | 0.053 | 2.91 | 0.0242 | 0.204 | 256 |
| `infer_plic` | 0.25 | 0.203 | 0.0287 | 3.07 | 0.0265 | 0.253 | 512 |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -0.514 | -0.0032 | -0.318 | -0.0045 | 0.0643 | 4.12 | — |
| `infer_aperture` | 0 | -0.0803 | 0.099 | 0.127 | 0.0359 | 0.192 | 0.384 | 9e-13 |
| `infer_plic` | 0 | -0.0701 | 0.074 | 0.0759 | 0.0182 | 0.191 | 0.176 | 9.12e-13 |

## RT: density 1000:1, Bond 2

### Mesh refinement (time-step factor 1)

| Method | n | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 8 | 0.1 | — | 1 | -9.88e-06 | 0.0262 | 7 |
| `cmdgfm` | 16 | 0.0341 | 1.56 | 1 | 0.00212 | 0.00421 | 19 |
| `cmdgfm` | 32 | 0.0115 | 1.57 | 1 | 0.00284 | 0.00284 | 54 |
| `infer_aperture` | 8 | 0.0816 | — | 0.813 | -0.0413 | 0.0639 | 7 |
| `infer_aperture` | 16 | 0.109 | -0.413 | 3.19 | -0.0557 | 0.0573 | 43 |
| `infer_aperture` | 32 | 0.0896 | 0.277 | 7.81 | -0.0579 | 0.0581 | 116 |
| `infer_plic` | 8 | 0.0922 | — | 0.919 | -0.0335 | 0.051 | 7 |
| `infer_plic` | 16 | 0.102 | -0.147 | 3 | -0.0171 | 0.0171 | 45 |
| `infer_plic` | 32 | 0.0879 | 0.217 | 7.66 | -0.0587 | 0.0587 | 114 |

### Time-step refinement (n=16)

| Method | Δt factor | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 1 | 0.0341 | — | 1 | 0.00212 | 0.00421 | 19 |
| `cmdgfm` | 0.5 | 0.0227 | 0.589 | 1 | 0.00213 | 0.00264 | 38 |
| `cmdgfm` | 0.25 | 0.017 | 0.414 | 1 | 0.00268 | 0.00269 | 76 |
| `infer_aperture` | 1 | 0.109 | — | 3.19 | -0.0557 | 0.0573 | 43 |
| `infer_aperture` | 0.5 | 0.141 | -0.378 | 6.23 | -0.043 | 0.043 | 87 |
| `infer_aperture` | 0.25 | 0.157 | -0.152 | 9.22 | -0.0432 | 0.0442 | 166 |
| `infer_plic` | 1 | 0.102 | — | 3 | -0.0171 | 0.0171 | 45 |
| `infer_plic` | 0.5 | 0.134 | -0.392 | 5.91 | -0.00838 | 0.00953 | 91 |
| `infer_plic` | 0.25 | 0.149 | -0.155 | 8.77 | -0.0344 | 0.0353 | 176 |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -0.00117 | -0.00271 | -0.00109 | 0.00212 | 0.00212 | 0.00421 | — |
| `infer_aperture` | 0 | -0.00599 | -0.0125 | -0.0353 | -0.0557 | 0 | 0.0573 | 7.63e-14 |
| `infer_plic` | 0 | -0.00463 | -0.00741 | -0.00554 | -0.0171 | 0 | 0.0171 | 3.26e-13 |

## Combined KH/RT: density 3:1, Weber 0.5, Bond 0.75

### Mesh refinement (time-step factor 1)

| Method | n | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 8 | 0.242 | — | 1 | -0.25 | 0.361 | 14 |
| `cmdgfm` | 16 | 0.125 | 0.955 | 1 | -0.0561 | 0.111 | 38 |
| `cmdgfm` | 32 | 0.0699 | 0.838 | 1 | -0.0161 | 0.0382 | 108 |
| `infer_aperture` | 8 | 0.202 | — | 0.834 | -0.183 | 0.34 | 14 |
| `infer_aperture` | 16 | 0.0464 | 2.12 | 0.372 | -0.061 | 0.105 | 38 |
| `infer_aperture` | 32 | 0.0613 | -0.402 | 0.878 | -0.0483 | 0.0559 | 108 |
| `infer_plic` | 8 | 0.251 | — | 1.04 | -0.17 | 0.348 | 14 |
| `infer_plic` | 16 | 0.0597 | 2.07 | 0.478 | -0.0466 | 0.108 | 38 |
| `infer_plic` | 32 | 0.06 | -0.00659 | 0.858 | -0.0462 | 0.0547 | 108 |

### Time-step refinement (n=16)

| Method | Δt factor | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 1 | 0.125 | — | 1 | -0.0561 | 0.111 | 38 |
| `cmdgfm` | 0.5 | 0.111 | 0.171 | 1 | -0.0314 | 0.0589 | 76 |
| `cmdgfm` | 0.25 | 0.104 | 0.0868 | 1 | -0.0196 | 0.035 | 152 |
| `infer_aperture` | 1 | 0.0464 | — | 0.372 | -0.061 | 0.105 | 38 |
| `infer_aperture` | 0.5 | 0.0304 | 0.613 | 0.274 | -0.0293 | 0.0535 | 76 |
| `infer_aperture` | 0.25 | 0.0302 | 0.00617 | 0.289 | -0.0138 | 0.029 | 152 |
| `infer_plic` | 1 | 0.0597 | — | 0.478 | -0.0466 | 0.108 | 38 |
| `infer_plic` | 0.5 | 0.0489 | 0.287 | 0.441 | -0.0148 | 0.0575 | 76 |
| `infer_plic` | 0.25 | 0.0475 | 0.0415 | 0.455 | -0.00158 | 0.0326 | 152 |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -0.108 | -0.0764 | -0.0596 | -0.0561 | 0 | 0.111 | — |
| `infer_aperture` | 0 | -0.0946 | -0.0551 | -0.0601 | -0.061 | 0 | 0.105 | 3.65e-12 |
| `infer_plic` | 0 | -0.0977 | -0.0492 | -0.0565 | -0.0466 | 0 | 0.108 | 5.87e-12 |

## Combined KH/RT: density 1000:1, Weber 0.5, Bond 0.75

### Mesh refinement (time-step factor 1)

| Method | n | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 8 | 0.803 | — | 1 | -0.175 | 0.618 | 224 |
| `cmdgfm` | 16 | 0.245 | 1.72 | 1 | -0.112 | 0.138 | 458 |
| `cmdgfm` | 32 | 0.182 | 0.425 | 1 | -0.0394 | 0.07 | 929 |
| `infer_aperture` | 8 | 0.84 | — | 1.05 | 0.733 | 0.804 | 218 |
| `infer_aperture` | 16 | 1.21 | -0.527 | 4.95 | 0.0323 | 0.325 | 438 |
| `infer_aperture` | 32 | not run | — | — | not run | not run | — |
| `infer_plic` | 8 | not run | — | — | not run | not run | — |
| `infer_plic` | 16 | not run | — | — | not run | not run | — |
| `infer_plic` | 32 | not run | — | — | not run | not run | — |

### Time-step refinement (n=16)

| Method | Δt factor | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 1 | 0.245 | — | 1 | -0.112 | 0.138 | 458 |
| `cmdgfm` | 0.5 | 0.183 | 0.416 | 1 | -0.0494 | 0.146 | 921 |
| `cmdgfm` | 0.25 | 0.154 | 0.247 | 1 | -0.041 | 0.109 | 1846 |
| `infer_aperture` | 1 | 1.21 | — | 4.95 | 0.0323 | 0.325 | 438 |
| `infer_aperture` | 0.5 | not run | — | — | not run | not run | — |
| `infer_aperture` | 0.25 | not run | — | — | not run | not run | — |
| `infer_plic` | 1 | not run | — | — | not run | not run | — |
| `infer_plic` | 0.5 | not run | — | — | not run | not run | — |
| `infer_plic` | 0.25 | not run | — | — | not run | not run | — |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -0.121 | -0.0752 | -0.0896 | -0.112 | 0 | 0.138 | — |
| `infer_aperture` | 0 | 0.166 | 0.135 | -0.042 | 0.0323 | 0.217 | 0.325 | 1.48e-11 |
| `infer_plic` | not run | not run | not run | not run | not run | not run | not run | not run |

## Reproduction and interpretation

Run `julia --project=. development/infer/instability_accuracy_overview.jl`. The script resumes from `development/infer/instability_accuracy_results_fixed.tsv` and writes this report after every run. A failed run is recorded and is not silently omitted. To retry failed runs, remove only their TSV rows. The raw table preserves unrounded metrics. The benchmark uses the exact setup in `test/integration/interfacial_instability_problem.jl`.

A small projection energy-identity residual does not imply small total-energy drift, because the complete time step also transports the interface and applies gravity and capillarity. Likewise, agreement with linear theory can cease to be a pure discretization-error measure after strong nonlinear growth.

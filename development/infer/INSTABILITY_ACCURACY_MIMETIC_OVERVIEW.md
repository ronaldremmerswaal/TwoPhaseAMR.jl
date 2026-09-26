# Instability accuracy: CMDGFM vs two mimetic variants

Generated 2026-09-26T19:15:09.907 UTC by `development/infer/instability_accuracy_overview.jl` with Julia 1.12.7.

The comparison includes CMDGFM and both supported mimetic variants: joint-smoothed and joint-face. All runs use a uniform mesh, frozen interface fluxes, the same aperture-based divergence, CFL 0.35, pressure tolerance 1e-7 and the case's linear-theory final time. The interface starts with amplitude 0.01 wavelength. Both mimetic variants fit normal and tangential observations on local patches and retain the diagonal K and symmetric KKT projection. `mimetic_smoothed` starts from one-ring-smoothed aperture vectors; `mimetic_face` starts from CMDGFM-selected face rows. Their active-mass threshold is 1e-6. CMDGFM uses pressure-dependent jump inference.

CMDGFM and joint-smoothed rows were measured before the code cleanup and retained in the combined TSV. The joint-smoothed model parameters and algebra are unchanged. Joint-face rows are new.

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

## Observed convergence orders from endpoints

Each cell shows `fine-endpoint amplitude error (observed order)`. The order is `log(error at coarse endpoint / error at fine endpoint) / log(4)`. A negative value means the measured error increased.

### Mesh: n=8 to n=32 (time-step factor 1)

| Physics case | `cmdgfm` | `mimetic_smoothed` | `mimetic_face` |
|---|---:|---:|---:|
| KH: density 1:1, Weber 0.5 | 0.204 (0.899) | 0.813 (0.268) | 0.209 (0.85) |
| KH: density 1000:1, Weber 0.5 | 0.331 (1.2) | failed | 0.842 (-0.214) |
| KH: density 1:1, Weber 5 | 0.0666 (0.968) | 0.0449 (1.21) | 0.0741 (0.955) |
| KH: density 1000:1, Weber 5 | 0.0725 (1.55) | 0.122 (0.635) | 0.308 (0.25) |
| RT: density 3:1, Bond 0.5 | 0.00863 (2.74) | 0.672 (1.09) | 0.175 (0.721) |
| RT: density 1000:1, Bond 2 | 0.0115 (1.56) | 0.168 (0.365) | 0.0515 (-0.0292) |
| Combined KH/RT: density 3:1, Weber 0.5, Bond 0.75 | 0.0699 (0.897) | 0.214 (0.789) | 0.0624 (1.1) |
| Combined KH/RT: density 1000:1, Weber 0.5, Bond 0.75 | 0.182 (1.07) | 0.435 (0.498) | 0.363 (0.385) |

### Time step: factor 1 to 0.25 (n=16)

| Physics case | `cmdgfm` | `mimetic_smoothed` | `mimetic_face` |
|---|---:|---:|---:|
| KH: density 1:1, Weber 0.5 | 0.359 (0.102) | 1.46 (-0.223) | 0.389 (0.0439) |
| KH: density 1000:1, Weber 0.5 | 0.497 (-0.048) | 1.84 (0.383) | 0.903 (-0.111) |
| KH: density 1:1, Weber 5 | 0.0698 (0.449) | 0.0475 (0.634) | 0.0309 (1.01) |
| KH: density 1000:1, Weber 5 | 0.123 (0.229) | 0.112 (-0.0844) | 0.3 (0.407) |
| RT: density 3:1, Bond 0.5 | 0.0661 (0.152) | 1.51 (0.356) | 0.2 (0.094) |
| RT: density 1000:1, Bond 2 | 0.017 (0.501) | 0.209 (-0.0654) | 0.0541 (-0.221) |
| Combined KH/RT: density 3:1, Weber 0.5, Bond 0.75 | 0.104 (0.129) | 0.407 (-0.158) | 0.0903 (-0.0329) |
| Combined KH/RT: density 1000:1, Weber 0.5, Bond 0.75 | 0.154 (0.332) | 0.676 (-0.0868) | 0.27 (0.482) |

## KH: density 1:1, Weber 0.5

### Mesh refinement (time-step factor 1)

| Method | n | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 8 | 0.71 | — | 1 | -3.87 | 22.3 | 46 |
| `cmdgfm` | 16 | 0.414 | 0.778 | 1 | -0.898 | 14.6 | 128 |
| `cmdgfm` | 32 | 0.204 | 1.02 | 1 | -0.291 | 3.41 | 363 |
| `mimetic_smoothed` | 8 | 1.18 | — | 1.66 | -5.74 | 29.8 | 46 |
| `mimetic_smoothed` | 16 | 1.07 | 0.134 | 2.6 | -11.1 | 25.3 | 135 |
| `mimetic_smoothed` | 32 | 0.813 | 0.403 | 3.99 | -1.17 | 12.7 | 374 |
| `mimetic_face` | 8 | 0.678 | — | 0.956 | -4.9 | 27 | 46 |
| `mimetic_face` | 16 | 0.413 | 0.715 | 0.998 | -1.53 | 15.6 | 128 |
| `mimetic_face` | 32 | 0.209 | 0.985 | 1.02 | 0.243 | 1.7 | 363 |

### Time-step refinement (n=16)

| Method | Δt factor | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 1 | 0.414 | — | 1 | -0.898 | 14.6 | 128 |
| `cmdgfm` | 0.5 | 0.378 | 0.129 | 1 | -0.473 | 18.6 | 256 |
| `cmdgfm` | 0.25 | 0.359 | 0.0744 | 1 | -0.308 | 9.81 | 512 |
| `mimetic_smoothed` | 1 | 1.07 | — | 2.6 | -11.1 | 25.3 | 135 |
| `mimetic_smoothed` | 0.5 | 1.17 | -0.118 | 3.08 | -1.6 | 6.29 | 256 |
| `mimetic_smoothed` | 0.25 | 1.46 | -0.328 | 4.08 | -2.79 | 8.05 | 516 |
| `mimetic_face` | 1 | 0.413 | — | 0.998 | -1.53 | 15.6 | 128 |
| `mimetic_face` | 0.5 | 0.41 | 0.0098 | 1.08 | -0.475 | 5.59 | 256 |
| `mimetic_face` | 0.25 | 0.389 | 0.0779 | 1.08 | -0.783 | 3.44 | 512 |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -2.41 | -0.91 | -1.81 | -0.898 | 0.0259 | 14.6 | — |
| `mimetic_smoothed` | 0 | -4.31 | -25.3 | -4.76 | -11.1 | 0 | 25.3 | 3.39e-13 |
| `mimetic_face` | 0 | -4.38 | -1.82 | -7.54 | -1.53 | 0 | 15.6 | 3.48e-13 |

## KH: density 1000:1, Weber 0.5

### Mesh refinement (time-step factor 1)

| Method | n | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 8 | 1.75 | — | 1 | -2.79 | 8.32 | 758 |
| `cmdgfm` | 16 | 0.465 | 1.91 | 1 | -0.673 | 4.61 | 1508 |
| `cmdgfm` | 32 | 0.331 | 0.489 | 1 | -0.235 | 4.33 | 3026 |
| `mimetic_smoothed` | 8 | 3.2 | — | 1.83 | 2.31 | 4.61 | 732 |
| `mimetic_smoothed` | 16 | 3.13 | 0.0304 | 6.73 | -0.155 | 4.58 | 1493 |
| `mimetic_smoothed` | 32 | failed | — | — | failed | failed | 0 |
| `mimetic_face` | 8 | 0.626 | — | 0.358 | 9.14 | 10.7 | 745 |
| `mimetic_face` | 16 | 0.774 | -0.308 | 1.67 | 3.91 | 8.12 | 1509 |
| `mimetic_face` | 32 | 0.842 | -0.121 | 2.54 | 4.64 | 6.52 | 3183 |

### Time-step refinement (n=16)

| Method | Δt factor | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 1 | 0.465 | — | 1 | -0.673 | 4.61 | 1508 |
| `cmdgfm` | 0.5 | 0.505 | -0.121 | 1 | -0.404 | 2.58 | 3029 |
| `cmdgfm` | 0.25 | 0.497 | 0.0246 | 1 | 0.048 | 1.85 | 6051 |
| `mimetic_smoothed` | 1 | 3.13 | — | 6.73 | -0.155 | 4.58 | 1493 |
| `mimetic_smoothed` | 0.5 | 2.94 | 0.0913 | 5.81 | 3.72 | 4.96 | 2992 |
| `mimetic_smoothed` | 0.25 | 1.84 | 0.674 | 3.71 | -3.35 | 10.7 | 5956 |
| `mimetic_face` | 1 | 0.774 | — | 1.67 | 3.91 | 8.12 | 1509 |
| `mimetic_face` | 0.5 | 0.93 | -0.265 | 1.84 | 6.06 | 7.94 | 2955 |
| `mimetic_face` | 0.25 | 0.903 | 0.0438 | 1.82 | 5.15 | 7.86 | 6173 |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -4.6 | -1.3 | -1.8 | -0.673 | 0 | 4.61 | — |
| `mimetic_smoothed` | 0 | -0.876 | -1.82 | -1.23 | -0.155 | 2.02 | 4.58 | 1.94e-10 |
| `mimetic_face` | 0 | 2.15 | 4.02 | 5.67 | 3.91 | 8.12 | 2.39 | 3.72e-12 |

- Failed run `mimetic_smoothed` n=32 factor=1.0: pressure projection failed to converge (tolerance=1.0e-7, dimensionless_residual=1.0877602720387269e-6, dt=0.0017351123067814074, time=8.079641401145341, max_cycles=3000. Full diagnostics remain in the TSV.

## KH: density 1:1, Weber 5

### Mesh refinement (time-step factor 1)

| Method | n | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 8 | 0.255 | — | 1 | -0.602 | 0.863 | 6 |
| `cmdgfm` | 16 | 0.13 | 0.97 | 1 | -0.233 | 0.391 | 12 |
| `cmdgfm` | 32 | 0.0666 | 0.966 | 1 | -0.124 | 0.161 | 27 |
| `mimetic_smoothed` | 8 | 0.241 | — | 0.946 | -0.709 | 0.859 | 6 |
| `mimetic_smoothed` | 16 | 0.114 | 1.08 | 0.879 | -0.317 | 0.391 | 12 |
| `mimetic_smoothed` | 32 | 0.0449 | 1.35 | 0.674 | -0.224 | 0.237 | 27 |
| `mimetic_face` | 8 | 0.278 | — | 1.09 | -1.05 | 1.05 | 6 |
| `mimetic_face` | 16 | 0.125 | 1.16 | 0.959 | -0.196 | 0.368 | 12 |
| `mimetic_face` | 32 | 0.0741 | 0.754 | 1.11 | 0.299 | 0.299 | 28 |

### Time-step refinement (n=16)

| Method | Δt factor | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 1 | 0.13 | — | 1 | -0.233 | 0.391 | 12 |
| `cmdgfm` | 0.5 | 0.0905 | 0.524 | 1 | -0.119 | 0.192 | 24 |
| `cmdgfm` | 0.25 | 0.0698 | 0.375 | 1 | -0.068 | 0.097 | 48 |
| `mimetic_smoothed` | 1 | 0.114 | — | 0.879 | -0.317 | 0.391 | 12 |
| `mimetic_smoothed` | 0.5 | 0.0618 | 0.888 | 0.683 | -0.151 | 0.216 | 24 |
| `mimetic_smoothed` | 0.25 | 0.0475 | 0.38 | 0.68 | -0.0797 | 0.148 | 48 |
| `mimetic_face` | 1 | 0.125 | — | 0.959 | -0.196 | 0.368 | 12 |
| `mimetic_face` | 0.5 | 0.0774 | 0.691 | 0.855 | 0.318 | 0.363 | 28 |
| `mimetic_face` | 0.25 | 0.0309 | 1.33 | 0.442 | 0.298 | 0.298 | 50 |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -0.336 | -0.314 | -0.307 | -0.233 | 0 | 0.391 | — |
| `mimetic_smoothed` | 0 | -0.342 | -0.314 | -0.337 | -0.317 | 0 | 0.391 | 7.84e-13 |
| `mimetic_face` | 0 | -0.328 | -0.302 | -0.294 | -0.196 | 0 | 0.368 | 7.83e-12 |

## KH: density 1000:1, Weber 5

### Mesh refinement (time-step factor 1)

| Method | n | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 8 | 0.624 | — | 1 | -0.915 | 1.65 | 176 |
| `cmdgfm` | 16 | 0.169 | 1.89 | 1 | 0.626 | 0.626 | 376 |
| `cmdgfm` | 32 | 0.0725 | 1.22 | 1 | -1.56 | 3.03 | 803 |
| `mimetic_smoothed` | 8 | 0.294 | — | 0.471 | 2.25 | 2.25 | 176 |
| `mimetic_smoothed` | 16 | 0.0997 | 1.56 | 0.591 | 3.04 | 3.04 | 365 |
| `mimetic_smoothed` | 32 | 0.122 | -0.291 | 1.68 | 4 | 4 | 770 |
| `mimetic_face` | 8 | 0.435 | — | 0.698 | 7.24 | 7.24 | 177 |
| `mimetic_face` | 16 | 0.527 | -0.276 | 3.13 | 5.43 | 5.43 | 360 |
| `mimetic_face` | 32 | 0.308 | 0.776 | 4.25 | 4.32 | 4.32 | 732 |

### Time-step refinement (n=16)

| Method | Δt factor | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 1 | 0.169 | — | 1 | 0.626 | 0.626 | 376 |
| `cmdgfm` | 0.5 | 0.126 | 0.425 | 1 | 0.338 | 0.405 | 741 |
| `cmdgfm` | 0.25 | 0.123 | 0.0327 | 1 | 0.244 | 0.343 | 1484 |
| `mimetic_smoothed` | 1 | 0.0997 | — | 0.591 | 3.04 | 3.04 | 365 |
| `mimetic_smoothed` | 0.5 | 0.0934 | 0.0935 | 0.744 | 1.76 | 1.76 | 718 |
| `mimetic_smoothed` | 0.25 | 0.112 | -0.262 | 0.913 | 1.32 | 1.32 | 1434 |
| `mimetic_face` | 1 | 0.527 | — | 3.13 | 5.43 | 5.43 | 360 |
| `mimetic_face` | 0.5 | 0.408 | 0.369 | 3.25 | 5.8 | 5.8 | 714 |
| `mimetic_face` | 0.25 | 0.3 | 0.446 | 2.44 | 6.89 | 6.89 | 1429 |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -0.166 | -0.245 | 0.371 | 0.626 | 0.626 | 0.307 | — |
| `mimetic_smoothed` | 0 | 0.0245 | 0.43 | 1.32 | 3.04 | 3.04 | 0.186 | 4.02e-11 |
| `mimetic_face` | 0 | 2.25 | 3.84 | 4.89 | 5.43 | 5.43 | 0.135 | 2.47e-11 |

## RT: density 3:1, Bond 0.5

### Mesh refinement (time-step factor 1)

| Method | n | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 8 | 0.386 | — | 1 | -0.0198 | 3.51 | 46 |
| `cmdgfm` | 16 | 0.0816 | 2.24 | 1 | -0.0045 | 4.12 | 128 |
| `cmdgfm` | 32 | 0.00863 | 3.24 | 1 | 0.00198 | 0.915 | 363 |
| `mimetic_smoothed` | 8 | 3.04 | — | 7.86 | -1.17 | 3.3 | 46 |
| `mimetic_smoothed` | 16 | 2.48 | 0.293 | 30.4 | -0.817 | 1.56 | 142 |
| `mimetic_smoothed` | 32 | 0.672 | 1.88 | 77.9 | -0.276 | 1.01 | 402 |
| `mimetic_face` | 8 | 0.476 | — | 1.23 | -0.0758 | 1.21 | 46 |
| `mimetic_face` | 16 | 0.228 | 1.06 | 2.79 | -0.0526 | 1.09 | 128 |
| `mimetic_face` | 32 | 0.175 | 0.378 | 20.3 | -0.00117 | 0.405 | 363 |

### Time-step refinement (n=16)

| Method | Δt factor | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 1 | 0.0816 | — | 1 | -0.0045 | 4.12 | 128 |
| `cmdgfm` | 0.5 | 0.0711 | 0.197 | 1 | -0.00273 | 0.874 | 256 |
| `cmdgfm` | 0.25 | 0.0661 | 0.106 | 1 | -0.00178 | 0.951 | 512 |
| `mimetic_smoothed` | 1 | 2.48 | — | 30.4 | -0.817 | 1.56 | 142 |
| `mimetic_smoothed` | 0.5 | 5 | -1.01 | 70.3 | -0.372 | 1.4 | 271 |
| `mimetic_smoothed` | 0.25 | 1.51 | 1.72 | 22.9 | -0.27 | 0.998 | 539 |
| `mimetic_face` | 1 | 0.228 | — | 2.79 | -0.0526 | 1.09 | 128 |
| `mimetic_face` | 0.5 | 0.216 | 0.0734 | 3.04 | -0.00396 | 0.755 | 256 |
| `mimetic_face` | 0.25 | 0.2 | 0.115 | 3.03 | -0.0448 | 0.679 | 513 |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -0.514 | -0.0032 | -0.318 | -0.0045 | 0.0643 | 4.12 | — |
| `mimetic_smoothed` | 0 | -0.421 | -0.357 | -0.434 | -0.817 | 0.119 | 1.56 | 9.37e-12 |
| `mimetic_face` | 0 | -0.102 | 0.0586 | -0.118 | -0.0526 | 0.119 | 1.09 | 8.34e-13 |

## RT: density 1000:1, Bond 2

### Mesh refinement (time-step factor 1)

| Method | n | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 8 | 0.1 | — | 1 | -9.88e-06 | 0.0262 | 7 |
| `cmdgfm` | 16 | 0.0341 | 1.56 | 1 | 0.00212 | 0.00421 | 19 |
| `cmdgfm` | 32 | 0.0115 | 1.57 | 1 | 0.00284 | 0.00284 | 54 |
| `mimetic_smoothed` | 8 | 0.279 | — | 2.78 | 0.00431 | 0.221 | 34 |
| `mimetic_smoothed` | 16 | 0.191 | 0.546 | 5.6 | 0.0515 | 0.0774 | 37 |
| `mimetic_smoothed` | 32 | 0.168 | 0.184 | 14.7 | -0.107 | 0.108 | 65 |
| `mimetic_face` | 8 | 0.0495 | — | 0.493 | 0.0596 | 0.129 | 17 |
| `mimetic_face` | 16 | 0.0398 | 0.314 | 1.17 | -0.0412 | 0.0896 | 51 |
| `mimetic_face` | 32 | 0.0515 | -0.372 | 4.49 | -0.0379 | 0.0393 | 128 |

### Time-step refinement (n=16)

| Method | Δt factor | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 1 | 0.0341 | — | 1 | 0.00212 | 0.00421 | 19 |
| `cmdgfm` | 0.5 | 0.0227 | 0.589 | 1 | 0.00213 | 0.00264 | 38 |
| `cmdgfm` | 0.25 | 0.017 | 0.414 | 1 | 0.00268 | 0.00269 | 76 |
| `mimetic_smoothed` | 1 | 0.191 | — | 5.6 | 0.0515 | 0.0774 | 37 |
| `mimetic_smoothed` | 0.5 | 0.2 | -0.07 | 8.85 | 0.0353 | 0.0842 | 54 |
| `mimetic_smoothed` | 0.25 | 0.209 | -0.0607 | 12.3 | 0.0577 | 0.0845 | 108 |
| `mimetic_face` | 1 | 0.0398 | — | 1.17 | -0.0412 | 0.0896 | 51 |
| `mimetic_face` | 0.5 | 0.0638 | -0.68 | 2.81 | -0.0102 | 0.0658 | 99 |
| `mimetic_face` | 0.25 | 0.0541 | 0.238 | 3.18 | -0.0464 | 0.0491 | 172 |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -0.00117 | -0.00271 | -0.00109 | 0.00212 | 0.00212 | 0.00421 | — |
| `mimetic_smoothed` | 0 | -0.0369 | 0.0107 | 0.0708 | 0.0515 | 0.0774 | 0.0415 | 1.77e-13 |
| `mimetic_face` | 0 | -0.0798 | -0.0599 | -0.0239 | -0.0412 | 0 | 0.0896 | 1.23e-13 |

## Combined KH/RT: density 3:1, Weber 0.5, Bond 0.75

### Mesh refinement (time-step factor 1)

| Method | n | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 8 | 0.242 | — | 1 | -0.25 | 0.361 | 14 |
| `cmdgfm` | 16 | 0.125 | 0.955 | 1 | -0.0561 | 0.111 | 38 |
| `cmdgfm` | 32 | 0.0699 | 0.838 | 1 | -0.0161 | 0.0382 | 108 |
| `mimetic_smoothed` | 8 | 0.638 | — | 2.64 | -0.871 | 0.957 | 15 |
| `mimetic_smoothed` | 16 | 0.327 | 0.965 | 2.62 | -0.358 | 0.436 | 40 |
| `mimetic_smoothed` | 32 | 0.214 | 0.613 | 3.06 | -0.189 | 0.219 | 108 |
| `mimetic_face` | 8 | 0.286 | — | 1.18 | -0.433 | 0.456 | 15 |
| `mimetic_face` | 16 | 0.0863 | 1.73 | 0.691 | -0.0236 | 0.144 | 38 |
| `mimetic_face` | 32 | 0.0624 | 0.468 | 0.893 | -0.0186 | 0.0551 | 108 |

### Time-step refinement (n=16)

| Method | Δt factor | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 1 | 0.125 | — | 1 | -0.0561 | 0.111 | 38 |
| `cmdgfm` | 0.5 | 0.111 | 0.171 | 1 | -0.0314 | 0.0589 | 76 |
| `cmdgfm` | 0.25 | 0.104 | 0.0868 | 1 | -0.0196 | 0.035 | 152 |
| `mimetic_smoothed` | 1 | 0.327 | — | 2.62 | -0.358 | 0.436 | 40 |
| `mimetic_smoothed` | 0.5 | 0.37 | -0.177 | 3.33 | -0.395 | 0.469 | 78 |
| `mimetic_smoothed` | 0.25 | 0.407 | -0.14 | 3.9 | -0.239 | 0.329 | 158 |
| `mimetic_face` | 1 | 0.0863 | — | 0.691 | -0.0236 | 0.144 | 38 |
| `mimetic_face` | 0.5 | 0.0471 | 0.874 | 0.424 | -0.0325 | 0.124 | 76 |
| `mimetic_face` | 0.25 | 0.0903 | -0.94 | 0.865 | -0.0107 | 0.0766 | 152 |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -0.108 | -0.0764 | -0.0596 | -0.0561 | 0 | 0.111 | — |
| `mimetic_smoothed` | 0 | -0.0386 | -0.26 | -0.36 | -0.358 | 0.0156 | 0.436 | 1.53e-11 |
| `mimetic_face` | 0 | -0.139 | -0.081 | -0.0973 | -0.0236 | 0 | 0.144 | 7.2e-12 |

## Combined KH/RT: density 1000:1, Weber 0.5, Bond 0.75

### Mesh refinement (time-step factor 1)

| Method | n | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 8 | 0.803 | — | 1 | -0.175 | 0.618 | 224 |
| `cmdgfm` | 16 | 0.245 | 1.72 | 1 | -0.112 | 0.138 | 458 |
| `cmdgfm` | 32 | 0.182 | 0.425 | 1 | -0.0394 | 0.07 | 929 |
| `mimetic_smoothed` | 8 | 0.868 | — | 1.08 | 0.126 | 0.353 | 223 |
| `mimetic_smoothed` | 16 | 0.599 | 0.533 | 2.45 | 0.0548 | 0.142 | 454 |
| `mimetic_smoothed` | 32 | 0.435 | 0.463 | 2.39 | -0.208 | 0.283 | 960 |
| `mimetic_face` | 8 | 0.618 | — | 0.769 | 0.639 | 0.681 | 230 |
| `mimetic_face` | 16 | 0.527 | 0.23 | 2.16 | 0.256 | 0.481 | 467 |
| `mimetic_face` | 32 | 0.363 | 0.54 | 1.99 | 0.24 | 0.287 | 971 |

### Time-step refinement (n=16)

| Method | Δt factor | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 1 | 0.245 | — | 1 | -0.112 | 0.138 | 458 |
| `cmdgfm` | 0.5 | 0.183 | 0.416 | 1 | -0.0494 | 0.146 | 921 |
| `cmdgfm` | 0.25 | 0.154 | 0.247 | 1 | -0.041 | 0.109 | 1846 |
| `mimetic_smoothed` | 1 | 0.599 | — | 2.45 | 0.0548 | 0.142 | 454 |
| `mimetic_smoothed` | 0.5 | 0.617 | -0.0411 | 3.37 | 0.0409 | 0.0724 | 908 |
| `mimetic_smoothed` | 0.25 | 0.676 | -0.132 | 4.38 | 0.0776 | 0.0779 | 1819 |
| `mimetic_face` | 1 | 0.527 | — | 2.16 | 0.256 | 0.481 | 467 |
| `mimetic_face` | 0.5 | 0.571 | -0.115 | 3.12 | 0.369 | 0.444 | 918 |
| `mimetic_face` | 0.25 | 0.27 | 1.08 | 1.75 | 0.331 | 0.517 | 1833 |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -0.121 | -0.0752 | -0.0896 | -0.112 | 0 | 0.138 | — |
| `mimetic_smoothed` | 0 | -0.0966 | -0.0954 | -0.0834 | 0.0548 | 0.129 | 0.142 | 2.23e-11 |
| `mimetic_face` | 0 | 0.246 | 0.167 | 0.171 | 0.256 | 0.481 | 0.139 | 8.36e-12 |

## Reproduction and interpretation

Run `julia --project=. development/infer/instability_accuracy_overview.jl`. Set `STUDY_METHODS=mimetic_face` to rerun only joint-face rows. `STUDY_IMPORT` can import completed rows from another TSV. Historical `infer_joint_smoothed` and `mimetic` rows are recognized as `mimetic_smoothed`. The script resumes from the selected TSV and writes this report after every run. A failed run is recorded and is not silently omitted. To retry failed runs, remove only their TSV rows. The raw table preserves unrounded metrics. The benchmark uses the exact setup in `test/integration/interfacial_instability_problem.jl`.

A small projection energy-identity residual does not imply small total-energy drift, because the complete time step also transports the interface and applies gravity and capillarity. Likewise, agreement with linear theory can cease to be a pure discretization-error measure after strong nonlinear growth.

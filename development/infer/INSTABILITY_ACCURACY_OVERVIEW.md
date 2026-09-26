# Instability accuracy: CMDGFM and inferred jumps

Generated 2026-09-26T14:03:08.780 UTC by `development/infer/instability_accuracy_overview.jl` with Julia 1.12.7.

The comparison includes CMDGFM and four mimetic inferred-jump variants. All runs use a uniform mesh, frozen interface fluxes, the same aperture-based divergence, CFL 0.35, pressure tolerance 1e-7 and the case's linear-theory final time. The interface starts with amplitude 0.01 wavelength. `:infer_aperture` uses raw cell-integrated D(a_l n), `:infer_smoothed` uses a one-ring sum of aligned cut-cell aperture vectors, and `:infer_plic` uses the reconstructed PLIC normal. These three use the same paired K-orthogonal tangent/normal split. `:infer_smoothed_stencil` uses the smoothed aperture normal interpolated to faces but CMDGFM's beta-based facewise choice between imposed and inferred jump components. On unmatched paired faces, both aperture variants use the raw vector interpolated from neighboring cells. All four share the existing diagonal K and symmetric KKT projection. The mimetic solver uses separate exact Cholesky factors for its independent preconditioner blocks. CMDGFM uses its own pressure-dependent jump inference.

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

## Observed convergence orders from endpoints

Each cell shows `fine-endpoint amplitude error (observed order)`. The order is `log(error at coarse endpoint / error at fine endpoint) / log(4)`. A negative value means the measured error increased.

### Mesh: n=8 to n=32 (time-step factor 1)

| Physics case | `cmdgfm` | `infer_aperture` | `infer_smoothed` | `infer_smoothed_stencil` | `infer_plic` |
|---|---:|---:|---:|---:|---:|
| KH: density 1:1, Weber 0.5 | 0.204 (0.899) | 0.282 (0.73) | 0.279 (0.688) | 0.278 (0.694) | 0.279 (0.693) |
| KH: density 1000:1, Weber 0.5 | 0.331 (1.2) | 1.79 (-0.0555) | 1.24 (0.635) | 1.03 (0.396) | 0.877 (0.486) |
| KH: density 1:1, Weber 5 | 0.0666 (0.968) | 0.0631 (0.93) | 0.0637 (0.979) | 0.0607 (1.04) | 0.0637 (0.955) |
| KH: density 1000:1, Weber 5 | 0.0725 (1.55) | 0.445 (0.351) | 0.354 (0.761) | 0.161 (0.585) | 0.343 (0.928) |
| RT: density 3:1, Bond 0.5 | 0.00863 (2.74) | 0.124 (1.01) | 0.124 (0.941) | 0.188 (0.543) | 0.124 (0.915) |
| RT: density 1000:1, Bond 2 | 0.0115 (1.56) | 0.0896 (-0.0677) | 0.0867 (0.0363) | 0.108 (0.255) | 0.0879 (0.0348) |
| Combined KH/RT: density 3:1, Weber 0.5, Bond 0.75 | 0.0699 (0.897) | 0.0613 (0.859) | 0.0632 (0.982) | 0.0649 (1.29) | 0.06 (1.03) |
| Combined KH/RT: density 1000:1, Weber 0.5, Bond 0.75 | 0.182 (1.07) | 0.565 (0.286) | 0.663 (0.981) | 0.376 (-0.198) | 0.36 (0.895) |

### Time step: factor 1 to 0.25 (n=16)

| Physics case | `cmdgfm` | `infer_aperture` | `infer_smoothed` | `infer_smoothed_stencil` | `infer_plic` |
|---|---:|---:|---:|---:|---:|
| KH: density 1:1, Weber 0.5 | 0.359 (0.102) | 0.424 (0.0793) | 0.41 (0.0807) | 0.486 (-0.0434) | 0.413 (0.0802) |
| KH: density 1000:1, Weber 0.5 | 0.497 (-0.048) | 2.02 (-0.177) | 1.25 (0.139) | 0.992 (0.304) | 1.06 (-0.0328) |
| KH: density 1:1, Weber 5 | 0.0698 (0.449) | 0.0639 (0.462) | 0.0606 (0.52) | 0.061 (0.517) | 0.0599 (0.525) |
| KH: density 1000:1, Weber 5 | 0.123 (0.229) | 0.515 (0.0654) | 0.439 (0.0809) | 0.14 (0.222) | 0.429 (0.182) |
| RT: density 3:1, Bond 0.5 | 0.0661 (0.152) | 0.212 (0.029) | 0.203 (0.0358) | 0.258 (-0.0112) | 0.203 (0.0408) |
| RT: density 1000:1, Bond 2 | 0.017 (0.501) | 0.157 (-0.265) | 0.147 (-0.29) | 0.203 (-0.232) | 0.149 (-0.273) |
| Combined KH/RT: density 3:1, Weber 0.5, Bond 0.75 | 0.104 (0.129) | 0.0302 (0.31) | 0.0485 (0.193) | 0.135 (0.0407) | 0.0475 (0.164) |
| Combined KH/RT: density 1000:1, Weber 0.5, Bond 0.75 | 0.154 (0.332) | 0.994 (0.142) | 0.658 (0.341) | 0.312 (0.073) | 0.686 (0.332) |

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
| `infer_smoothed` | 8 | 0.723 | — | 1.02 | -2.45 | 23.3 | 46 |
| `infer_smoothed` | 16 | 0.459 | 0.657 | 1.11 | -0.742 | 25.7 | 128 |
| `infer_smoothed` | 32 | 0.279 | 0.718 | 1.37 | -0.281 | 5.55 | 363 |
| `infer_smoothed_stencil` | 8 | 0.729 | — | 1.03 | -4.46 | 22.3 | 46 |
| `infer_smoothed_stencil` | 16 | 0.457 | 0.673 | 1.1 | -1.11 | 14.5 | 128 |
| `infer_smoothed_stencil` | 32 | 0.278 | 0.716 | 1.36 | -0.308 | 7.58 | 363 |
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
| `infer_smoothed` | 1 | 0.459 | — | 1.11 | -0.742 | 25.7 | 128 |
| `infer_smoothed` | 0.5 | 0.425 | 0.11 | 1.12 | -0.401 | 17.4 | 256 |
| `infer_smoothed` | 0.25 | 0.41 | 0.0513 | 1.14 | -0.265 | 11.9 | 512 |
| `infer_smoothed_stencil` | 1 | 0.457 | — | 1.1 | -1.11 | 14.5 | 128 |
| `infer_smoothed_stencil` | 0.5 | 0.436 | 0.0677 | 1.15 | -0.402 | 12.5 | 256 |
| `infer_smoothed_stencil` | 0.25 | 0.486 | -0.155 | 1.35 | -0.541 | 6 | 512 |
| `infer_plic` | 1 | 0.462 | — | 1.12 | -0.856 | 17.8 | 128 |
| `infer_plic` | 0.5 | 0.428 | 0.11 | 1.13 | -0.431 | 14 | 256 |
| `infer_plic` | 0.25 | 0.413 | 0.0505 | 1.15 | -0.273 | 10.1 | 512 |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -2.41 | -0.91 | -1.81 | -0.898 | 0.0259 | 14.6 | — |
| `infer_aperture` | 0 | -1.69 | -0.761 | -1.17 | -0.739 | 0.673 | 13.5 | 9.17e-14 |
| `infer_smoothed` | 0 | -1.92 | -0.869 | -1.29 | -0.742 | 0.613 | 25.7 | 9.89e-14 |
| `infer_smoothed_stencil` | 0 | -1.73 | -0.813 | -1.18 | -1.11 | 0.199 | 14.5 | 3.08e-13 |
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
| `infer_smoothed` | 8 | 3 | — | 1.72 | 6.88 | 8.25 | 726 |
| `infer_smoothed` | 16 | 1.52 | 0.982 | 3.27 | 4.51 | 6.13 | 1468 |
| `infer_smoothed` | 32 | 1.24 | 0.288 | 3.75 | -1.05 | 3.77 | 2993 |
| `infer_smoothed_stencil` | 8 | 1.79 | — | 1.02 | -4.75 | 9.51 | 729 |
| `infer_smoothed_stencil` | 16 | 1.51 | 0.245 | 3.25 | 1.86 | 16.8 | 1460 |
| `infer_smoothed_stencil` | 32 | 1.03 | 0.548 | 3.12 | -11 | 18.5 | 3097 |
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
| `infer_smoothed` | 1 | 1.52 | — | 3.27 | 4.51 | 6.13 | 1468 |
| `infer_smoothed` | 0.5 | 1.53 | -0.0145 | 3.03 | 5.73 | 7.38 | 2920 |
| `infer_smoothed` | 0.25 | 1.25 | 0.293 | 2.52 | 5.68 | 7.96 | 5846 |
| `infer_smoothed_stencil` | 1 | 1.51 | — | 3.25 | 1.86 | 16.8 | 1460 |
| `infer_smoothed_stencil` | 0.5 | 1.21 | 0.319 | 2.4 | -0.475 | 9.4 | 2920 |
| `infer_smoothed_stencil` | 0.25 | 0.992 | 0.29 | 2 | -4.47 | 9.23 | 5919 |
| `infer_plic` | 1 | 1.01 | — | 2.18 | 2.42 | 6.61 | 1456 |
| `infer_plic` | 0.5 | 1.03 | -0.0201 | 2.03 | 2.8 | 6.42 | 2914 |
| `infer_plic` | 0.25 | 1.06 | -0.0454 | 2.13 | 4.98 | 5.84 | 5825 |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -4.6 | -1.3 | -1.8 | -0.673 | 0 | 4.61 | — |
| `infer_aperture` | 0 | 3.02 | 3.81 | 6.08 | 4.79 | 7.75 | 4.19 | 3.83e-12 |
| `infer_smoothed` | 0 | 0.628 | 3.41 | 2.63 | 4.51 | 6.13 | 4.22 | 9.24e-12 |
| `infer_smoothed_stencil` | 0 | -0.709 | -10.6 | 2.39 | 1.86 | 3.64 | 16.8 | 7.16e-12 |
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
| `infer_smoothed` | 8 | 0.248 | — | 0.971 | -0.61 | 0.827 | 6 |
| `infer_smoothed` | 16 | 0.125 | 0.991 | 0.957 | -0.233 | 0.388 | 12 |
| `infer_smoothed` | 32 | 0.0637 | 0.966 | 0.957 | -0.108 | 0.162 | 27 |
| `infer_smoothed_stencil` | 8 | 0.255 | — | 1 | -0.548 | 0.835 | 6 |
| `infer_smoothed_stencil` | 16 | 0.125 | 1.03 | 0.959 | -0.192 | 0.388 | 12 |
| `infer_smoothed_stencil` | 32 | 0.0607 | 1.04 | 0.91 | -0.105 | 0.162 | 27 |
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
| `infer_smoothed` | 1 | 0.125 | — | 0.957 | -0.233 | 0.388 | 12 |
| `infer_smoothed` | 0.5 | 0.0832 | 0.581 | 0.92 | -0.125 | 0.191 | 24 |
| `infer_smoothed` | 0.25 | 0.0606 | 0.459 | 0.868 | -0.0788 | 0.107 | 48 |
| `infer_smoothed_stencil` | 1 | 0.125 | — | 0.959 | -0.192 | 0.388 | 12 |
| `infer_smoothed_stencil` | 0.5 | 0.0835 | 0.579 | 0.923 | -0.0772 | 0.191 | 24 |
| `infer_smoothed_stencil` | 0.25 | 0.061 | 0.454 | 0.873 | -0.0304 | 0.0986 | 48 |
| `infer_plic` | 1 | 0.124 | — | 0.953 | -0.225 | 0.391 | 12 |
| `infer_plic` | 0.5 | 0.0828 | 0.582 | 0.915 | -0.119 | 0.192 | 24 |
| `infer_plic` | 0.25 | 0.0599 | 0.467 | 0.858 | -0.0682 | 0.0981 | 48 |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -0.336 | -0.314 | -0.307 | -0.233 | 0 | 0.391 | — |
| `infer_aperture` | 0 | -0.344 | -0.31 | -0.297 | -0.246 | 0 | 0.393 | 1.34e-12 |
| `infer_smoothed` | 0 | -0.339 | -0.307 | -0.298 | -0.233 | 0 | 0.388 | 1.88e-12 |
| `infer_smoothed_stencil` | 0 | -0.338 | -0.303 | -0.282 | -0.192 | 0 | 0.388 | 6.16e-13 |
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
| `infer_smoothed` | 8 | 1.01 | — | 1.63 | 10.2 | 10.2 | 172 |
| `infer_smoothed` | 16 | 0.491 | 1.05 | 2.91 | 4.58 | 4.58 | 347 |
| `infer_smoothed` | 32 | 0.354 | 0.474 | 4.87 | 3.07 | 3.13 | 699 |
| `infer_smoothed_stencil` | 8 | 0.362 | — | 0.58 | 0.494 | 0.535 | 173 |
| `infer_smoothed_stencil` | 16 | 0.19 | 0.929 | 1.13 | 3.03 | 3.03 | 347 |
| `infer_smoothed_stencil` | 32 | 0.161 | 0.241 | 2.22 | 2.64 | 2.64 | 708 |
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
| `infer_smoothed` | 1 | 0.491 | — | 2.91 | 4.58 | 4.58 | 347 |
| `infer_smoothed` | 0.5 | 0.361 | 0.445 | 2.87 | 4.34 | 4.34 | 695 |
| `infer_smoothed` | 0.25 | 0.439 | -0.283 | 3.58 | 4.79 | 4.79 | 1389 |
| `infer_smoothed_stencil` | 1 | 0.19 | — | 1.13 | 3.03 | 3.03 | 347 |
| `infer_smoothed_stencil` | 0.5 | 0.139 | 0.454 | 1.11 | 3.49 | 3.49 | 696 |
| `infer_smoothed_stencil` | 0.25 | 0.14 | -0.00951 | 1.14 | 3.49 | 3.49 | 1394 |
| `infer_plic` | 1 | 0.552 | — | 3.27 | 4.69 | 4.72 | 345 |
| `infer_plic` | 0.5 | 0.505 | 0.128 | 4.02 | 4.56 | 4.57 | 692 |
| `infer_plic` | 0.25 | 0.429 | 0.236 | 3.49 | 5.7 | 5.7 | 1390 |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -0.166 | -0.245 | 0.371 | 0.626 | 0.626 | 0.307 | — |
| `infer_aperture` | 0 | 2.64 | 3.84 | 5.2 | 5.8 | 5.81 | -0 | 4.35e-11 |
| `infer_smoothed` | 0 | 2 | 3.56 | 4.19 | 4.58 | 4.58 | -0 | 3.48e-11 |
| `infer_smoothed_stencil` | 0 | 0.599 | 1.57 | 2.31 | 3.03 | 3.03 | 0.0265 | 4.87e-11 |
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
| `infer_smoothed` | 8 | 0.457 | — | 1.18 | -0.0162 | 1.49 | 46 |
| `infer_smoothed` | 16 | 0.213 | 1.1 | 2.61 | 0.0324 | 0.283 | 128 |
| `infer_smoothed` | 32 | 0.124 | 0.781 | 14.4 | -0.0276 | 0.359 | 363 |
| `infer_smoothed_stencil` | 8 | 0.399 | — | 1.03 | 0.181 | 1.16 | 46 |
| `infer_smoothed_stencil` | 16 | 0.254 | 0.654 | 3.11 | 0.0573 | 1.09 | 128 |
| `infer_smoothed_stencil` | 32 | 0.188 | 0.433 | 21.8 | 0.0122 | 0.482 | 363 |
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
| `infer_smoothed` | 1 | 0.213 | — | 2.61 | 0.0324 | 0.283 | 128 |
| `infer_smoothed` | 0.5 | 0.206 | 0.0502 | 2.89 | 0.0242 | 0.208 | 256 |
| `infer_smoothed` | 0.25 | 0.203 | 0.0214 | 3.07 | 0.0371 | 0.304 | 512 |
| `infer_smoothed_stencil` | 1 | 0.254 | — | 3.11 | 0.0573 | 1.09 | 128 |
| `infer_smoothed_stencil` | 0.5 | 0.251 | 0.0146 | 3.53 | -0.0236 | 0.938 | 256 |
| `infer_smoothed_stencil` | 0.25 | 0.258 | -0.037 | 3.9 | -0.286 | 0.604 | 512 |
| `infer_plic` | 1 | 0.215 | — | 2.63 | 0.0182 | 0.191 | 128 |
| `infer_plic` | 0.5 | 0.207 | 0.053 | 2.91 | 0.0242 | 0.204 | 256 |
| `infer_plic` | 0.25 | 0.203 | 0.0287 | 3.07 | 0.0265 | 0.253 | 512 |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -0.514 | -0.0032 | -0.318 | -0.0045 | 0.0643 | 4.12 | — |
| `infer_aperture` | 0 | -0.0803 | 0.099 | 0.127 | 0.0359 | 0.192 | 0.384 | 9e-13 |
| `infer_smoothed` | 0 | -0.0727 | 0.0722 | 0.102 | 0.0324 | 0.19 | 0.283 | 9.4e-13 |
| `infer_smoothed_stencil` | 0 | -0.143 | 0.0861 | 0.0586 | 0.0573 | 0.296 | 1.09 | 2e-12 |
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
| `infer_smoothed` | 8 | 0.0912 | — | 0.908 | -0.0346 | 0.0522 | 7 |
| `infer_smoothed` | 16 | 0.0985 | -0.112 | 2.89 | -0.0294 | 0.0294 | 46 |
| `infer_smoothed` | 32 | 0.0867 | 0.184 | 7.56 | -0.064 | 0.064 | 114 |
| `infer_smoothed_stencil` | 8 | 0.153 | — | 1.53 | 0.0885 | 0.0885 | 7 |
| `infer_smoothed_stencil` | 16 | 0.147 | 0.0596 | 4.31 | -0.0328 | 0.0525 | 45 |
| `infer_smoothed_stencil` | 32 | 0.108 | 0.45 | 9.38 | -0.0445 | 0.0762 | 142 |
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
| `infer_smoothed` | 1 | 0.0985 | — | 2.89 | -0.0294 | 0.0294 | 46 |
| `infer_smoothed` | 0.5 | 0.131 | -0.408 | 5.77 | -0.0258 | 0.0258 | 92 |
| `infer_smoothed` | 0.25 | 0.147 | -0.172 | 8.65 | -0.0409 | 0.0419 | 173 |
| `infer_smoothed_stencil` | 1 | 0.147 | — | 4.31 | -0.0328 | 0.0525 | 45 |
| `infer_smoothed_stencil` | 0.5 | 0.185 | -0.334 | 8.18 | 0.0149 | 0.0523 | 74 |
| `infer_smoothed_stencil` | 0.25 | 0.203 | -0.13 | 11.9 | 0.0215 | 0.0575 | 102 |
| `infer_plic` | 1 | 0.102 | — | 3 | -0.0171 | 0.0171 | 45 |
| `infer_plic` | 0.5 | 0.134 | -0.392 | 5.91 | -0.00838 | 0.00953 | 91 |
| `infer_plic` | 0.25 | 0.149 | -0.155 | 8.77 | -0.0344 | 0.0353 | 176 |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -0.00117 | -0.00271 | -0.00109 | 0.00212 | 0.00212 | 0.00421 | — |
| `infer_aperture` | 0 | -0.00599 | -0.0125 | -0.0353 | -0.0557 | 0 | 0.0573 | 7.63e-14 |
| `infer_smoothed` | 0 | -0.00532 | -0.0123 | -0.0152 | -0.0294 | 0 | 0.0294 | 9.11e-14 |
| `infer_smoothed_stencil` | 0 | 0.0094 | 0.0271 | -0.0311 | -0.0328 | 0.0525 | 0.0516 | 1.76e-13 |
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
| `infer_smoothed` | 8 | 0.247 | — | 1.02 | -0.205 | 0.338 | 14 |
| `infer_smoothed` | 16 | 0.0633 | 1.96 | 0.507 | -0.0444 | 0.107 | 38 |
| `infer_smoothed` | 32 | 0.0632 | 0.00258 | 0.905 | -0.0487 | 0.0573 | 108 |
| `infer_smoothed_stencil` | 8 | 0.387 | — | 1.6 | -0.115 | 0.379 | 14 |
| `infer_smoothed_stencil` | 16 | 0.142 | 1.44 | 1.14 | -0.0121 | 0.112 | 38 |
| `infer_smoothed_stencil` | 32 | 0.0649 | 1.13 | 0.929 | -0.00208 | 0.0374 | 108 |
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
| `infer_smoothed` | 1 | 0.0633 | — | 0.507 | -0.0444 | 0.107 | 38 |
| `infer_smoothed` | 0.5 | 0.0539 | 0.232 | 0.486 | -0.0114 | 0.0564 | 76 |
| `infer_smoothed` | 0.25 | 0.0485 | 0.154 | 0.464 | -0.00194 | 0.0316 | 152 |
| `infer_smoothed_stencil` | 1 | 0.142 | — | 1.14 | -0.0121 | 0.112 | 38 |
| `infer_smoothed_stencil` | 0.5 | 0.123 | 0.21 | 1.11 | 0.0103 | 0.0611 | 76 |
| `infer_smoothed_stencil` | 0.25 | 0.135 | -0.129 | 1.29 | 0.0218 | 0.0382 | 152 |
| `infer_plic` | 1 | 0.0597 | — | 0.478 | -0.0466 | 0.108 | 38 |
| `infer_plic` | 0.5 | 0.0489 | 0.287 | 0.441 | -0.0148 | 0.0575 | 76 |
| `infer_plic` | 0.25 | 0.0475 | 0.0415 | 0.455 | -0.00158 | 0.0326 | 152 |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -0.108 | -0.0764 | -0.0596 | -0.0561 | 0 | 0.111 | — |
| `infer_aperture` | 0 | -0.0946 | -0.0551 | -0.0601 | -0.061 | 0 | 0.105 | 3.65e-12 |
| `infer_smoothed` | 0 | -0.0951 | -0.0485 | -0.0536 | -0.0444 | 0 | 0.107 | 7.17e-12 |
| `infer_smoothed_stencil` | 0 | -0.0843 | -0.0377 | -0.0268 | -0.0121 | 0 | 0.112 | 7.01e-12 |
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
| `infer_aperture` | 32 | 0.565 | 1.1 | 3.1 | -0.206 | 0.43 | 922 |
| `infer_smoothed` | 8 | 2.58 | — | 3.22 | 0.13 | 0.405 | 220 |
| `infer_smoothed` | 16 | 1.05 | 1.29 | 4.31 | 0.131 | 0.356 | 437 |
| `infer_smoothed` | 32 | 0.663 | 0.669 | 3.64 | -0.347 | 0.376 | 929 |
| `infer_smoothed_stencil` | 8 | 0.286 | — | 0.356 | 0.134 | 0.272 | 219 |
| `infer_smoothed_stencil` | 16 | 0.346 | -0.273 | 1.41 | 0.142 | 0.195 | 441 |
| `infer_smoothed_stencil` | 32 | 0.376 | -0.123 | 2.07 | -0.374 | 0.553 | 944 |
| `infer_plic` | 8 | 1.24 | — | 1.55 | -0.307 | 0.593 | 216 |
| `infer_plic` | 16 | 1.09 | 0.196 | 4.44 | 0.25 | 0.401 | 438 |
| `infer_plic` | 32 | 0.36 | 1.59 | 1.98 | -0.00986 | 0.378 | 982 |

### Time-step refinement (n=16)

| Method | Δt factor | Amplitude error | Rate | / CMDGFM | Final energy loss % | Max abs drift % | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 1 | 0.245 | — | 1 | -0.112 | 0.138 | 458 |
| `cmdgfm` | 0.5 | 0.183 | 0.416 | 1 | -0.0494 | 0.146 | 921 |
| `cmdgfm` | 0.25 | 0.154 | 0.247 | 1 | -0.041 | 0.109 | 1846 |
| `infer_aperture` | 1 | 1.21 | — | 4.95 | 0.0323 | 0.325 | 438 |
| `infer_aperture` | 0.5 | 0.751 | 0.689 | 4.1 | 0.479 | 0.495 | 886 |
| `infer_aperture` | 0.25 | 0.994 | -0.405 | 6.44 | 0.371 | 0.422 | 1755 |
| `infer_smoothed` | 1 | 1.05 | — | 4.31 | 0.131 | 0.356 | 437 |
| `infer_smoothed` | 0.5 | 1.03 | 0.0326 | 5.63 | 0.135 | 0.278 | 877 |
| `infer_smoothed` | 0.25 | 0.658 | 0.649 | 4.26 | 0.119 | 0.429 | 1746 |
| `infer_smoothed_stencil` | 1 | 0.346 | — | 1.41 | 0.142 | 0.195 | 441 |
| `infer_smoothed_stencil` | 0.5 | 0.247 | 0.482 | 1.35 | 0.02 | 0.134 | 883 |
| `infer_smoothed_stencil` | 0.25 | 0.312 | -0.336 | 2.02 | -0.146 | 0.216 | 1770 |
| `infer_plic` | 1 | 1.09 | — | 4.44 | 0.25 | 0.401 | 438 |
| `infer_plic` | 0.5 | 1.01 | 0.1 | 5.53 | 0.0874 | 0.267 | 875 |
| `infer_plic` | 0.25 | 0.686 | 0.563 | 4.44 | 0.0893 | 0.313 | 1747 |

### Energy loss over time (n=16, Δt factor 1)

| Method | 0 | 0.25T | 0.5T | 0.75T | T | Peak loss % | Peak gain % | Max projection identity residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmdgfm` | 0 | -0.121 | -0.0752 | -0.0896 | -0.112 | 0 | 0.138 | — |
| `infer_aperture` | 0 | 0.166 | 0.135 | -0.042 | 0.0323 | 0.217 | 0.325 | 1.48e-11 |
| `infer_smoothed` | 0 | -0.11 | 0.0801 | 0.0454 | 0.131 | 0.151 | 0.356 | 9.21e-12 |
| `infer_smoothed_stencil` | 0 | -0.0259 | 0.14 | 0.0934 | 0.142 | 0.149 | 0.195 | 6.41e-12 |
| `infer_plic` | 0 | -0.149 | 0.0319 | -0.0957 | 0.25 | 0.259 | 0.401 | 5.49e-12 |

## Reproduction and interpretation

Run `julia --project=. development/infer/instability_accuracy_overview.jl`. The script resumes from `development/infer/instability_accuracy_results_fixed.tsv` and writes this report after every run. A failed run is recorded and is not silently omitted. To retry failed runs, remove only their TSV rows. The raw table preserves unrounded metrics. The benchmark uses the exact setup in `test/integration/interfacial_instability_problem.jl`.

A small projection energy-identity residual does not imply small total-energy drift, because the complete time step also transports the interface and applies gravity and capillarity. Likewise, agreement with linear theory can cease to be a pure discretization-error measure after strong nonlinear growth.

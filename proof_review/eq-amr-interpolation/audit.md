# Audit: eq:amr:interpolation — factor 4 under coarsening and the hysteresis condition $4E_{\rm coarsen}\le E_{\rm refine}$

- Source: `docs/techref/chapters/12_amr_criteria_transfer.tex:260-287` (`eq:amr:interpolation` l. 265; the factor-4 sentence l. 274-278; measurement l. 781-782), code `src/mesh/adaptivity/indicators.jl:refinement_labels!` (`InterpolationErrorIndicator`), `docs/techref/figures/amr.jl:amr_indicator_check`, commit `2959760`
- Date: 2026-11-10
- Overall status: ⚠️ claim verified in substance; the wording "only if $4E_{\rm coarsen}\le E_{\rm refine}$" is a conservative sufficient condition (the exact threshold is the measured ratio, $<4$ at finite resolution, $\to4$ as $h\to0$) and holds for smooth data only
- Rerun: `julia --project=@claude-symbolics proof_review/eq-amr-interpolation/scripts/run_all.jl`

## Statement and assumptions

$E_c=\dfrac{|a_c-\frac14\sum_k(a_{c-e_k}+a_{c+e_k})|}{\max\{\varepsilon_{\rm rel},|a_c|,\max_{k,\pm}|a_{c\pm e_k}|\}}$, block value $E_b=\max_cE_c$, labels by `eq:amr:error-labels`. Claims: for a smooth field the numerator is a second difference, $O(h^2)$, "so coarsening a block multiplies its value by about four" (measured 3.83); "a merged block whose children were below $E_{\rm coarsen}$ is then not refined again only if $4E_{\rm coarsen}\le E_{\rm refine}$; the code requires only $E_{\rm coarsen}<E_{\rm refine}$".

Assumptions: smooth field (second differences scale as $h^2$); the merged block's values are the cell means of the children (restriction); the merged block's neighbours are read through the halo of its own level.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| Package labels = replica of `eq:amr:interpolation` + `eq:amr:error-labels` | l. 265-272, 43-55 | ✅ | 744 block labels on a composite mesh (smooth and noisy fields, 3 wavenumbers) equal the independent replica (I1) | `code_01_ratio_hysteresis.jl` | also compared by `amr.jl` (316 labels) |
| Numerator is a second difference, $O(h^2)$ | l. 274 | ✅ | $u-\frac14\sum=-\frac14(D_x+D_y)$ with $D$ the second differences, exact | — | |
| Ratio of second differences at spacing $2h$ and $h$ | l. 275 | ✅ | exact 1D identity $u_{i+2}+u_{i-2}-2u_i=D_{i+1}+D_{i-1}+2D_i$ (so the ratio is $\le4$ for any data in 1D) and for a harmonic $\sin kx$ the ratio is $4\cos^2(kh/2)\le4$ (I2) | same | in 2D the coarse numerator mixes $D_x$ and $D_y$ at different cells, so no general bound by 4 exists for non-smooth data (a field with $D_x+D_y=0$ at every fine cell has $E=0$ and may have $E_{\rm coarse}\ne0$) |
| Measured factor 3.83 | l. 781-782 | ✅ | chapter fixture $1+\tfrac12\sin2\pi x\cos2\pi y$, $E(32)/E(64)=3.8268$ (I3); series $n\to2n$: $3.403,\,3.827,\,3.955,\,3.989$ for $n=16,32,64,128$: tends to 4 from below | same | |
| Same factor by actual coarsening (adapt!, cell-mean restriction, merged halo) | l. 275-276 | ✅ | $E(\text{merged})/\max E(\text{children})=3.351$ ($16\!\leftarrow\!32$ cells, exact cell averages), the point-sample series gives $3.403$ at $n=16$ | same | the restriction adds a small cell-average correction |
| Hysteresis: not refined again only if $4E_c\le E_r$ | l. 276-278 | ⚠️ true as a sufficient condition; the exact condition is $E_r\ge E_0=\text{ratio}\cdot E_c$ (for a family at the threshold) | static smooth field, $E_c=1.05\max E_{\rm children}$: $E_r=1.5E_c$ and $3.0E_c$ (below $E_0=3.19E_c$) → merge at attempt 1, refine again at attempt 2 (16 blocks); $E_r=3.4,3.9,4.2E_c$ → stays merged (I4) | same | $E_0/E_c=3.351/1.05=3.19$ at this resolution; $3.83$ would be the threshold at $n=32\to64$ |

## Errors and consequences

No error. Wording: the sentence l. 276-278 reads as a necessary condition ("not refined again *only if* $4E_c\le E_r$"). It is a conservative sufficient condition: the exact no-re-refinement condition for a smooth field is $E_r\ge\rho_hE_c$ with $\rho_h=E(2h)/E(h)=4\cos^2(kh/2)\cdots<4$ (3.35, 3.83, 3.95, 3.99 for 16, 32, 64, 128 cells). Suggested LaTeX:

```latex
For a smooth field the numerator is a second difference,
$O(\cellmax{c}^2)$, so coarsening a block multiplies its value by a factor that
tends to four from below as the mesh is refined (\cref{sec:amr:measurements});
a merged block whose children were below $\tolcoarsen$ is then not refined
again if $4\tolcoarsen\le\tolrefine$, and may be if
$\tolrefine<\tolcoarsen\cdot\text{(that factor)}$; the code requires only
$\tolcoarsen<\tolrefine$. For fields that are not resolved on the mesh the
factor has no bound.
```

(The "static smooth field" re-refinement is a *one-time* event in the test: the prolonged field then keeps the children above $E_c$, so it does not cycle forever; in a simulation the field is re-computed and the cycle can persist.)

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| Factor 4 | **sharp as a limit**: $4\cos^2(kh/2)\to4$; numerically $3.40,3.83,3.95,3.99$ | I2, I3 | `code_01_ratio_hysteresis.jl` | state "$\le4$ for smooth data, $\to4$" |
| $4E_c\le E_r$ sufficient | **likely sharp** as $h\to0$; conservative by $4-3.83=4\,\%$ at the chapter fixture and $16\,\%$ at 16 cells | I4: re-refinement occurs exactly when $E_r<E_0$ | same | use the measured factor of the resolution in use |
| Smoothness assumption | **necessary**: no bound for rough fields (the ratio of $E$ can be arbitrarily large when $D_x+D_y$ cancels at the fine level) | argument (I2: the 2D numerator is $-\frac14(D_x+D_y)$ and the coarse one combines $D_x$, $D_y$ at different cells) | — | — |
| "the code requires only $E_c<E_r$" | true | constructor throws unless $0\le E_c<E_r$ (read from `indicators.jl`) | — | — |

## Open points

- The relative floor $\varepsilon_{\rm rel}$ and fields with zero crossings change the denominator, hence the factor: not studied.
- Mesh interfaces (halo slots coarse-to-fine are piecewise constant) make $E$ at a level change larger than the smooth-field value for blocks next to a coarser block; the check above uses a uniform mesh.

## History

- 2026-11-10: first audit (commit `2959760`): I1–I4 pass; `run_all.jl` passes.

# Copied-source regular-face flux tiling diagnostic

This is an exploratory comparison of the final block-interior implementation
with a copied version whose regular-face flux pass visits 128×8 cell panels.
It applies the same panel loop to every AMR leaf. The column owner, face IDs,
live velocity and aperture reads, one flux evaluation per owned regular face,
residual-face path, and ordered cell reduction remain unchanged. The scalar
field is not updated until the flux pass finishes. The generator accepts only
the frozen final source hash and writes a diff plus source-tree hashes.

After the final cohort and its diagnostics release the numerical CPU, run
from the repository root with the Julia tooling project:

```sh
JULIA_DEPOT_PATH=/tmp/twophase-scaling-tools-depot:/home/ronald/.julia \
  julia --startup-file=no --project=benchmark/scaling \
  benchmark/scaling/stage3/block_interior/make_flux_tile_probe.jl \
  --source benchmark/scaling/results/20261010-block-interior-final-julia/current-source \
  --output benchmark/scaling/results/flux-tile-generator-smoke
```

Inspect the generated `flux-tile.patch` and `flux-tile-receipt.json`.
The runner makes fresh control and tile copies in a new ignored result
directory and runs the frozen-reference compact advection suite against
those explicit copies. The validation receipt records the loaded advection
file's hash and the actual regular-flux method's source path. Only after
both copies pass does it run three adjacent fresh-process control/tile
pairs at each of p1, p4, and p8:

```sh
JULIA_DEPOT_PATH=/tmp/twophase-scaling-tools-depot:/home/ronald/.julia \
  julia --startup-file=no --project=benchmark/scaling \
  benchmark/scaling/stage3/block_interior/flux_tile_study.jl \
  --cohort benchmark/scaling/results/20261010-block-interior-final-julia \
  --output benchmark/scaling/results/flux-tile-2048-oneblock \
  --repeats 3
```

The default case is 2048² in one block with 128 complete forward-Euler steps
per process. A separate run may add one `--case 1024:8` or
`--case 1024:1024` for a general-leaf regression; those use 1024 steps.
Every timed sample must last at least 0.2 seconds and match a freshly
generated clean-source oracle bitwise. The runner preserves commands,
reports, full-field hashes, source and environment hashes, pair order, and
raw times. Its median ratio is an exploratory diagnostic, not a performance
claim or a new accepted scaling cohort.

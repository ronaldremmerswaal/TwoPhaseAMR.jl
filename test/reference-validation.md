# MimeticFV cross-package validation

`integration/mimeticfv_reference.jl` compares one nontrivial, divergence-free, first-order
upwind step against the local `MimeticFV.jl` checkout. It is separate from the
normal test target so a correctness reference does not become a runtime or
test dependency.

From this repository, with a `MimeticFV.jl` checkout at `$MIMETICFV_DIR`, run:

```sh
JULIA_LOAD_PATH="$MIMETICFV_DIR:$PWD:@stdlib" \
    julia --startup-file=no test/integration/mimeticfv_reference.jl
```

The 2026-09-13 validation with Julia 1.12.7 passed at relative and absolute
tolerances of `3e-14`.

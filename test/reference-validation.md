# MimeticFV cross-package validation

`validation/mimeticfv_reference.jl` compares one nontrivial, divergence-free, first-order
upwind step against the local `MimeticFV.jl` checkout. It is separate from the
normal test target so a correctness reference does not become a runtime or
test dependency.

From this repository, with the sibling checkout at
`/home/ronald/Documents/MimeticFV.jl`, run:

```sh
JULIA_LOAD_PATH=/home/ronald/Documents/MimeticFV.jl:/home/ronald/Documents/TwoPhaseAMR.jl:@stdlib \
    julia --startup-file=no test/validation/mimeticfv_reference.jl
```

The 2026-09-13 validation with Julia 1.12.7 passed at relative and absolute
tolerances of `3e-14`.

#!/usr/bin/env bash
# Full study, single thread (≈ 4 h on one core). Run from this directory.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
J="julia --project=. -t 1"
$J -e 'using Pkg; Pkg.instantiate()'
$J verify_reference.jl                                   # results/verify_reference.txt
$J make_reference.jl                                     # results/reference_*.csv, reference_summary.txt
$J kh_run.jl r1_thin,r10_thin both 16,64,128,256 skip
$J kh_run.jl r1_thin,r10_thin both 32,512 field skip     # also fields for the 2×2 figure
$J kh_run.jl r1_thick,r10_thick both 16,32,64,128,256 skip
$J kh_run.jl r1_thin both 128 uniform skip               # graded vs uniform mesh
$J kh_run.jl r1_thin both 256 amp=1e-4 skip              # linearity of the measurement
$J jump_reproducer.jl | tee results/jump_reproducer.txt
(cd ctu_threshold && JC="julia --project=.. -t 1" && $JC ctu_study.jl reproducer && $JC ctu_study.jl kh MC && $JC ctu_report.jl)
$J analyze.jl                                            # results/tables.md, errors.csv
$J plots.jl r1_thin 32 512                               # figures/*.png
$J prelim.jl r1_thin 32 512                              # prelim/*
$J mc_compare.jl run                                       # MC vs LaxWendroff
$J mc_compare.jl report

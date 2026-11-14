#!/bin/bash
# Plain 1/2-1/2 aperture runs and the invariants study, sequential.
cd /home/ronald/Documents/mm-runs/mm_momentum
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1
J="julia --heap-size-hint=3G -t 1 --project=/home/ronald/Documents/wt-mm-momentum"
export MM_ACTIVITY=aperture
MM_ACTIVITY=volume $J impact.jl D0 > log_impact_D0.log 2>&1
for v in "D0" "DI" "DP match" "DG match"; do
  $J impact.jl $v > "log_ap_impact_${v// /_}.log" 2>&1
done
$J transitions.jl > log_transitions.log 2>&1
MM_APERTURE=plain $J transitions.jl > log_transitions_pl.log 2>&1
MM_APERTURE=plain $J smoke_ap.jl > smoke_ap_pl.txt 2>&1
MM_APERTURE=plain $J freestream.jl > log_ap_pl_freestream.log 2>&1
MM_APERTURE=plain $J smallcell.jl > log_ap_pl_smallcell.log 2>&1
for v in "D0" "DI" "DP match" "DG match"; do
  MM_APERTURE=plain $J impact.jl $v > "log_ap_pl_impact_${v// /_}.log" 2>&1
done
$J invariants.jl > log_ap_invariants.log 2>&1
MM_ACTIVITY=volume $J invariants.jl > log_invariants.log 2>&1
echo QUEUE3 DONE

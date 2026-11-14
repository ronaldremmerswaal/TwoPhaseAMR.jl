#!/bin/bash
# Aperture-activity study runs, sequential (one Julia process at a time).
cd /home/ronald/Documents/mm-runs/mm_momentum
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1
J="julia --heap-size-hint=3G -t 1 --project=/home/ronald/Documents/wt-mm-momentum"
export MM_ACTIVITY=aperture
$J geometry.jl > log_ap_geometry.log 2>&1
$J freestream.jl > log_ap_freestream.log 2>&1
$J smallcell.jl > log_ap_smallcell.log 2>&1
for v in "D0" "DI" "DP match" "DG match"; do
  $J drop.jl $v > "log_ap_drop_${v// /_}.log" 2>&1
done
MM_ACTIVITY=volume $J impact.jl D0 > log_impact_D0.log 2>&1
for v in "D0" "DI" "DP match" "DG match"; do
  $J impact.jl $v > "log_ap_impact_${v// /_}.log" 2>&1
done
MM_TIES=0 $J smallcell.jl > log_apnt_smallcell.log 2>&1
echo QUEUE2 DONE

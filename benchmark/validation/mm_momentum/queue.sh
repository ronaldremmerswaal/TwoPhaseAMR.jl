#!/bin/bash
# Sequential runs (one Julia process at a time).
cd /home/ronald/Documents/mm-runs/mm_momentum
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1
J="julia --heap-size-hint=3G -t 1 --project=/home/ronald/Documents/wt-mm-momentum"
$J smallcell.jl > log_smallcell.log 2>&1
for v in "D0" "D0 match" "DI" "DP match"; do
  $J drop.jl $v > "log_drop_${v// /_}.log" 2>&1
done
for g in rt waves khdt kh; do
  for v in "D0" "D0 match" "DI" "DP match"; do
    $J cases.jl $g $v > "log_${g}_${v// /_}.log" 2>&1
  done
done
for v in "D0" "DI" "DP match"; do
  $J cases.jl timing $v > "log_timing_${v// /_}.log" 2>&1
done
echo QUEUE DONE

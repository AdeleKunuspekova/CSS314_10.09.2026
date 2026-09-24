#!/usr/bin/env bash
# run_benchmarks.sh -- drives Phase 3 (thread scaling) and Experiment B (scheduling)
# Edit N below to YOUR computed workload size before running.
set -e

N=13231000   

echo "== Compiling =="
gcc -O2 -fopenmp collatz_seq.c -o collatz_seq
gcc -O2 -fopenmp collatz_parallel.c -o collatz_parallel
gcc -O2 -fopenmp collatz_falsesharing.c -o collatz_falsesharing

echo "== Phase 2: Sequential baseline (3 runs, discard run 1) =="
for run in 1 2 3; do
  echo "--- seq run $run ---"
  ./collatz_seq "$N"
done

echo "== Phase 3: Thread scaling (schedule=static) =="
for k in 1 2 4 8 16; do
  echo "--- k=$k ---"
  OMP_NUM_THREADS=$k OMP_SCHEDULE="static" ./collatz_parallel "$N"
done

echo "== Experiment A: False sharing (use your max PHYSICAL core count) =="
MAXCORES=4   # <-- set this to your actual physical core count
for mode in naive padded reduction; do
  echo "--- mode=$mode ---"
  OMP_NUM_THREADS=$MAXCORES ./collatz_falsesharing "$N" "$mode"
done

echo "== Experiment B: Scheduling sweep (use your max PHYSICAL core count) =="
for sched in "static" "static,1000" "dynamic,100" "dynamic,10000" "guided"; do
  echo "--- schedule=$sched ---"
  OMP_NUM_THREADS=$MAXCORES OMP_SCHEDULE="$sched" ./collatz_parallel "$N"
done

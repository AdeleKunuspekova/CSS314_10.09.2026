"""
Lab Practicum 01 - Task 2
Multi-Process Scaling Benchmark & Contention Wall

Workload: Compute-bound prime sieve, split across N worker processes.
Run this on Windows with: python task2_scaling_benchmark.py

Uses multiprocessing (not threading) because Python's GIL prevents
threads from running CPU-bound Python bytecode in true parallel.
multiprocessing spawns real OS processes, each with its own interpreter,
so this actually exercises your physical/logical cores.
"""

import time
import multiprocessing as mp

# ---- Workload: count primes in a numeric range (CPU-bound, no I/O) ----
def count_primes_in_range(args):
    start, end = args
    count = 0
    for n in range(max(start, 2), end):
        if n < 2:
            continue
        is_prime = True
        i = 2
        while i * i <= n:
            if n % i == 0:
                is_prime = False
                break
            i += 1
        if is_prime:
            count += 1
    return count


def run_sieve(upper_bound: int, num_procs: int) -> float:
    """Splits [0, upper_bound) into num_procs chunks and runs them in parallel.
    Returns wall-clock time in seconds."""
    chunk_size = upper_bound // num_procs
    ranges = [
        (i * chunk_size, upper_bound if i == num_procs - 1 else (i + 1) * chunk_size)
        for i in range(num_procs)
    ]

    start_time = time.perf_counter()
    if num_procs == 1:
        # Avoid pool overhead noise for the N=1 baseline
        total = sum(count_primes_in_range(r) for r in ranges)
    else:
        with mp.Pool(processes=num_procs) as pool:
            results = pool.map(count_primes_in_range, ranges)
        total = sum(results)
    elapsed = time.perf_counter() - start_time
    return elapsed, total


if __name__ == "__main__":
    UPPER_BOUND = 5_000_000   # matches the sieve-to-5,000,000 example in the handout
    THREAD_COUNTS = [1, 2, 4, 8, 16, 32]
    RUNS_PER_N = 3

    print(f"Workload: Prime Sieve to {UPPER_BOUND:,}")
    print(f"{'N':>4} | {'Run1(s)':>8} | {'Run2(s)':>8} | {'Run3(s)':>8} | {'Avg(s)':>8}")
    print("-" * 55)

    baseline_avg = None
    for n in THREAD_COUNTS:
        times = []
        for r in range(RUNS_PER_N):
            elapsed, total_primes = run_sieve(UPPER_BOUND, n)
            times.append(elapsed)
        avg = sum(times) / len(times)
        if n == 1:
            baseline_avg = avg
        speedup = baseline_avg / avg if baseline_avg else 1.0
        efficiency = speedup / n * 100
        print(f"{n:>4} | {times[0]:>8.3f} | {times[1]:>8.3f} | {times[2]:>8.3f} | "
              f"{avg:>8.3f}  (Speedup {speedup:.2f}x, Eff {efficiency:.1f}%)")

    print("\nPrimes found (sanity check, should match across all N):", total_primes)
    print("\nCopy the Avg Time column into your Task 2 table.")
    print("Note: if your CPU has fewer than 32 logical threads, N=16/32 will show")
    print("oversubscription — this IS your contention wall data for Q2.1.")
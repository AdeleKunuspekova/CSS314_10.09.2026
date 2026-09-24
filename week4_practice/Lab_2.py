import time
import csv
import threading
import numpy as np
from numba import njit, prange
import numba

# for naive race / critical section (pure Python threads)
N_STEPS_PYTHON_LOOP = 2_000_000
# for serial / reduction (matches manual's starter code)
N_STEPS_NUMBA = 100_000_000


# ---------- Variant A: Naive unsynchronized race ----------
def calc_pi_naive_race(num_steps: int, num_threads: int):
    step = 1.0 / num_steps
    shared_sum = [0.0]
    chunk = num_steps // num_threads

    def worker(start, end):
        for i in range(start, end):
            x = (i + 0.5) * step
            shared_sum[0] += 4.0 / (1.0 + x * x)  # unprotected shared write

    threads = []
    for t in range(num_threads):
        start = t * chunk
        end = num_steps if t == num_threads - 1 else start + chunk
        th = threading.Thread(target=worker, args=(start, end))
        threads.append(th)
        th.start()
    for th in threads:
        th.join()

    return shared_sum[0] * step


# ---------- Variant B: Critical section (lock) ----------
def calc_pi_critical_section(num_steps: int, num_threads: int):
    step = 1.0 / num_steps
    shared_sum = [0.0]
    lock = threading.Lock()
    chunk = num_steps // num_threads

    def worker(start, end):
        for i in range(start, end):
            x = (i + 0.5) * step
            term = 4.0 / (1.0 + x * x)
            with lock:  # emulates #pragma omp critical
                shared_sum[0] += term

    threads = []
    for t in range(num_threads):
        start = t * chunk
        end = num_steps if t == num_threads - 1 else start + chunk
        th = threading.Thread(target=worker, args=(start, end))
        threads.append(th)
        th.start()
    for th in threads:
        th.join()

    return shared_sum[0] * step


# ---------- Variant C: Serial baseline + Numba parallel reduction ----------
@njit
def calc_pi_serial(num_steps: int) -> float:
    step = 1.0 / num_steps
    total_sum = 0.0
    for i in range(num_steps):
        x = (i + 0.5) * step
        total_sum += 4.0 / (1.0 + x * x)
    return total_sum * step


@njit(parallel=True)
def calc_pi_reduction(num_steps: int) -> float:
    step = 1.0 / num_steps
    total_sum = 0.0
    for i in prange(num_steps):
        x = (i + 0.5) * step
        total_sum += 4.0 / (1.0 + x * x)
    return total_sum * step


def task_2_1_race_quantification(outfile="task2_1_race_quantification.csv"):
    print("Task 2.1: Race Condition Quantification (Variant A)")
    p_values = [1, 2, 4, 8]
    rows = []
    for p in p_values:
        pi_val = calc_pi_naive_race(N_STEPS_PYTHON_LOOP, p)
        err = abs(pi_val - np.pi)
        rows.append((p, pi_val, err))
        print(f"  P={p} | Pi = {pi_val:.10f} | Error = {err:.2e}")
    with open(outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_threads", "pi_value", "abs_error"])
        w.writerows(rows)
    print(f"Saved to {outfile}\n")
    return rows


def task_2_2_critical_overhead(outfile="task2_2_critical_overhead.csv"):
    """Critical section vs single-threaded baseline, N=1,000,000."""
    print("Task 2.2: Critical Section Overhead (Variant B)")
    n = 1_000_000
    t0 = time.perf_counter()
    # 1 thread == effectively serial, no contention
    pi_serial = calc_pi_naive_race(n, 1)
    t1 = time.perf_counter()
    time_serial = t1 - t0

    p_values = [1, 2, 4, 8]
    rows = [("baseline_1thread_no_lock", time_serial)]
    for p in p_values:
        t0 = time.perf_counter()
        calc_pi_critical_section(n, p)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        overhead_pct = (elapsed - time_serial) / time_serial * 100
        rows.append((f"critical_P{p}", elapsed))
        print(
            f"  P={p} | Time = {elapsed:.4f}s | Overhead vs baseline = {overhead_pct:.1f}%")
    with open(outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "time_seconds"])
        w.writerows(rows)
    print(f"Saved to {outfile}\n")
    return rows


def task_2_3_2_4_scaling(outfile="task2_3_4_scaling.csv"):
    """Strong scaling of the Numba parallel reduction; Speedup & Efficiency."""
    print("Task 2.3/2.4: Strong Scaling of Parallel Reduction (Variant C)")
    # warm-up JIT
    _ = calc_pi_serial(1000)
    _ = calc_pi_reduction(1000)

    t0 = time.perf_counter()
    pi_serial = calc_pi_serial(N_STEPS_NUMBA)
    t1 = time.perf_counter()
    t_serial = t1 - t0
    print(
        f"  Serial baseline (1 thread): {t_serial:.4f}s | Pi={pi_serial:.10f}")

    max_threads = numba.config.NUMBA_NUM_THREADS
    p_values = [p for p in [1, 2, 4, 8, 16] if p <= max_threads] or [1]
    if max_threads < 16:
        print(f"  NOTE: this machine exposes only {max_threads} thread(s) to Numba; "
              f"P values above that are skipped. Re-run on a multi-core machine for the full sweep.")

    rows = [(1, t_serial, t_serial / t_serial, (t_serial / t_serial) / 1)]
    for p in p_values:
        numba.set_num_threads(p)
        times = []
        for trial in range(5):
            t0 = time.perf_counter()
            pi_val = calc_pi_reduction(N_STEPS_NUMBA)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg_t = sum(times) / len(times)
        speedup = t_serial / avg_t
        efficiency = speedup / p
        rows.append((p, avg_t, speedup, efficiency))
        print(
            f"  P={p:2d} | avg T={avg_t:.4f}s | Speedup S(P)={speedup:.2f}x | Efficiency E(P)={efficiency:.2f}")

    with open(outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_threads", "avg_time_seconds",
                   "speedup", "efficiency"])
        w.writerows(rows)
    print(f"Saved to {outfile}\n")
    return rows


if __name__ == "__main__":
    print("=" * 60)
    task_2_1_race_quantification()
    print("=" * 60)
    task_2_2_critical_overhead()
    print("=" * 60)
    task_2_3_2_4_scaling()

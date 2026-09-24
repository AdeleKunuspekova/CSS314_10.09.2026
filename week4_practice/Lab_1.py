
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import csv
import math


def worker_task(thread_id: int, team_size: int):
    """Emulates one thread's body inside '#pragma omp parallel'."""
    native_tid = threading.get_native_id()
    role = "Master" if thread_id == 0 else "Worker"
    # small stagger so interleaving is visible, mirrors starter code
    time.sleep(0.001 * (thread_id % 3))
    return f"[{role}] Logical Rank: {thread_id} of {team_size} | Native OS TID: {native_tid}"


def run_team(num_threads: int, verbose: bool = True):
    if verbose:
        print(f"--- Forking a team of {num_threads} threads ---")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker_task, tid, num_threads)
                   for tid in range(num_threads)]
        lines = [f.result() for f in futures]  # implicit barrier
    if verbose:
        for line in lines:
            print(line)
        print("--- Joined thread team. Execution returned to serial master ---\n")
    return lines


def task_1_1_nondeterminism(runs: int = 10, team_size: int = 4, outfile: str = "task1_1_nondeterminism.txt"):
    """Run the team `runs` times, save stdout-equivalent output to a text file."""
    with open(outfile, "w") as f:
        for r in range(runs):
            f.write(f"=== Run {r + 1} ===\n")
            lines = run_team(team_size, verbose=False)
            for line in lines:
                f.write(line + "\n")
            f.write("\n")
    print(f"Task 1.1: saved {runs} runs to {outfile}")


def busy_work(iters: int = 2_000_000):
    """CPU saturation workload for Task 1.3: floating point sqrt loop."""
    total = 0.0
    for i in range(1, iters):
        total += math.sqrt(i)
    return total


def worker_task_with_load(thread_id: int, team_size: int, iters: int):
    busy_work(iters)
    return thread_id


def task_1_2_oversubscription(outfile: str = "task1_2_oversubscription.csv"):
    """Sweep P over {1,2,4,8,16,32,64}; measure wall-clock team instantiate+join time."""
    p_values = [1, 2, 4, 8, 16, 32, 64]
    rows = []
    print("Task 1.2: Oversubscription sweep (team create+join only, no extra load)")
    for p in p_values:
        t0 = time.perf_counter()
        run_team(p, verbose=False)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        rows.append((p, elapsed))
        print(f"  P={p:3d} | Team create+join time = {elapsed*1000:.3f} ms")

    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_threads", "time_seconds"])
        writer.writerows(rows)
    print(f"Task 1.2: saved sweep data to {outfile}\n")
    return rows


def task_1_3_saturation(outfile: str = "task1_3_saturation.csv", iters_per_thread: int = 2_000_000):

    p_values = [1, 2, 4, 8]
    rows = []
    print("Task 1.3: CPU saturation sweep (open htop / Task Manager while this runs)")
    for p in p_values:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=p) as executor:
            futures = [executor.submit(
                worker_task_with_load, tid, p, iters_per_thread) for tid in range(p)]
            for f in futures:
                f.result()
        t1 = time.perf_counter()
        elapsed = t1 - t0
        rows.append((p, elapsed))
        print(f"  P={p:3d} | Wall-clock time = {elapsed:.4f} s")

    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_threads", "time_seconds"])
        writer.writerows(rows)
    print(f"Task 1.3: saved saturation data to {outfile}\n")
    return rows


if __name__ == "__main__":
    print("=" * 60)
    print("Task 1.1: Non-determinism verification")
    print("=" * 60)
    task_1_1_nondeterminism()

    print("=" * 60)
    print("Task 1.2: Thread oversubscription sweep")
    print("=" * 60)
    task_1_2_oversubscription()

    print("=" * 60)
    print("Task 1.3: CPU core saturation analysis")
    print("=" * 60)
    task_1_3_saturation()

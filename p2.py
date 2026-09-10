"""
Lab Practicum 01 - Task 3
The Unsynchronized Shared Counter & Race Condition Trap

Uses threading.Thread (not multiprocessing) because this task needs
threads that share the SAME memory address for `counter`. Even though
Python's GIL serializes bytecode execution, the GIL can switch between
threads mid read-modify-write cycle, so `counter += 1` (which is NOT
atomic — it's LOAD, ADD, STORE at the bytecode level) still produces
lost updates. That's exactly the RMW hazard Q3.1 asks about.

Run this on Windows with: python task3_race_condition.py
"""

import threading
import time

NUM_THREADS = 10
INCREMENTS_PER_THREAD = 1_000_000
EXPECTED = NUM_THREADS * INCREMENTS_PER_THREAD


def unsynchronized_worker(counter_box):
    for _ in range(INCREMENTS_PER_THREAD):
        counter_box[0] += 1   # NOT atomic: read, add, write


def synchronized_worker(counter_box, lock):
    for _ in range(INCREMENTS_PER_THREAD):
        with lock:
            counter_box[0] += 1


def run_unsynchronized():
    counter_box = [0]
    threads = [threading.Thread(target=unsynchronized_worker, args=(counter_box,))
               for _ in range(NUM_THREADS)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start
    return counter_box[0], elapsed


def run_synchronized():
    counter_box = [0]
    lock = threading.Lock()
    threads = [threading.Thread(target=synchronized_worker, args=(counter_box, lock))
               for _ in range(NUM_THREADS)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start
    return counter_box[0], elapsed


if __name__ == "__main__":
    print(f"Expected total: {EXPECTED:,}\n")

    print("=== Unsynchronized runs (fill in Task 3 table) ===")
    print(f"{'Run':>4} | {'Measured Output':>16} | {'Error (Expected-Actual)':>24} | {'Time(ms)':>10}")
    unsync_times_ms = []
    for run_num in range(1, 11):
        result, elapsed = run_unsynchronized()
        error = EXPECTED - result
        unsync_times_ms.append(elapsed * 1000)
        print(f"{run_num:>4} | {result:>16,} | {error:>24,} | {elapsed*1000:>10.1f}")

    avg_unsync_ms = sum(unsync_times_ms) / len(unsync_times_ms)

    print("\n=== Synchronized (Lock-protected) run for Q3.2 ===")
    result, elapsed_sync = run_synchronized()
    sync_ms = elapsed_sync * 1000
    print(f"Result: {result:,} (should exactly equal {EXPECTED:,})")
    print(
        f"\nUnlocked avg = {avg_unsync_ms:.1f} ms  vs.  Locked = {sync_ms:.1f} ms")
    print("Plug these two numbers into Q3.2.")

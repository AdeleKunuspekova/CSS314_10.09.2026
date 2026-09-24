/*
 * collatz_falsesharing.c -- Experiment A: False Sharing Penalty
 *
 * Compile: gcc -O2 -fopenmp collatz_falsesharing.c -o collatz_falsesharing
 * Run:     OMP_NUM_THREADS=<max_physical_cores> ./collatz_falsesharing <N> <mode>
 *
 *   mode = naive     -> Variant 1: int hit_count[MAX_THREADS], hit_count[tid]++
 *                        (deliberately triggers false sharing)
 *   mode = padded    -> Variant 2a: each counter padded to its own cache line
 *   mode = reduction -> Variant 2b: OpenMP reduction(+:total_hits)
 *
 * Counts how many i in [1,N] have collatz_steps(i) > 100.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "collatz_common.h"

#define MAX_THREADS 256
#define THRESHOLD 100

// Variant 2a: pad each counter out to a full 64-byte cache line so no two
// threads' counters share a line -> no MESI invalidation traffic.
typedef struct {
    long count;
    char pad[64 - sizeof(long)];
} PaddedCounter;

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <N> <naive|padded|reduction>\n", argv[0]);
        return 1;
    }
    uint64_t N = strtoull(argv[1], NULL, 10);
    const char *mode = argv[2];

    long total_hits = 0;
    double t0, t1;

    if (strcmp(mode, "naive") == 0) {
        // VARIANT 1: adjacent array elements share cache lines -> false sharing
        static long hit_count[MAX_THREADS];
        memset(hit_count, 0, sizeof(hit_count));

        t0 = omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (uint64_t i = 1; i <= N; i++) {
            if (collatz_steps(i) > THRESHOLD) {
                hit_count[omp_get_thread_num()]++;
            }
        }
        t1 = omp_get_wtime();

        for (int t = 0; t < MAX_THREADS; t++) total_hits += hit_count[t];

    } else if (strcmp(mode, "padded") == 0) {
        // VARIANT 2a: each thread's counter isolated on its own cache line
        static PaddedCounter hit_count[MAX_THREADS];
        memset(hit_count, 0, sizeof(hit_count));

        t0 = omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (uint64_t i = 1; i <= N; i++) {
            if (collatz_steps(i) > THRESHOLD) {
                hit_count[omp_get_thread_num()].count++;
            }
        }
        t1 = omp_get_wtime();

        for (int t = 0; t < MAX_THREADS; t++) total_hits += hit_count[t].count;

    } else if (strcmp(mode, "reduction") == 0) {
        // VARIANT 2b: OpenMP reduction -- no shared memory location at all
        // during the loop; combined only at the end.
        t0 = omp_get_wtime();
        #pragma omp parallel for schedule(static) reduction(+:total_hits)
        for (uint64_t i = 1; i <= N; i++) {
            if (collatz_steps(i) > THRESHOLD) {
                total_hits++;
            }
        }
        t1 = omp_get_wtime();

    } else {
        fprintf(stderr, "Unknown mode '%s'\n", mode);
        return 1;
    }

    double elapsed = t1 - t0;
    printf("mode=%s\n", mode);
    printf("threads=%d\n", omp_get_max_threads());
    printf("total_hits=%ld\n", total_hits);
    printf("elapsed_seconds=%.6f\n", elapsed);
    printf("throughput_iter_per_sec=%.2f\n", (double)N / elapsed);

    return 0;
}

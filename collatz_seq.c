/*
 * collatz_seq.c -- Sequential baseline (Phase 2)
 *
 * Compile: gcc -O2 -fopenmp collatz_seq.c -o collatz_seq
 * Run:     ./collatz_seq <N>
 *
 * Computes, for i in [1, N]:
 *   - max stopping time (Collatz steps)
 *   - sum of all stopping times mod 1,000,000,007 (anti-cheat checksum)
 *
 * Uses omp_get_wtime() for timing even though this file has no parallel
 * region, so timing methodology is identical to the parallel version.
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "collatz_common.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        return 1;
    }
    uint64_t N = strtoull(argv[1], NULL, 10);

    double t0 = omp_get_wtime();

    uint32_t max_steps = 0;
    uint64_t checksum = 0;

    for (uint64_t i = 1; i <= N; i++) {
        uint32_t s = collatz_steps(i);
        if (s > max_steps) max_steps = s;
        checksum = (checksum + s) % MOD;
    }

    double t1 = omp_get_wtime();

    printf("N=%llu\n", (unsigned long long)N);
    printf("max_steps=%u\n", max_steps);
    printf("checksum=%llu\n", (unsigned long long)checksum);
    printf("elapsed_seconds=%.6f\n", t1 - t0);

    return 0;
}

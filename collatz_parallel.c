/*
 * collatz_parallel.c -- OpenMP parallel version (Phase 3 + Experiment B)
 *
 * Compile: gcc -O2 -fopenmp collatz_parallel.c -o collatz_parallel
 * Run:     OMP_NUM_THREADS=<k> OMP_SCHEDULE="static" ./collatz_parallel <N>
 *
 * Using schedule(runtime) lets you sweep every scheduling policy in
 * Experiment B WITHOUT recompiling -- just change the OMP_SCHEDULE
 * environment variable:
 *
 *   OMP_SCHEDULE="static"          -> schedule(static), default chunk
 *   OMP_SCHEDULE="static,1000"     -> schedule(static,1000)
 *   OMP_SCHEDULE="dynamic,100"     -> schedule(dynamic,100)
 *   OMP_SCHEDULE="dynamic,10000"   -> schedule(dynamic,10000)
 *   OMP_SCHEDULE="guided"          -> schedule(guided)
 *
 * For Phase 3 (Amdahl scaling table), just vary OMP_NUM_THREADS with a
 * single fixed schedule (static is the natural default for that table).
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

    #pragma omp parallel for schedule(runtime) reduction(max:max_steps) reduction(+:checksum)
    for (uint64_t i = 1; i <= N; i++) {
        uint32_t s = collatz_steps(i);
        if (s > max_steps) max_steps = s;
        checksum += s;
    }
    checksum %= MOD; // safe: total sum for this N range fits well within uint64_t, so
                      // mod-at-end == mod-per-step in collatz_seq.c

    double t1 = omp_get_wtime();

    printf("threads=%d\n", omp_get_max_threads());
    printf("N=%llu\n", (unsigned long long)N);
    printf("max_steps=%u\n", max_steps);
    printf("checksum=%llu\n", (unsigned long long)checksum);
    printf("elapsed_seconds=%.6f\n", t1 - t0);

    return 0;
}

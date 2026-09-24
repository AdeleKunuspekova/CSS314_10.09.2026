#ifndef COLLATZ_COMMON_H
#define COLLATZ_COMMON_H

#include <stdint.h>

// Collatz Stopping Time Kernel (as specified in the worksheet)
static inline uint32_t collatz_steps(uint64_t n) {
    uint32_t steps = 0;
    while (n > 1) {
        if ((n & 1) == 0) n >>= 1;
        else n = 3 * n + 1;
        steps++;
    }
    return steps;
}

#define MOD 1000000007ULL

#endif

#include <stdio.h>
#include <stdint.h>
#include <cpuid.h>

int main(void)
{
    unsigned int eax, ebx, ecx, edx;

    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx))
    {
        unsigned int clflush_units = (ebx >> 8) & 0xFF;
        unsigned int cache_line_size = clflush_units * 8;

        printf("CPU cache line size: %u bytes\n", cache_line_size);
    }
    else
    {
        printf("Unable to query CPUID.\n");
        return 1;
    }

    return 0;
}
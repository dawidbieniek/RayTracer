#ifndef DOUBLEUTILS_H
#define DOUBLEUTILS_H

#include <cstdlib>

__device__ __host__ inline double randomDouble() 
{
    return rand() / (RAND_MAX + 1.0);
}

__device__ __host__ inline double clamp(double x, double min, double max)
{
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

#endif // DOUBLEUTILS_H
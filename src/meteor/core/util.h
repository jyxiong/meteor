#pragma once

#include <cuda_runtime.h>

namespace meteor {

template <typename IntegerType>
inline __host__ __device__ IntegerType roundUp(IntegerType x, IntegerType y)
{
    return ( ( x + y - 1 ) / y ) * y;
}

} // namespace meteor
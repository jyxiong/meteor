#pragma once

#include <cuda_runtime.h>

#include "meteor/core/macro.h"

namespace meteor {

template <typename T>
inline void cuda_free(T& data) {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(data)));
}
/** @brief Recursive release of object from a device. */
template <typename Head, typename... Args>
inline void cuda_frees(Head& head, Args... args) {
    cuda_free(head);
    if constexpr (sizeof...(args) > 0) 
        cuda_frees(args...);
}

} // namespace meteor
#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>

#include <optix.h>

#define THROW(msg)                                                             \
  do {                                                                         \
    std::stringstream ss;                                                      \
    ss << "'(" __FILE__ << ":" << __LINE__ << ")"                              \
       << ", " << msg;                                                         \
    throw std::runtime_error(ss.str());                                        \
  } while (0)

#define ASSERT(cond, msg)                                                      \
  do {                                                                         \
    if (!(bool)(cond)) {                                                       \
      std::stringstream ss;                                                    \
      ss << "Assertion failed at "                                             \
         << "(" __FILE__ << ":" << __LINE__ << ")"                             \
         << ", " << msg;                                                       \
      throw std::runtime_error(ss.str());                                      \
    }                                                                          \
  } while (0)

// CUDA error handles
// --------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::stringstream ss;                                                    \
      ss << "CUDA call (" << #call << " ) failed with error: '"                \
         << cudaGetErrorString(error) << "' (" __FILE__ << ":" << __LINE__     \
         << ")\n";                                                             \
      std::cout << ss.str() << std::endl;                                      \
      throw std::runtime_error(ss.str());                                      \
    }                                                                          \
  } while (0)

#define CUDA_SYNC_CHECK()                                                      \
  do {                                                                         \
    cudaDeviceSynchronize();                                                   \
    cudaError_t error = cudaGetLastError();                                    \
    if (error != cudaSuccess) {                                                \
      std::stringstream ss;                                                    \
      ss << "CUDA error on synchronize with error '"                           \
         << cudaGetErrorString(error) << "' (" __FILE__ << ":" << __LINE__     \
         << ")\n";                                                             \
      std::cout << ss.str() << std::endl;                                      \
      throw std::runtime_error(ss.str());                                      \
    }                                                                          \
  } while (0)

// OptiX error handles
// -------------------------------------------------------------
#define OPTIX_CHECK(call)                                                      \
  do {                                                                         \
    OptixResult res = call;                                                    \
    if (res != OPTIX_SUCCESS) {                                                \
      std::stringstream ss;                                                    \
      ss << "ERROR: " << res << ", ";                                          \
      ss << "Optix call '" << #call << "' failed: " __FILE__ ":" << __LINE__   \
         << ")\n";                                                             \
      std::cout << ss.str() << std::endl;                                      \
      throw std::runtime_error(ss.str());                                      \
    }                                                                          \
  } while (0)

#define OPTIX_CHECK_LOG(call)                                                  \
  do {                                                                         \
    OptixResult res = call;                                                    \
    const size_t sizeof_log_returned = sizeof_log;                             \
    sizeof_log = sizeof(log); /* reset sizeof_log for future calls */          \
    if (res != OPTIX_SUCCESS) {                                                \
      std::stringstream ss;                                                    \
      ss << "Optix call '" << #call << "' failed: " __FILE__ ":" << __LINE__   \
         << ")\nLog:\n"                                                        \
         << log << (sizeof_log_returned > sizeof(log) ? "<TRUNCATED>" : "")    \
         << "\n";                                                              \
      throw std::runtime_error(ss.str());                                      \
    }                                                                          \
  } while (0)

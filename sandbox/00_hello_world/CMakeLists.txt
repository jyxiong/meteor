include(embed_cuda)

embed_cuda(
  GEN_DEBUG
  CONST HEADER otkHelloKernelCuda.h
  OUTPUT_TARGET
    otkHelloKernel
  LIBRARIES
    optix
  SOURCES
    hello.cu
)

add_executable(00_hello_world main.cpp main.cu hello.h)
target_link_libraries(00_hello_world PRIVATE meteor otkHelloKernel)
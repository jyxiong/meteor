file(GLOB_RECURSE OPTIX_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/optix/*.h)

find_package(CUDAToolkit REQUIRED)

set(OptiX_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/optix/include)

add_library(optix INTERFACE ${OPTIX_SOURCES})
target_include_directories(optix INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/optix/include)
# target_link_libraries(OptiX::OptiX INTERFACE ${CMAKE_DL_LIBS})

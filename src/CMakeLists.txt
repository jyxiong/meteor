file(GLOB CORE_SOURCES core/*.cpp core/*.h)
file(GLOB OPTIX_SOURCES meteor/optix/*.cpp meteor/optix/*.h)

add_library(meteor STATIC 
    ${CORE_SOURCES}
    ${OPTIX_SOURCES}
    )

target_include_directories(meteor SYSTEM PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(meteor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(meteor PUBLIC glfw glm imgui spdlog stb optix) 

target_compile_definitions(meteor PUBLIC 
STB_IMAGE_WRITE_IMPLEMENTATION)

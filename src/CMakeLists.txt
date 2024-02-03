file(GLOB SOURCES project/*.cpp)
file(GLOB HEADERS project/*.h)

add_library(project STATIC ${SOURCES} ${HEADERS})

target_include_directories(project PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(project PUBLIC glfw glm imgui spdlog stb)

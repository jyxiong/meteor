set(IMGUI_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/imgui)

file(GLOB IMGUI_SOURCES CONFIGURE_DEPENDS ${IMGUI_SOURCE_DIR}/*.cpp)
file(GLOB IMGUI_IMPL CONFIGURE_DEPENDS
    ${IMGUI_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_SOURCE_DIR}/backends/imgui_impl_glfw.h
    ${IMGUI_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    ${IMGUI_SOURCE_DIR}/backends/imgui_impl_opengl3.h)

add_library(imgui STATIC ${IMGUI_SOURCES} ${IMGUI_IMPL})
target_include_directories(imgui PUBLIC $<BUILD_INTERFACE:${IMGUI_SOURCE_DIR}>)
target_link_libraries(imgui PUBLIC glfw)
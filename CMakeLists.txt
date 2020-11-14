cmake_minimum_required(VERSION 3.0.0)
project(alpha_zero VERSION 0.1.0)

list(APPEND CMAKE_PREFIX_PATH "~/libtorch")
set(CUDNN_INCLUDE_DIR "/usr/lib/cuda")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare -std=c++2a ${TORCH_CXX_FLAGS}")

file(GLOB TTT_SRC "tictactoe/*.cpp")
file(GLOB NEURAL_SRC "neural/*.cpp")

add_executable("${PROJECT_NAME}" main.cpp ${NEURAL_SRC} ${TTT_SRC})
target_link_libraries("${PROJECT_NAME}" "${TORCH_LIBRARIES}")
set_property(TARGET "${PROJECT_NAME}" PROPERTY CXX_STANDARD 20)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET "${PROJECT_NAME}"
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:"${PROJECT_NAME}">)
endif (MSVC)

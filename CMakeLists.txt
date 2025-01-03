cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CudaStudy LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 11)

find_package(CUDA REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(CudaApp
    src/main.cpp
    src/cuda_kernel.cu
)

set_target_properties(CudaApp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(CudaApp ${CUDA_LIBRARIES})
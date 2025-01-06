cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CudaStudy LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 11)

set(SDL2_USE_STATIC_LIBS ON)

find_package(CUDA REQUIRED)
find_package(SDL2 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${SDL2_INCLUDE_DIRS})

add_executable(CudaApp
    src/Complex.h
    src/Complex.cpp
    src/main.cpp
    src/cuda_kernel.cu
)

set_target_properties(CudaApp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(CudaApp 
${CUDA_LIBRARIES}
SDL2::SDL2-static 
SDL2::SDL2main
)
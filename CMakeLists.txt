cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CudaStudy LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 11)

set(SDL2_USE_STATIC_LIBS ON)

find_package(CUDA REQUIRED)
find_package(SDL2 REQUIRED)

file(GLOB SOURCES "src/*.cpp")
file(GLOB CUFILES "src/*.cu" "src/*.cuh")
file(GLOB CUHEADERS "src/*.cuh")
file(GLOB HEADERS "src/*.h")

source_group("CUDA\\Kernels" FILES ${CUFILES})
source_group("CUDA\\Headers" FILES ${CUHEADERS})

file(GLOB RIPPLE_SOURCES "ripple/*.cpp")
file(GLOB RIPPLE_HEADERS "ripple/*.h")
file(GLOB RIPPLE_CUFILES "ripple/*.cu"  "ripple/*.cuh")

file(GLOB JULIA_SOURCES "julia/*.cpp")
file(GLOB JULIA_HEADERS "julia/*.h")
file(GLOB JULIA_CUFILES "julia/*.cu"  "julia/*.cuh")


#file(GLOB NBODY_SOURCES "nbody/*.cpp")
#file(GLOB NBODY_HEADERS "nbody/*.h")
#file(GLOB NBODY_CUFILES "nbody/*.cu"  "nbody/*.cuh")


#file(GLOB BLUR_SOURCES "blur/*.cpp")
#file(GLOB BLUR_HEADERS "blur/*.h")
#file(GLOB BLUR_CUFILES "blur/*.cu"  "blur/*.cuh")


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${SDL2_INCLUDE_DIRS})

add_executable(CUDARayTracing 
	${SOURCES}
	${HEADERS}
	${CUFILES}
)

add_executable(CUDARipple
	${RIPPLE_SOURCES}
	${RIPPLE_HEADERS}
	${RIPPLE_CUFILES}
)

add_executable(CUDAJulia 
	${JULIA_SOURCES} 	${JULIA_HEADERS}
	${JULIA_CUFILES}
)

set_target_properties(CUDARayTracing PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(CUDARipple PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(CUDAJulia PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(CUDARayTracing 
${CUDA_LIBRARIES}
SDL2::SDL2-static 
SDL2::SDL2main
)

target_link_libraries(CUDAJulia 
${CUDA_LIBRARIES}
SDL2::SDL2-static 
SDL2::SDL2main
)


target_link_libraries(CUDARipple 
${CUDA_LIBRARIES}
SDL2::SDL2-static 
SDL2::SDL2main
)


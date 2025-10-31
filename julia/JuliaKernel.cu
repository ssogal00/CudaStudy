

#include <cuda_runtime.h>
#include <iostream>
#include "JuliaKernel.cuh"
__global__ void renderJuliaSetCudaKernel(uint32_t* pixels, int width, int height, float c_real, float c_imag, int max_iterations);

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            /* 문제가 발생했을 때 d_pixels가 유효하면 해제 시도 후 종료 */ \
            if (err != cudaErrorInvalidDevicePointer && err != cudaErrorInvalidValue) \
            { \
                 /* cudaMalloc의 성공 여부에 따라 d_pixels가 유효하지 않을 수 있으므로 주의 */ \
            } \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void renderJuliaSetCuda(uint32_t* pixels, int width, int height, float C_Real, float C_Imag, int max_iterations)
{
    uint32_t* d_pixels;

    CUDA_CHECK(cudaMalloc((void**)&d_pixels, width * height * sizeof(uint32_t)));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    renderJuliaSetCudaKernel << <blocksPerGrid, threadsPerBlock >> > (d_pixels, width, height, C_Real, C_Imag, max_iterations);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaMemcpy(pixels, d_pixels, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cudaFree(d_pixels);
}


__global__ void renderJuliaSetCudaKernel(uint32_t* pixels, int width, int height, float c_real, float c_imag, int max_iterations)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    int index = y * width + x;

    float z_real = (x - width / 2.0) * 4.0 / width;
    float z_imag = (y - height / 2.0) * 4.0 / height;

    int iterations = 0;

    while (z_real * z_real + z_imag * z_imag < 4 && iterations < max_iterations)
    {
        float temp = z_real * z_real - z_imag * z_imag + c_real;
        z_imag = 2 * z_real * z_imag + c_imag;
        z_real = temp;
        iterations++;
    }

    if (iterations == max_iterations)
    {
        pixels[index] = 0;
    }
    else
    {
        pixels[index] = iterations; // *255 / max_iterations;
    }
}
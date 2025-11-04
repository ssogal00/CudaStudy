
#include "Synchronize.cuh"



__global__ void synchronizeKernel(uint32_t* d_pixels)
{
    __shared__ float sharedMem[16][16];

    const float period = 128.0f;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    sharedMem[threadIdx.x][threadIdx.y] =
        255 * (sinf(x * 2.f * M_PI / period) + 1.f) * (sinf(y * 2.f * M_PI / period) + 1.f) / 4.f;

	__syncthreads(); 

    d_pixels[offset] = (uint32_t)sharedMem[15 - threadIdx.x][15 - threadIdx.y]; // R
}

void render(uint32_t* pixels, int width, int height)
{
    uint32_t* d_pixels;

    cudaMalloc((void**)&d_pixels, width * height * sizeof(uint32_t));

    dim3 threadsPerBlock(16, 16);

    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    synchronizeKernel << <blocksPerGrid, threadsPerBlock >> > (d_pixels);

    cudaDeviceSynchronize();
    cudaMemcpy(pixels, d_pixels, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}

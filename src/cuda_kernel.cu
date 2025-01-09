#include "cuda_kernel.cuh"

__global__ void add_arrays_kernel(const int *a, const int *b, int *c, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

void add_arrays(const int *a, const int *b, int *c, int size) {
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size * sizeof(int));
    cudaMalloc((void**)&d_b, size * sizeof(int));
    cudaMalloc((void**)&d_c, size * sizeof(int));

    cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    add_arrays_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);

    cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
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
        pixels[index] = iterations * 255 / max_iterations;
    }
}


void renderJuliaSetCuda(uint32_t* pixels, int width, int height, float C_Real, float C_Imag, int max_iterations)
{
    uint32_t* d_pixels;

    cudaMalloc((void**)&d_pixels, width * height * sizeof(uint32_t));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);


    renderJuliaSetCudaKernel <<<blocksPerGrid, threadsPerBlock >>> (d_pixels, width, height, C_Real, C_Imag, max_iterations);

    cudaDeviceSynchronize();

    cudaMemcpy(pixels, d_pixels, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_pixels);
}
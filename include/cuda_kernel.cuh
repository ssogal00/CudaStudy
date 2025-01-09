#ifndef CUDA_KERNEL_CUH
#define CUDA_KERNEL_CUH

void add_arrays(const int *a, const int *b, int *c, int size);

void renderJuliaSetCuda(uint32_t* pixels, int width, int height, float C_Real, float C_Imag, int max_iterations);

#endif // CUDA_KERNEL_CUH

#pragma once

#include <cuda_runtime.h>

#define M_PI (3.14159265358979323846)
__global__ void synchronizeKernel(uint32_t* d_pixels);
void render(uint32_t* pixels, int width, int height);

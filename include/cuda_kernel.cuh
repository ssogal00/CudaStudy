#ifndef CUDA_KERNEL_CUH
#define CUDA_KERNEL_CUH

#include <vector_types.h>
#include "Vector.h"

struct CuRay;

__device__ float3 Unit(const float3& InValue);
__device__ float3 GetColor(const CuRay& ray);
__device__ float Dot(const float3& lhs, const float3& rhs);
__device__ float3 operator+(const float3& lhs, const float3& rhs);
__device__ float3 operator-(const float3& lhs, const float3& rhs);
__device__ float3 operator*(const float3& lhs, const float f);
__device__ float3 operator*(const float f, const float3& rhs);

struct CuRay 
{
public:
	float3 mOrigin;
	float3 mDir; 

	__device__ CuRay(float3 origin, float3 dir) : mOrigin{ origin }, mDir{ Unit(dir) } {}

	__device__ CuRay() : mOrigin{ 0.0f, 0.0f, 0.0f }, mDir{ 0.0f, 0.0f, 1.0f } {}

	__device__ float3 At(float t) const;
};

struct CuSphere
{
public:
	float3 mOrigin;
	float mRadius;
	__device__ CuSphere() : mOrigin{ 0.0f, 0.0f, 0.0f }, mRadius{ 1.0f } {}
	__device__ CuSphere(float3 origin, float radius) : mOrigin{ origin }, mRadius{ radius } {}
	__device__ bool Intersect(const CuRay& ray, float& t) const;	
};

struct CuCamera
{
public:
	float3 mOrigin;
	float3 mUpperLeft;
	float3 mHorizontal;
	float3 mVertical;

	__device__ CuCamera(float3 origin, float3 upperLeft, float3 horizontal, float3 vertical)
		: mOrigin{ origin }, mUpperLeft{ upperLeft }, mHorizontal{ horizontal }, mVertical{ vertical } {}

	__device__ CuRay GetRay(float u, float v) const;
};

void add_arrays(const int *a, const int *b, int *c, int size);

void renderJuliaSetCuda(uint32_t* pixels, int width, int height, float C_Real, float C_Imag, int max_iterations);

void renderRGBCuda(float* redValues, float* greenValues, int width, int height);

void renderSphereCuda(float* redValues, float* greenValues, float* blueValues, int width, int height);
#endif // CUDA_KERNEL_CUH
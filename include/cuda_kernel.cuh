#ifndef CUDA_KERNEL_CUH
#define CUDA_KERNEL_CUH

#include <vector_types.h>
#include "Vector.h"
#include <cmath>
#include <math.h>
#include <curand_kernel.h>

struct CuRay;

#define M_PI (3.14159265358979323846)

__device__ inline float3 Unit(const float3& InValue);
__device__ float3 GetColor(const CuRay& ray);
__device__ float Dot(const float3& lhs, const float3& rhs);
__device__ inline float3 Cross(const float3& lhs, const float3& rhs);
__device__ inline float3 operator+(const float3& lhs, const float3& rhs);
__device__ inline float3 operator-(const float3& lhs, const float3& rhs);
__device__ inline float3 operator*(const float3& lhs, const float f);
__device__ inline float3 operator*(const float f, const float3& rhs);
__device__ inline float3 operator/(const float3& lhs, const float f);

__device__ inline float3 Reflect(const float3& v, const float3& n);

__device__ inline double RandomDouble(curandState* state)
{
	return static_cast<double>(curand_uniform(state));
}

__device__ inline float RandomFloat(curandState* state)
{
	return static_cast<float>(curand_uniform(state));
}

__device__ inline double RandomDoubleInRange(double min, double max, curandState* state)
{
	return min + (max - min) * RandomDouble(state);
}

__device__ inline float3 RandomUnitVector(curandState* state) 
{
	float z = curand_uniform(state) * 2.0f - 1.0f;  // [-1,1]
	float theta = curand_uniform(state) * 2.0f * M_PI; // [0, 2¥ð)
	float r = sqrtf(1.0f - z * z);

	float x = r * cosf(theta);
	float y = r * sinf(theta);

	return make_float3(x, y, z);
}

__global__ void SetupRandomState(curandState* state, unsigned long long seed, int idx);



struct CuHitRecord
{
public:
	float3 mPoint;  // Point of intersection
	float3 mNormal; // Normal at the intersection point
	float mT;       // Distance along the ray to the intersection point

	__device__ CuHitRecord() 
		: mPoint{ 0.0f, 0.0f, 0.0f }, mNormal{ 0.0f, 0.0f, 1.0f }, mT{ 0.0f }
	{
	}
	__device__ CuHitRecord(float3 point, float3 normal, float t) 
		: mPoint{ point }, mNormal{ normal }, mT{ t }
	{
	}
};

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
	__device__ bool Hit(const CuRay& ray, CuHitRecord& hitRecord, float tMin, float tMax) const;
	__device__ bool Intersect(const CuRay& ray, float& t) const;	
};

struct CuCamera
{
public:
	float3 mOrigin;
	float3 mUpperLeft;
	float3 mHorizontal;
	float3 mVertical;
	float3 mLookAt;
	float3 mUp;
	
	__device__ void Initialize();

	__device__ CuCamera(float3 vOrigin, float3 vLookAt, float3 vUp, float fov, float aspectRatio)
		: mOrigin{ vOrigin }, 
		mLookAt{ vLookAt }, 
		mUp{ vUp }, 
		mFOV{ fov }, 
		mAspectRatio{ aspectRatio }
	{
		Initialize();
	}

	__device__ CuRay GetRay(float u, float v) const;

private:
	float3 mPixel00;
	float3 mPixelDeltaU;
	float3 mPixelDeltaV;
	float mFOV = 60;
	float mAspectRatio = 2.0f;
	float mFocalLength = 100;
	float mImageWidth = 1024;
	float mImageHeight = 512;
};

void add_arrays(const int *a, const int *b, int *c, int size);

void renderJuliaSetCuda(uint32_t* pixels, int width, int height, float C_Real, float C_Imag, int max_iterations);

void renderRGBCuda(float* redValues, float* greenValues, int width, int height);

void renderSphereCuda(float* redValues, float* greenValues, float* blueValues, int width, int height, float fCameraDistance, float fCameraHeight);
#endif // CUDA_KERNEL_CUH
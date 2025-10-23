#ifndef CUDA_KERNEL_CUH
#define CUDA_KERNEL_CUH

#include <vector_types.h>
#include "Vector.h"
#include <cmath>
#include <math.h>
#include <curand_kernel.h>

struct CuRay;

#define M_PI (3.14159265358979323846)
#define MAX_BOUNCES 10

__device__ inline float3 Unit(const float3& InValue);
__device__ float3 GetColor(const CuRay& ray);
__device__ float Dot(const float3& lhs, const float3& rhs);
__device__ inline float3 Cross(const float3& lhs, const float3& rhs);
__device__ inline float3 operator+(const float3& lhs, const float3& rhs);
__device__ inline float3 operator-(const float3& lhs, const float3& rhs);
__device__ inline float3 operator*(const float3& lhs, const float3& rhs);
__device__ inline float3 operator-(const float3& rhs);
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
	float theta = curand_uniform(state) * 2.0f * M_PI; // [0, 2π)
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
__device__ bool LambertScatter(const CuRay& rayIn, const CuHitRecord& hitRecord, float3& attenuation, CuRay& scattered, curandState* state);


__device__ bool MetalScatter(const CuRay& rayIn, const CuHitRecord& hitRecord, float3& attenuation, CuRay& scattered, curandState* state);

__device__ inline float3 RandomUnitVectorInHemisphere(const float3& normal, curandState* state)
{
	float3 inUnitSphere = RandomUnitVector(state);
	if (Dot(inUnitSphere, normal) < 0.0f)
	{
		inUnitSphere = -inUnitSphere; // Ensure the vector is in the hemisphere defined by the normal
	}
	return inUnitSphere;
}

__device__ __forceinline__ float3 GetSkyColor(const CuRay& ray)
{
	float t = 0.5f * (ray.mDir.y + 1.0f);
	return (1.0f - t) * make_float3(1.0f, 1.0f, 1.0f) + t * make_float3(0.5f, 0.7f, 1.0f);
}


__device__ __forceinline__ float3 RayColor(const CuRay& ray, CuSphere& sphere, CuSphere& sphereGreen, curandState* state)
{
	CuRay currentRay = ray;

	// 1. 'throughput'이 광선이 누적하는 색상입니다.
	float3 throughput = make_float3(1.0f, 1.0f, 1.0f);

	// (기존 attenuation 변수는 루프 안으로 이동합니다)

	for (int depth = 0; depth < MAX_BOUNCES; ++depth)
	{
		CuHitRecord hitRecord;
		CuHitRecord tempHitRecord;
		bool hitAnything = false;
		float closestSoFar = 10000.0f;

		// 2. 'materialAlbedo' (재질 색상) 변수를 선언합니다.
		//    이 변수는 '최종적으로' 부딪힌 물체의 색상만 저장합니다.
		float3 materialAlbedo = make_float3(1.0f, 1.0f, 1.0f); // 기본값(흰색)

		// 첫 번째 구체 검사
		if (sphere.Hit(currentRay, tempHitRecord, 0.001f, closestSoFar))
		{
			hitAnything = true;
			closestSoFar = tempHitRecord.mT;
			hitRecord = tempHitRecord;
			materialAlbedo = make_float3(0.9, 0.9, 1); // 빨간색 재질

			// ★ 3. 버그 수정: 여기서 throughput을 곱하지 않습니다!
		}

		// 두 번째 구체 검사
		if (sphereGreen.Hit(currentRay, tempHitRecord, 0.001f, closestSoFar))
		{
			hitAnything = true;
			closestSoFar = tempHitRecord.mT;
			hitRecord = tempHitRecord;
			materialAlbedo = make_float3(0.1f, 0.8f, 0.1f); // 초록색 재질

			// ★ 3. 버그 수정: 여기서 throughput을 곱하지 않습니다!
		}

		if (hitAnything)
		{
			CuRay scattered;
			// 'attenuation'은 이제 LambertScatter의 *출력 전용* 변수입니다.
			// (LambertScatter가 이 값을 {1,1,1}로 설정해 줄 것입니다)
			float3 attenuation;

			if (LambertScatter(currentRay, hitRecord, attenuation, scattered, state))
			//if (MetalScatter(currentRay, hitRecord, attenuation, scattered, state))
			{
				// ★ 4. 올바른 로직:
				//    throughput에 최종적으로 선택된 'materialAlbedo'와
				//    Lambert가 반환한 'attenuation'을 곱합니다.
				throughput = throughput * materialAlbedo * attenuation;
				currentRay = scattered;
			}
			else
			{
				return make_float3(0, 0, 0);
			}
		}
		else
		{
			// 5. 하늘에 부딪힘
			return throughput * GetSkyColor(currentRay);
		}
	}

	// 최대 바운스 도달
	return make_float3(0.0f, 0.0f, 0.0f);
}

void add_arrays(const int *a, const int *b, int *c, int size);

void renderJuliaSetCuda(uint32_t* pixels, int width, int height, float C_Real, float C_Imag, int max_iterations);

void renderRGBCuda(float* redValues, float* greenValues, int width, int height);

void renderSphereCuda(
	float* redValues, float* greenValues, float* blueValues, 
	int width, int height, float fCameraDistance, float fCameraHeight, unsigned long long frameCount);

extern float* d_redValues;
extern float* d_greenValues;
extern float* d_blueValues;
extern float* d_accum_red;
extern float* d_accum_green;
extern float* d_accum_blue;

extern curandState* d_randomState;

extern unsigned long long FrameCount;
#endif // CUDA_KERNEL_CUH
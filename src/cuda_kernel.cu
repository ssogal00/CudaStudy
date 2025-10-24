#include "cuda_kernel.cuh"
#include <vector_functions.h> // 필요시 명시적 포함

float* d_redValues;
float* d_greenValues;
float* d_blueValues;
float* d_accum_red;
float* d_accum_green;
float* d_accum_blue;

curandState* d_randomState;

unsigned long long FrameCount = 0;

__device__ float3 CuRay::At(float t) const
{
    float3 result;
	result.x = mOrigin.x + t * mDir.x;
	result.y = mOrigin.y + t * mDir.y;
	result.z = mOrigin.z + t * mDir.z;
    return result;
}

__device__ bool CuSphere::Intersect(const CuRay& ray, float& t) const
{
    float3 oc = ray.mOrigin - mOrigin;
    float a = Dot(ray.mDir, ray.mDir);
    float b = 2.0f * Dot(oc, ray.mDir);
    float c = Dot(oc, oc) - mRadius * mRadius;
    float discriminant = (b * b) - (4 * a * c);

    if (discriminant < 0)
    {
        return false;
    }

    t = (-b - sqrtf(discriminant)) / (2.0f * a);

    return true;
}
__device__ bool CuSphere::Hit(const CuRay& ray, CuHitRecord& hitRecord, float tMin, float tMax) const
{
    float3 oc = mOrigin - ray.mOrigin;
    float a = Dot(ray.mDir, ray.mDir);
	float h = Dot(oc, ray.mDir);
    float c = Dot(oc, oc) - mRadius * mRadius;
	float discriminant = h * h - a * c;

    if (discriminant < 0)
    {
        return false;
    }   
	float sqrtd = sqrtf(discriminant);

	// Find the nearest root that lies in the acceptable range
	float root = (h - sqrtd) / a;
	if (root < tMin || root > tMax)
	{
		root = (h + sqrtd) / a;
		if (root < tMin || root > tMax)
		{
			return false;
		}
	}

	hitRecord.mPoint = ray.At(root);
	hitRecord.mNormal = Unit(hitRecord.mPoint - mOrigin);
	hitRecord.mT = root;
	hitRecord.mAlbedo = mAlbedo;
	return true;
}

__device__ CuRay CuCamera::GetRay(float u, float v) const
{
	//float3 dir = Unit(mUpperLeft + u * mHorizontal - v * mVertical);
	//return CuRay(mOrigin, dir);

	float3 pixelSample = mPixel00 + mPixelDeltaU * u + mPixelDeltaV * v;
	return CuRay(mOrigin, Unit(pixelSample - mOrigin));
}

__device__ void CuCamera::Initialize()
{
    double theta = mFOV * M_PI / 180.0f; // Convert FOV to radians
    double h = tanf(theta / 2);
    double viewportHeight = 2 * h * mFocalLength;
    double viewportWidth = viewportHeight * mAspectRatio;

    float3 vLookDir = Unit(mLookAt - mOrigin);
    float3 vRight = Unit(Cross(vLookDir, mUp));
    float3 vUp = Unit(Cross(vRight, vLookDir));

	float3 viewportU = viewportWidth * vRight;
	float3 viewportV = viewportHeight * (vUp);

    mPixelDeltaU = viewportU / mImageWidth;
    mPixelDeltaV = viewportV / mImageHeight;

    float3 viewportUpperLeft = mOrigin + vLookDir * mFocalLength - (viewportU * 0.5f) - (viewportV * 0.5f);
    mPixel00 = viewportUpperLeft + 0.5 * (mPixelDeltaU + mPixelDeltaV); // Center the pixel at the upper left corner
}
__device__ float3 Unit(const float3& InValue)
{
	float length = sqrtf(InValue.x * InValue.x + InValue.y * InValue.y + InValue.z * InValue.z);
	return float3{ InValue.x / length, InValue.y / length, InValue.z / length };
}

__device__ float3 GetRayHitColor(const CuRay& ray)
{
    
}

__device__ float3 GetColor(const CuRay& ray)
{
	float3 unitVec = Unit(ray.mDir);
	//float t = 0.5f * (unitVec.y + 1.0f);
    float t = unitVec.y;
	return (t) * float3 {0, 1.0f, 0.f} + (1-t) * float3{ 0, 0, 1.0f }; // Gradient from white to blue
}

__device__ float3 operator+(const float3& lhs, const float3& rhs)
{
	return float3{ lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z };
}
__device__ float3 operator-(const float3& lhs, const float3& rhs)
{
    return float3{ lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z };
}

__device__ float3 operator-(const float3& v)
{
	return float3{ -v.x, -v.y, -v.z };
}

__device__ float3 operator*(const float3& lhs, const float f)
{
	float3 result;
	result.x = lhs.x * f;
    result.y = lhs.y * f;
    result.z = lhs.z * f;
    return result;
}

__device__ float3 operator*(const float f, const float3& rhs)
{
    float3 result;
    result.x = rhs.x * f;
    result.y = rhs.y * f;
    result.z = rhs.z * f;
    return result;
}

__device__ float3 operator*(const float3& lhs, const float3& rhs)
{
    float3 result;
    result.x = rhs.x * lhs.x;
    result.y = rhs.y * lhs.y;
    result.z = rhs.z * lhs.z;
    return result;
}
__device__ float3 operator/(const float3& lhs, const float f)
{
	if (f == 0.0f) 
    {
		return float3{ 0.0f, 0.0f, 0.0f }; // Avoid division by zero
	}

	float3 result;
	result.x = lhs.x / f;
	result.y = lhs.y / f;
	result.z = lhs.z / f;
	return result;
}

__device__ float Dot(const float3& lhs, const float3& rhs)
{
	float result=0;
	result += lhs.x * rhs.x;
	result += lhs.y * rhs.y;
	result += lhs.z * rhs.z;
	return result;
}

__device__ float3 Cross(const float3& lhs, const float3& rhs)
{
		return Unit(make_float3(
		lhs.y * rhs.z - lhs.z * rhs.y,
		lhs.z * rhs.x - lhs.x * rhs.z,
		lhs.x * rhs.y - lhs.y * rhs.x));
}

__device__ float3 Reflect(const float3& v, const float3& n)
{
	return Unit(v - 2 * Dot(v, n) * n);
}

__device__ bool LambertScatter(const CuRay& rayIn, const CuHitRecord& hitRecord, float3& attenuation, CuRay& scattered, curandState* state)
{
    float3 scatterDirection = hitRecord.mNormal + RandomUnitVector(state);
    // Catch degenerate scatter direction
    if (Dot(scatterDirection, scatterDirection) < 1e-8)
    {
        scatterDirection = hitRecord.mNormal;
    }
    float3 offsetOrigin = hitRecord.mPoint + 0.001f * hitRecord.mNormal; // Offset to avoid self-intersection
    scattered = CuRay(offsetOrigin, Unit(scatterDirection));
    attenuation = make_float3(1.0f, 1.0f, 1.0f); // Lambertian has no color attenuation
    return true;
}

__device__ bool MetalScatter(const CuRay& rayIn, const CuHitRecord& hitRecord, float3& attenuation, CuRay& scattered, curandState* state)
{
    float3 reflected = Reflect(Unit(rayIn.mDir), hitRecord.mNormal);
	reflected = Unit(reflected) + (0.1 * RandomUnitVector(state)); // Add some fuzziness
    float3 offsetOrigin = hitRecord.mPoint + 0.001f * hitRecord.mNormal; // Offset to avoid self-intersection
    scattered = CuRay(offsetOrigin, reflected); 
    attenuation = make_float3(1.0f, 1.0f, 1.0f); // Metal has no color attenuation
    return (Dot(scattered.mDir, hitRecord.mNormal) > 0.0f);
}


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

__global__ void renderRGBCudaKernel(float* redValues, float* greenValues, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	redValues[x + y * width] = static_cast<float>(x) / width;
    greenValues[x + y * width] = static_cast<float>(y) / height;
}

__global__ void renderSphereKernel(float* redValues, float* greenValues, float* blueValues, 
	float* accum_red, float* accum_green, float* accum_blue,
    int width, int height, float CameraDistance, float CameraHeight, curandState* state, 
    unsigned long long frameCount)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int id = y * width + x;
	curandState* localState = &state[id];

	CuSphere sphereWhite(make_float3(0.0f, -0.00f, -1.00f), 0.1f);
	sphereWhite.mAlbedo = make_float3(0.9f, 0.9f, 0.9f);

    CuSphere sphereGreen(make_float3(0.20f, -0.00f, -1.00f), 0.1f);
	sphereGreen.mAlbedo = make_float3(0.1f, 0.8f, 0.1f);

	CuSphere* spheres[] = { &sphereWhite, &sphereGreen };

	CuCamera mainCamera{
		make_float3(CameraHeight, 0, CameraDistance), // Camera position
		make_float3(0.0f, 0.0f, -1.0f), // Look at point
		make_float3(0.0f, 1.0f, 0.0f), // Up vector
		60.0f, // Field of view
		float(width) / float(height) // Aspect ratio
    };

    mainCamera.Initialize();

	float3 upperLeft = make_float3(-2.0f, 1.0f, -1.0f);
	float3 horizontal = make_float3(4.0f, 0.0f, 0.0f);
	float3 vertical = make_float3(0.0f, 2.0f, 0.0f);

	float u = float(x) / float(width);
	float v = float(y) / float(height);

    CuRay dummyRay;
	dummyRay.mOrigin = make_float3(0.0f, CameraHeight, CameraDistance); // Camera position
	dummyRay.mDir = Unit(upperLeft + u * horizontal - v * vertical);
	CuRay cameraRay = mainCamera.GetRay(x, y);


	CuRay r = cameraRay;

	float3 currentFrameColor = RayColor(cameraRay, sphereWhite, sphereGreen, localState); // Get the color from the ray tracing function

	float3 prevAccColor = make_float3(accum_red[id], accum_green[id], accum_blue[id]);
    float3 newAccumColor = prevAccColor + currentFrameColor;
    
    accum_red[id] = newAccumColor.x;
    accum_green[id] = newAccumColor.y;
    accum_blue[id] = newAccumColor.z;
   
    float3 finalAverageColor = newAccumColor / (float)frameCount;

    //
	redValues[x + y * width] = finalAverageColor.x; // Assign the color to the pixel
    greenValues[x + y * width] = finalAverageColor.y; // Assign the color to the pixel
    blueValues[x + y * width] = finalAverageColor.z; // Assign the color to the pixel
	

}

__global__ void SetupRandomState(curandState* state, unsigned long long seed, int width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int id = y * width + x;
    curand_init(seed, id, 0, &state[id]);
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

void renderRGBCuda(float* redValues, float* greenValues, int width, int height)
{
	float* d_redValues, * d_greenValues;

	cudaMalloc((void**)&d_redValues, width * height * sizeof(float));
	cudaMalloc((void**)&d_greenValues, width * height * sizeof(float));

    
    dim3 dimBlock(32, 32, 1);
	dim3 dimGrid((width) / dimBlock.x, (height) / dimBlock.y,1);

	renderRGBCudaKernel << <dimGrid, dimBlock >> > (d_redValues, d_greenValues, width, height);

	cudaDeviceSynchronize();

	cudaMemcpy(redValues, d_redValues, width * height * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(greenValues, d_greenValues, width * height * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_redValues);
	cudaFree(d_greenValues);
}

void renderSphereCuda(float* redValues, float* greenValues, float* blueValues,
    int width, int height, float fCameraDistance, float fCameraHeight,unsigned long long frameCount)
{
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid((width) / dimBlock.x, (height) / dimBlock.y, 1);

	SetupRandomState << < dimGrid, dimBlock >> > (d_randomState, frameCount, width);
    renderSphereKernel << <dimGrid, dimBlock >> > (
        d_redValues, d_greenValues, d_blueValues, 
		d_accum_red, d_accum_blue, d_accum_green,
        width, height, fCameraDistance, fCameraHeight, d_randomState, frameCount);

    cudaDeviceSynchronize();

    cudaMemcpy(redValues, d_redValues, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(greenValues, d_greenValues, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(blueValues, d_blueValues, width * height * sizeof(float), cudaMemcpyDeviceToHost);
}
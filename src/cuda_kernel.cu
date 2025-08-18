#include "cuda_kernel.cuh"


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
    float discriminant = b * b - 4 * a * c;

    if (discriminant < 0)
    {
        return false;
    }

    t = (-b - sqrtf(discriminant)) / (2.0f * a);

    return true;
}
__device__ bool CuSphere::Hit(const CuRay& ray, CuHitRecord& hitRecord, float tMin, float tMax) const
{
    float3 oc = ray.mOrigin - mOrigin;
    float a = Dot(ray.mDir, ray.mDir);
    float b = 2.0f * Dot(oc, ray.mDir);
    float c = Dot(oc, oc) - mRadius * mRadius;
    float discriminant = b*b - a*c;

    if (discriminant > 0)
    {
        float t = (-b - sqrtf(discriminant)) / a;
        if (t < tMax && t > tMin)
        {
            hitRecord.mT = t;
            hitRecord.mPoint = ray.At(t);
            hitRecord.mNormal = Unit((hitRecord.mPoint - mOrigin) / mRadius);
            return true;
        }
    }
	return true;
}

__device__ CuRay CuCamera::GetRay(float u, float v) const
{
	float3 dir = Unit(mUpperLeft + u * mHorizontal - v * mVertical);
	return CuRay(mOrigin, dir);
}

__device__ float3 Unit(const float3& InValue)
{
	float length = sqrtf(InValue.x * InValue.x + InValue.y * InValue.y + InValue.z * InValue.z);
	return float3{ InValue.x / length, InValue.y / length, InValue.z / length };
}

__device__ float3 GetColor(const CuRay& ray)
{
	float3 unitVec = Unit(ray.mDir);
	float t = 0.5f * (unitVec.y + 1.0f);
	return (1-t) * float3 {1.0f, 1.0f, 1.0f} + (t) * float3{ 0.5f, 0.7f, 1.0f }; // Gradient from white to blue
}

__device__ float3 operator+(const float3& lhs, const float3& rhs)
{
	return float3{ lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z };
}


__device__ float3 operator-(const float3& lhs, const float3& rhs)
{
    return float3{ lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z };
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

__global__ void renderSphereKernel(float* redValues, float* greenValues, float* blueValues, int width, int height, float CameraDistance, float CameraHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

	CuSphere sphere(make_float3(0.0f, -0.00f, -1.00f), 0.5f);

	float3 upperLeft = make_float3(-2.0f, 1.0f, -1.0f);
	float3 horizontal = make_float3(4.0f, 0.0f, 0.0f);
	float3 vertical = make_float3(0.0f, 2.0f, 0.0f);

	float u = float(x) / float(width);
	float v = float(y) / float(height);

    CuRay ray;
	ray.mOrigin = make_float3(0.0f, CameraHeight, CameraDistance); // Camera position
	ray.mDir = Unit(upperLeft + u * horizontal - v * vertical);

    float t = -1;
    if (sphere.Intersect(ray, t))
    {
        redValues[x + y * width] = 1;
        greenValues[x + y * width] = 0;
        blueValues[x + y * width] = 0;
    }
    else
    {
        float3 Color = GetColor(ray);
        redValues[x + y * width] = Color.x;
        greenValues[x + y * width] = Color.y;
        blueValues[x + y * width] = Color.z;
    }
}

__global__ void SetupRandomState(curandState* state, unsigned long long seed, int idx)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, state);
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

void renderSphereCuda(float* redValues, float* greenValues, float* blueValues,int width, int height, float fCameraDistance, float fCameraHeight)
{
    float* d_redValues, * d_greenValues, * d_blueValues;

    cudaMalloc((void**)&d_redValues, width * height * sizeof(float));
    cudaMalloc((void**)&d_greenValues, width * height * sizeof(float));
    cudaMalloc((void**)&d_blueValues, width * height * sizeof(float));


    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid((width) / dimBlock.x, (height) / dimBlock.y, 1);

    renderSphereKernel << <dimGrid, dimBlock >> > (d_redValues, d_greenValues, d_blueValues, width, height, fCameraDistance, fCameraHeight);

    cudaDeviceSynchronize();

    cudaMemcpy(redValues, d_redValues, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(greenValues, d_greenValues, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(blueValues, d_blueValues, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_redValues);
    cudaFree(d_greenValues);
    cudaFree(d_blueValues);
}
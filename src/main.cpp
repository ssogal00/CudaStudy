#define SDL_MAIN_HANDLED

#include <iostream>
#include "cuda_kernel.cuh"
#include "SDL2/SDL.h"
#include "SDL2/SDL_render.h"
#include "Complex.h"
#include <cmath>
#include <complex>
#include <thread>
#include "Vector.h"
#include <iomanip>

const int WIDTH = 1024;
const int HEIGHT = 512;

float CameraDistance = 0.65;
float CameraDistanceTarget = 0; 
float CameraHeight = 0;

float CameraTheta = 250;
float CameraAzimuth = 25;

void InitCudaBuffers(int width, int height)
{
    int numPixels = width * height;

    // 최종 픽셀 버퍼 할당
    cudaMalloc((void**)&d_redValues, numPixels * sizeof(float));
    cudaMalloc((void**)&d_greenValues, numPixels * sizeof(float));
    cudaMalloc((void**)&d_blueValues, numPixels * sizeof(float));

    // ★★★ 누적 버퍼 할당 ★★★
    // (float3로 한 번에 할당해도 되지만, 일단 코드를 유지합니다)
    cudaMalloc((void**)&d_accum_red, numPixels * sizeof(float));
    cudaMalloc((void**)&d_accum_green, numPixels * sizeof(float));
    cudaMalloc((void**)&d_accum_blue, numPixels * sizeof(float));

    // 랜덤 상태 버퍼 할당
    cudaMalloc((void**)&d_randomState, numPixels * sizeof(curandState));

    // 처음엔 누적 버퍼를 0으로 초기화
    cudaMemset(d_accum_red, 0, numPixels * sizeof(float));
    cudaMemset(d_accum_green, 0, numPixels * sizeof(float));
    cudaMemset(d_accum_blue, 0, numPixels * sizeof(float));
}
void ShutdownCuda()
{
    cudaFree(d_redValues);
    cudaFree(d_greenValues);
    cudaFree(d_blueValues);
    cudaFree(d_accum_red);
    cudaFree(d_accum_green);
    cudaFree(d_accum_blue);
    cudaFree(d_randomState);
}

void ResetAccBufferAndFrameCount()
{
	cudaMemset(d_accum_red, 0, WIDTH * HEIGHT * sizeof(float));
	cudaMemset(d_accum_green, 0, WIDTH * HEIGHT * sizeof(float));
	cudaMemset(d_accum_blue, 0, WIDTH * HEIGHT * sizeof(float));
	FrameCount = 0;
}

void render(SDL_Renderer* renderer, float* rValues, float* gValues, float* bValues)
{
	renderSphereCuda(rValues, gValues, bValues, WIDTH, HEIGHT, 
        CameraDistance, CameraTheta, CameraAzimuth, FrameCount);

	for (int y = 0; y < HEIGHT; ++y)
	{
		for (int x = 0; x < WIDTH; ++x)
		{
            uint32_t r = static_cast<uint32_t>(rValues[y * WIDTH + x] * 255.f);
            uint32_t g = static_cast<uint32_t>(gValues[y * WIDTH + x] * 255.f);
            uint32_t b = static_cast<uint32_t>(bValues[y * WIDTH + x] * 255.f);

			//std::cout << "r: " << r << " g: " << g << std::endl;
			SDL_SetRenderDrawColor(renderer, r, g, b, 255);
			SDL_RenderDrawPoint(renderer, x, y);
		}
	}
}
float easeInEaseOut(float source, float target, float alpha) 
{
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;
    float eased_alpha = (1.0f - std::cos(alpha * static_cast<float>(M_PI))) / 2.0f;
    return source + (target - source) * eased_alpha;
}

int main() 
{
    // vector test
	Vector4 v1(1.0f, 2.0f, 3.0f, 4.0f);

	std::cout << v1.Length() << std::endl;
	std::cout << v1.X() << " " << v1.Y() << " " << std::endl;

    // SDL 초기화
    if (SDL_Init(SDL_INIT_VIDEO) < 0) 
    {
        std::cerr << "SDL 초기화 실패: " << SDL_GetError() << std::endl;
        return 1;
    }

    // 윈도우 생성
    SDL_Window* window = SDL_CreateWindow(
        "CUDA Raytracing Example",                  // 윈도우 제목
        SDL_WINDOWPOS_UNDEFINED,        // 윈도우 x 위치
        SDL_WINDOWPOS_UNDEFINED,        // 윈도우 y 위치
        WIDTH,                            // 윈도우 너비
        HEIGHT,                            // 윈도우 높이
        SDL_WINDOW_SHOWN                // 윈도우 플래그
    );

    if (window == nullptr) 
    {
        std::cerr << "윈도우 생성 실패: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

	if (renderer == nullptr) 
    {
		std::cerr << "렌더러 생성 실패: " << SDL_GetError() << std::endl;
		SDL_DestroyWindow(window);
		SDL_Quit();
		return 1;
	}

    bool quit = false;
    SDL_Event event;

	float* PixelsR = new float[WIDTH * HEIGHT];
    float* PixelsG = new float[WIDTH * HEIGHT];
	float* PixelsB = new float[WIDTH * HEIGHT];

	InitCudaBuffers(WIDTH, HEIGHT);

    while (!quit) 
    {
        while (SDL_PollEvent(&event)) 
        {
            if (event.type == SDL_QUIT) 
            {
                quit = true;
            }
            else if (event.type == SDL_KEYDOWN)
            {
                if (event.key.keysym.sym == SDLK_UP)
                {
					CameraDistance -= 0.01f; // 카메라 거리를 줄여줍니다.
                    ResetAccBufferAndFrameCount();
                }
                else if (event.key.keysym.sym == SDLK_DOWN)
                {
					CameraDistance += 0.01f; // 카메라 거리를 늘려줍니다.
                    
                    ResetAccBufferAndFrameCount();
                }
				else if (event.key.keysym.sym == SDLK_LEFT)
				{
                    CameraTheta -= 5.f;
                    ResetAccBufferAndFrameCount();
				}
                else if (event.key.keysym.sym == SDLK_RIGHT)
                {
					CameraTheta += 5.f;
                    ResetAccBufferAndFrameCount();
                }
				else if (event.key.keysym.sym == SDLK_1)
				{
					CameraAzimuth += 5.f;
					ResetAccBufferAndFrameCount();
				}
				else if (event.key.keysym.sym == SDLK_2)
				{
                    CameraAzimuth -= 5.f;
					ResetAccBufferAndFrameCount();
				}
            }
        }

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        auto start = std::chrono::steady_clock::now();
        
        FrameCount++;

		render(renderer, PixelsR, PixelsG, PixelsB);

       
        auto end = std::chrono::steady_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

      
        std::cout << "Rendered in " << duration << " ms\r" << std::flush;

        SDL_RenderPresent(renderer);
    }

    ShutdownCuda();

    // 정리
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
    return 0;
}
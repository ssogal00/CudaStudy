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

std::complex<float> c(-0.29609091, 0.62491);  // You can change this constant to get different Julia sets
std::complex<float> dest(-0.20509091, 0.71591);

Complex8 C(-0.29609091f, 0.62491f);

const int WIDTH = 512;
const int HEIGHT = 512;

const int MAX_ITERATIONS = 75;

uint32_t julia(float x, float y) 
{
    std::complex<float> z(x, y);
    int iterations = 0;

    while (std::abs(z) < 2 && iterations < MAX_ITERATIONS) 
    {
        z = z * z + c;
        iterations++;
    }

    if (iterations == MAX_ITERATIONS) return 0;

    return static_cast<uint32_t>(iterations * 255 / MAX_ITERATIONS);
}

__m256 juliaSimd2(Complex8& A)
{

    Complex8 Z = A;
    __m256i NotAlreadyDiverged = _mm256_set1_epi32(0xFFFFFFFF);
    alignas(32) float iterationsArray[8] = { 0 };
    __m256 IterationsUntilDiverge = _mm256_load_ps(iterationsArray);

    for (int i = 0; i < MAX_ITERATIONS; ++i)
    {
        __m256 SquaredLength = Z.CalcSquaredLength();
        __m256 Diverging = _mm256_cmp_ps(SquaredLength, _mm256_set1_ps(4.0f), _CMP_GT_OQ);

        __m256i DivergingNow = _mm256_and_si256(NotAlreadyDiverged, _mm256_castps_si256(Diverging));
        _mm256_maskstore_ps(iterationsArray, DivergingNow, _mm256_set1_ps(static_cast<float>(i)));

        NotAlreadyDiverged = _mm256_andnot_si256(DivergingNow, NotAlreadyDiverged);

        if (_mm256_testz_si256(NotAlreadyDiverged, NotAlreadyDiverged))
            break;

        Z = Z * Z + C;
    }

    return _mm256_load_ps(iterationsArray);

}


#pragma optimize("", off)  
__m256 juliaSimd(Complex8& A)
{
    Complex8 Z = A;
    int iterations = 0;
    __m256 all_ones = _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()));
    __m256 IterationsUntilDiverge = _mm256_set1_ps(0.0f);
    __m256i NotAlreadyDiverged = _mm256_set1_epi32(0xFFFFFFFF);
	__m256i DivergedInThePast = _mm256_set1_epi32(0x0);

    // early out 
    for (int i = 0; i < MAX_ITERATIONS; ++i)
    {
		__m256 SquaredLength = Z.CalcSquaredLength();  // 8 float
		__m256 Length = _mm256_set1_ps(4.0f); // 8 float
		__m256 Diverging = _mm256_cmp_ps(SquaredLength, Length, _CMP_GT_OQ); // 8 float

        if (_mm256_testc_ps(Diverging, all_ones) == 1)
        {
            __m256i DivergingNow = _mm256_and_si256(NotAlreadyDiverged, _mm256_castps_si256(Diverging)); // 8 int
            _mm256_maskstore_ps((float*)&IterationsUntilDiverge, DivergingNow, _mm256_set1_ps(i)); // 8 float
            return IterationsUntilDiverge;
        }
        
        Z = Z * Z + C;

		__m256i DivergingNow = _mm256_and_si256(NotAlreadyDiverged, _mm256_castps_si256(Diverging)); // 8 int
		_mm256_maskstore_ps((float*)&IterationsUntilDiverge, DivergingNow, _mm256_set1_ps(i)); // 8 float

		NotAlreadyDiverged = _mm256_andnot_si256(DivergingNow, NotAlreadyDiverged); // 8 int
		DivergedInThePast = _mm256_or_si256(DivergedInThePast, DivergingNow); // 8 int

        //Z.MaskZeroPairs(DivergedInThePast);
    }

	return IterationsUntilDiverge;

}
#pragma optimize("", on)  
// normal 
void renderJuliaSet(SDL_Renderer* renderer) 
{
    for (int y = 0; y < HEIGHT; ++y) 
    {
        for (int x = 0; x < WIDTH; ++x) 
        {
            float real = (x - WIDTH / 2.0) * 4.0 / WIDTH;
            float imag = (y - HEIGHT / 2.0) * 4.0 / HEIGHT;

            uint32_t color = julia(real, imag);

            SDL_SetRenderDrawColor(renderer, color, color, color, 255);
            SDL_RenderDrawPoint(renderer, x, y);
            
        }
    }
}
// simd version
// simultaneous 8 pixels
void renderJuliaSetSimd(SDL_Renderer* renderer)
{
    for (int y = 0; y < HEIGHT; ++y)
    {
        for (int x = 0; x < WIDTH; x += 8)
        {
            float real0 = (x - WIDTH / 2.0) * 4.0 / WIDTH;
            float real1 = (x+1 - WIDTH / 2.0) * 4.0 / WIDTH;
            float real2 = (x+2 - WIDTH / 2.0) * 4.0 / WIDTH;
            float real3 = (x+3 - WIDTH / 2.0) * 4.0 / WIDTH;
            float real4 = (x+4 - WIDTH / 2.0) * 4.0 / WIDTH;
            float real5 = (x+5 - WIDTH / 2.0) * 4.0 / WIDTH;
            float real6 = (x+6 - WIDTH / 2.0) * 4.0 / WIDTH;
            float real7 = (x+7 - WIDTH / 2.0) * 4.0 / WIDTH;


            float imag = (y - HEIGHT / 2.0) * 4.0 / HEIGHT;

            Complex8 A{ real0, imag, real1, imag, real2, imag, real3, imag, real4, imag, real5, imag, real6, imag, real7, imag };

            __m256 Color = juliaSimd2(A);

			Color = _mm256_mul_ps(Color, _mm256_set1_ps(255.0f/MAX_ITERATIONS));
            
            for (int i = 0; i < 8; i++)
            {
                float* p = ((float*)&Color);
                SDL_SetRenderDrawColor(renderer, static_cast<uint32_t>(p[i]), static_cast<uint32_t>(p[i]), static_cast<uint32_t>(p[i]), 255);
                SDL_RenderDrawPoint(renderer, x+i, y);
            }
        }
    }
}


void renderJuliaSetWithCuda(SDL_Renderer* renderer, uint32_t* pixels)
{

	renderJuliaSetCuda(pixels, WIDTH, HEIGHT, c.real(), c.imag(), MAX_ITERATIONS);
    
	for (int y = 0; y < HEIGHT; ++y)
	{
		for (int x = 0; x < WIDTH; ++x)
		{
			uint32_t color = pixels[y * WIDTH + x];
			SDL_SetRenderDrawColor(renderer, color, color, color, 255);
			SDL_RenderDrawPoint(renderer, x, y);
		}
	}
}

void render(SDL_Renderer* renderer, float* rValues, float* gValues)
{

    renderRGBCuda(rValues, gValues, WIDTH, HEIGHT);

	for (int y = 0; y < HEIGHT; ++y)
	{
		for (int x = 0; x < WIDTH; ++x)
		{
            uint32_t r = static_cast<uint32_t>(rValues[y * WIDTH + x] * 255.f);
            uint32_t g = static_cast<uint32_t>(gValues[y * WIDTH + x] * 255.f);

			//std::cout << "r: " << r << " g: " << g << std::endl;
			SDL_SetRenderDrawColor(renderer, r, g, 20, 255);
			SDL_RenderDrawPoint(renderer, x, y);
		}
	}
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
        "SDL2 Window",                  // 윈도우 제목
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

	uint32_t* pixels = new uint32_t[WIDTH * HEIGHT];

	float* PixelsR = new float[WIDTH * HEIGHT];
    float* PixelsG = new float[WIDTH * HEIGHT];

    while (!quit) 
    {
        while (SDL_PollEvent(&event)) 
        {
            if (event.type == SDL_QUIT) 
            {
                quit = true;
            }

        }

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        auto start = std::chrono::steady_clock::now();

		render(renderer, PixelsR, PixelsG);

        auto end = std::chrono::steady_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

		std::cout << "Rendered in " << duration << " ms\n" << std::endl;
        
        SDL_RenderPresent(renderer);
    }


    // 정리
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
    return 0;
}
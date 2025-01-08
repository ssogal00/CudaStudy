#define SDL_MAIN_HANDLED

#include <iostream>
#include "cuda_kernel.cuh"
#include "SDL2/SDL.h"
#include "Complex.h"
#include <cmath>
#include <complex>
#include <thread>

std::complex<double> c(-0.29609091, 0.62491);  // You can change this constant to get different Julia sets
std::complex<double> dest(-0.20509091, 0.71591);

Complex8 C(-0.29609091f, 0.62491f);

const int WIDTH = 640;
const int HEIGHT = 480;

const int MAX_ITERATIONS = 75;

uint32_t julia(double x, double y) 
{
    std::complex<double> z(x, y);
    int iterations = 0;

    while (std::abs(z) < 2 && iterations < MAX_ITERATIONS) 
    {
        z = z * z + c;
        iterations++;
    }

    if (iterations == MAX_ITERATIONS) return 0;

    return static_cast<uint32_t>(iterations * 255 / MAX_ITERATIONS);
}

__m256 juliaSimd(Complex8& A)
{
    Complex8 Z = A;
    int iterations = 0;

    __m256 EarlyOutLength = Z.CalcSquaredLength();  // 8 float
    __m256 LengthCompare = _mm256_set1_ps(4.0f); // 8 float
	__m256 IterationsUntilDiverge = _mm256_set1_ps(0.0f);
    __m256 Diverging = _mm256_cmp_ps(EarlyOutLength, LengthCompare, _CMP_GT_OQ); // 8 float

    __m256 all_ones = _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()));
    
	if (_mm256_testc_ps(Diverging, all_ones) == 1)
	{
		return IterationsUntilDiverge;
	}


	__m256i NotAlreadyDiverged = _mm256_set1_epi32(0xFFFFFFFF);
	__m256i DivergedInThePast = _mm256_set1_epi32(0x0);

    // early out 


    for (int i = 0; i < MAX_ITERATIONS; ++i)
    {
        Z = Z * Z + C;

		__m256 SquaredLength = Z.CalcSquaredLength();  // 8 float
		__m256 Length = _mm256_set1_ps(4.0f); // 8 float
		__m256 Diverging = _mm256_cmp_ps(SquaredLength, Length, _CMP_GT_OQ); // 8 float

		__m256i DivergingNow = _mm256_and_si256(NotAlreadyDiverged, _mm256_castps_si256(Diverging)); // 8 int
		_mm256_maskstore_ps((float*)&IterationsUntilDiverge, DivergingNow, _mm256_set1_ps(i)); // 8 float

		NotAlreadyDiverged = _mm256_andnot_si256(DivergingNow, NotAlreadyDiverged); // 8 int
		DivergedInThePast = _mm256_or_si256(DivergedInThePast, DivergingNow); // 8 int

        Z.MaskZeroPairs(DivergedInThePast);
    }

	return IterationsUntilDiverge;
}

void renderJuliaSet(SDL_Renderer* renderer) 
{
    for (int y = 0; y < HEIGHT; ++y) 
    {
        for (int x = 0; x < WIDTH; ++x) 
        {
            double real = (x - WIDTH / 2.0) * 4.0 / WIDTH;
            double imag = (y - HEIGHT / 2.0) * 4.0 / HEIGHT;

            uint32_t color = julia(real, imag);

            SDL_SetRenderDrawColor(renderer, color, color, color, 255);
            SDL_RenderDrawPoint(renderer, x, y);
            
        }
    }
}

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

            __m256 Color = juliaSimd(A);

			Color = _mm256_mul_ps(Color, _mm256_set1_ps(255.0f));
			Color = _mm256_div_ps(Color, _mm256_set1_ps(MAX_ITERATIONS));
            
            for (int i = 0; i < 8; i++)
            {
                float* p = (float*)&Color;
                SDL_SetRenderDrawColor(renderer, static_cast<uint32_t>(p[i]), static_cast<uint32_t>(p[i]), static_cast<uint32_t>(p[i]), 255);
                //SDL_SetRenderDrawColor(renderer, static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0), 255);
                SDL_RenderDrawPoint(renderer, x+i, y);
            }
        }
    }
}


int main() 
{
	Complex4 a(1, 2, 3, 4, 1,2,3,4);

    Complex4 b(1, 2, 3, 4, 1, 2, 3, 4);

	Complex4 c = a * b;

	Complex8 d(1, 2, 3, 4, 
        5,6,7,8,
        9,10,11,12,
        13,14,15,16);

    PrintM256( d.CalcSquaredLength());

    __m256 A = _mm256_set_ps(1,2,3,4,5,6,7,8); // 8 float
    __m256 B = _mm256_set1_ps(4);

	__m256 C = _mm256_cmp_ps(A, B, _CMP_LT_OQ); 

	PrintM256(c.mValue);

    /*const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    add_arrays(a, b, c, arraySize);

    std::cout << "Result: ";
    for (int i = 0; i < arraySize; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    SDL_Init(SDL_INIT_VIDEO);
    while (1) {}
    SDL_Quit();
    */

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
        640,                            // 윈도우 너비
        480,                            // 윈도우 높이
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

    while (!quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                quit = true;
            }
        }

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        //renderJuliaSet(renderer);
		renderJuliaSetSimd(renderer);
		std::this_thread::sleep_for(std::chrono::milliseconds(16));

        SDL_RenderPresent(renderer);
    }


    // 정리
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
    return 0;
}
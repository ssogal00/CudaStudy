#define SDL_MAIN_HANDLED

#include <iostream>
#include "cuda_kernel.cuh"
#include "SDL2/SDL.h"
#include "Complex.h"
#include <cmath>
#include <complex>

std::complex<double> c(-0.29609091, 0.62491);  // You can change this constant to get different Julia sets
std::complex<double> dest(-0.20509091, 0.71591);

Complex8 C(-0.29609091f, 0.62491f);

const int WIDTH = 640;
const int HEIGHT = 480;

const int MAX_ITERATIONS = 100;

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

__m256 juliaSimd(const Complex8& A)
{
    for (int i = 0; i < MAX_ITERATIONS; ++i)
    {
        Complex8 Z = A * A + C;

		__m256 SquaredLength = Z.CalcSquaredLength();  // 8 float
		__m256 Length = _mm256_set1_ps(4.0f); // 8 float
		// If the length of Z is greater than 2

		__m256 Mask = _mm256_cmp_ps(SquaredLength, Length, _CMP_LT_OQ); // 8 float

    }
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


int main() 
{
	Complex4 a(1, 2, 3, 4, 1,2,3,4);

    Complex4 b(1, 2, 3, 4, 1, 2, 3, 4);

	Complex4 c = a * b;

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

        renderJuliaSet(renderer);

        SDL_RenderPresent(renderer);
    }


    // 정리
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
    return 0;
}
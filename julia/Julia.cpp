
#include <cmath>
#include <complex>
#include "SDL2/SDL.h"
#include "SDL2/SDL_render.h"
#include "JuliaKernel.cuh"

std::complex<float> c(-0.745f, 0.186f);;  // You can change this constant to get different Julia sets

std::complex<float> delta_c(0.0001f, -0.00015f);

const int MAX_ITERATIONS = 75;

const int WIDTH = 1024;
const int HEIGHT = 1024;

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

// 반복 횟수 'n'을 받아 RGB (0~255) 값으로 변환하는 함수 예시
void getColorFromIteration(int n, int max_iter, uint8_t* r, uint8_t* g, uint8_t* b)
{
    if (n == max_iter) {
        *r = *g = *b = 0; // 최대 반복 횟수: 검은색 (집합 내부)
        return;
    }

    // 반복 횟수를 0.0 ~ 1.0 사이의 값으로 정규화 (색상환의 위치)
    float t = (float)n / max_iter;

    // 이 예시는 간단한 무지개색 팔레트를 만듭니다.
    // Hue(색상)를 기반으로 컬러를 생성합니다.

    // 0 ~ 1530 범위의 인덱스
    int idx = (int)(t * 1530.0f);

    // 6단계 색상 변화의 위치 (0~5)
    int region = idx / 255;

    // 각 지역 내의 위치 (0~255)
    int p = idx % 255;

    switch (region) {
    case 0: *r = 230;      *g = p;        *b = 0;        break; // Red -> Yellow
    case 1: *r = 230 - p;  *g = 230;      *b = 0;        break; // Yellow -> Green
    case 2: *r = 0;        *g = 230;      *b = p;        break; // Green -> Cyan
    case 3: *r = 0;        *g = 230 - p;  *b = 230;      break; // Cyan -> Blue
    case 4: *r = p;        *g = 0;        *b = 230;      break; // Blue -> Magenta
    case 5: *r = 230;      *g = 0;        *b = 230 - p;  break; // Magenta -> Red
    default: *r = *g = *b = 0; break;
    }
}


void renderJuliaSetWithCuda(SDL_Renderer* renderer, uint32_t* pixels)
{
    c = c + delta_c;
    
    renderJuliaSetCuda(pixels, WIDTH, HEIGHT, c.real(), c.imag(), MAX_ITERATIONS);

    for (int y = 0; y < HEIGHT; ++y)
    {
        for (int x = 0; x < WIDTH; ++x)
        {
            uint32_t color = pixels[y * WIDTH + x];

            uint32_t iterations = pixels[y * WIDTH + x];

            uint8_t r, g, b;

            // 반복 횟수를 컬러 RGB 값으로 변환
            getColorFromIteration(iterations, MAX_ITERATIONS, &r, &g, &b);

            SDL_SetRenderDrawColor(renderer, r,g,b, 255);
            SDL_RenderDrawPoint(renderer, x, y);
        }
    }
}




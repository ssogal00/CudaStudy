#define SDL_MAIN_HANDLED

#include <iostream>

#include "SDL2/SDL.h"
#include "SDL2/SDL_render.h"

#include "Synchronize.cuh"
#include <cmath>
#include <complex>
#include <thread>

#include <iomanip>

const int WIDTH = 512;
const int HEIGHT = 512;



int main() 
{

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

	uint32_t* Pixels = new uint32_t[WIDTH * HEIGHT];

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

        //
		render(Pixels, WIDTH, HEIGHT);

        for (int y = 0; y < HEIGHT; ++y)
        {
            for (int x = 0; x < WIDTH; ++x)
            {
                uint32_t color = Pixels[y * WIDTH + x];                                

                SDL_SetRenderDrawColor(renderer, 0, color, 0, 255);
                SDL_RenderDrawPoint(renderer, x, y);
            }
        }


        SDL_RenderPresent(renderer);
    }

    // 정리
    SDL_DestroyWindow(window);
    SDL_Quit();


    return 0;
}
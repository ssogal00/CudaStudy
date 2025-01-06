#pragma once

#include <immintrin.h>
#include <xmmintrin.h>
#include <iostream>

class Complex4
{
public:

	Complex4(float real0, float imag0, float real1, float imag1, float real2, float imag2, float real3, float imag3)
	{
		mValue = _mm256_set_ps(imag3, real3, imag2, real2, imag1, real1, imag0, real0);
	}
	__m256 mValue;
};


class Complex2
{
public:

	Complex2(__m128 InValue) : mValue(InValue) 
	{ }

	Complex2(float real0, float imag0, float real1, float imag1)
	{
		mValue = _mm_set_ps(imag1, real1, imag0, real0);
	}

	void TestShuffle()
	{
		mValue = _mm_shuffle_ps(mValue, mValue, _MM_SHUFFLE(2, 3, 0, 1));
	}

	void DebugPrint()
	{
		float* p = (float*)&mValue;
		printf("%.2f %.2f %.2f %.2f\n", p[0], p[1], p[2], p[3]);

	}

	__m128 mValue;
};

void PrintM128(__m128 InValue);
void PrintM256(__m256 InValue);

Complex2 operator*(const Complex2& A, const Complex2& B);
Complex4 operator*(const Complex4& A, const Complex4& B);
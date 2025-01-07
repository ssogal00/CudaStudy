#pragma once

#include <immintrin.h>
#include <xmmintrin.h>
#include <iostream>

class Complex8
{
public:
	__m512 mValue;

	Complex8(__m512 InValue) : mValue(InValue)
	{
	}

	Complex8(float real0, float imag0, float real1, float imag1, float real2, float imag2, float real3, float imag3, float real4, float imag4, float real5, float imag5, float real6, float imag6, float real7, float imag7)
	{
		mValue = _mm512_set_ps(imag7, real7, imag6, real6, imag5, real5, imag4, real4, imag3, real3, imag2, real2, imag1, real1, imag0, real0);
	}


	__m256 CalcSquaredLength();
};

// can hold 4 complex numbers
class Complex4
{
public:
	Complex4(__m256 InValue) : mValue(InValue)
	{ 
	}

	Complex4(float real0, float imag0, float real1, float imag1, float real2, float imag2, float real3, float imag3)
	{
		mValue = _mm256_set_ps(imag3, real3, imag2, real2, imag1, real1, imag0, real0);
	}

	__m256 mValue;
};

// can hold 2 complex numbers
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
Complex8 operator*(const Complex8& A, const Complex8& B);
Complex8 operator+(const Complex8& A, const Complex8& B);

#include "Complex.h"

void PrintM128(__m128 InValue)
{
	float* p = (float*)&InValue;
	printf("%.2f %.2f %.2f %.2f\n", p[0], p[1], p[2], p[3]);
}

void PrintM256(__m256 InValue)
{
	float* p = (float*)&InValue;
	printf("%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
}

void PrintM512(__m512 InValue)
{
	float* p = (float*)&InValue;
	printf("%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", 
		p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
		p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15]);
}

__m256 Complex8::CalcSquaredLength()
{
	__m512 Squared = _mm512_mul_ps(mValue, mValue);

	__m256 low = _mm512_extractf32x8_ps(Squared, 0);
	__m256 high = _mm512_extractf32x8_ps(Squared, 1);

	__m256 sum_low = _mm256_hadd_ps(low, low);
	__m256 sum_high = _mm256_hadd_ps(high, high);

	return _mm256_blend_ps(sum_low, sum_high, 0xF0);
}


Complex2 operator*(const Complex2& A, const Complex2& B)
{
	__m128 BSwap = _mm_shuffle_ps(B.mValue, B.mValue, _MM_SHUFFLE(2, 3, 0, 1));
	PrintM128(BSwap);
	__m128 AImag = _mm_shuffle_ps(A.mValue, A.mValue, _MM_SHUFFLE(3, 3, 1, 1));
	PrintM128(AImag);
	__m128 AReal = _mm_shuffle_ps(A.mValue, A.mValue, _MM_SHUFFLE(2, 2, 0, 0));
	PrintM128(AReal);

	__m128 AImagMulBSwap = _mm_mul_ps(AImag, BSwap);
	PrintM128(AImagMulBSwap);

	return Complex2{ _mm_fmaddsub_ps(AReal, B.mValue, AImagMulBSwap) };
}

Complex4 operator*(const Complex4& A, const Complex4& B)
{
	__m256 BSwap = _mm256_shuffle_ps(B.mValue, B.mValue, _MM_SHUFFLE(2, 3, 0, 1));
	__m256 AImag = _mm256_shuffle_ps(A.mValue, A.mValue, _MM_SHUFFLE(3, 3, 1, 1));
	__m256 AReal = _mm256_shuffle_ps(A.mValue, A.mValue, _MM_SHUFFLE(2, 2, 0, 0));

	__m256 AImagMulBSwap = _mm256_mul_ps(AImag, BSwap);

	return Complex4{ _mm256_fmaddsub_ps(AReal, B.mValue, AImagMulBSwap) };
}

Complex8 operator*(const Complex8& A, const Complex8& B)
{
	__m512 BSwap = _mm512_shuffle_ps(B.mValue, B.mValue, _MM_SHUFFLE(2, 3, 0, 1));
	__m512 AImag = _mm512_shuffle_ps(A.mValue, A.mValue, _MM_SHUFFLE(3, 3, 1, 1));
	__m512 AReal = _mm512_shuffle_ps(A.mValue, A.mValue, _MM_SHUFFLE(2, 2, 0, 0));

	__m512 AImagMulBSwap = _mm512_mul_ps(AImag, BSwap);

	return Complex8{ _mm512_fmaddsub_ps(AReal, B.mValue, AImagMulBSwap) };
}

Complex8 operator+(const Complex8& A, const Complex8& B)
{
	return Complex8{ _mm512_add_ps(A.mValue, B.mValue) };
}
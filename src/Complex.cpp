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
#include "Complex.h"

void PrintM128(__m128 InValue)

{
	float* p = (float*)&InValue;
	printf("%.2f %.2f %.2f %.2f\n", p[3], p[2], p[1], p[0]);
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

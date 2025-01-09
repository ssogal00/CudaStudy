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

	__m256 even = _mm512_castps512_ps256(_mm512_maskz_compress_ps(0x5555, Squared));

	__m256 odd = _mm512_castps512_ps256(_mm512_maskz_compress_ps(0xAAAA, Squared));

	__m256 sum = _mm256_add_ps(even, odd);

	return sum;
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

Complex8 MaskedMultiply(const Complex8& A, const Complex8& B, unsigned int mask)
{
	__m512 BSwap = _mm512_shuffle_ps(B.mValue, B.mValue, _MM_SHUFFLE(2, 3, 0, 1));
	__m512 AImag = _mm512_shuffle_ps(A.mValue, A.mValue, _MM_SHUFFLE(3, 3, 1, 1));
	__m512 AReal = _mm512_shuffle_ps(A.mValue, A.mValue, _MM_SHUFFLE(2, 2, 0, 0));
	
	__m512 AImagMulBSwap = _mm512_mask_mul_ps(AImag, mask, AImag, BSwap);

	return Complex8{ _mm512_mask_fmaddsub_ps(AReal, mask, B.mValue, AImagMulBSwap)};
}


void Complex8::MaskZeroPairs(__m256i mask)
{
	__m512i expanded_mask = _mm512_cvtepi32_epi64(mask);

	expanded_mask = _mm512_permutexvar_epi64(_mm512_set_epi64(7, 7, 6, 6, 5, 5, 4, 4), expanded_mask);

	// float를 int로 재해석하여 마스킹 연산을 수행합니다
	__m512i src_as_int = _mm512_castps_si512(mValue);
	__m512i masked = _mm512_andnot_si512(expanded_mask, src_as_int);

	// 결과를 다시 float로 변환합니다
	mValue = _mm512_castsi512_ps(masked);
}

Complex8 operator+(const Complex8& A, const Complex8& B)
{
	return Complex8{ _mm512_add_ps(A.mValue, B.mValue) };
}
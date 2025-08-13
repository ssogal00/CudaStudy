

#include "Vector.h"

Vector4 operator+(const Vector4& a, const Vector4& b)
{
	return Vector4(_mm_add_ps(a.mValue, b.mValue));

}

Vector4 operator-(const Vector4& a, const Vector4& b)
{
	return Vector4(_mm_sub_ps(a.mValue, b.mValue));
}

float Vector4::Length() const
{
	__m128 squared = _mm_mul_ps(mValue, mValue);
	__m128 sum = _mm_hadd_ps(squared, squared);
	sum = _mm_hadd_ps(sum, sum);
	return _mm_cvtss_f32(_mm_sqrt_ss(sum));	
}
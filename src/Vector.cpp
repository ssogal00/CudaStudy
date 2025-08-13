

#include "Vector.h"

Vector4 operator+(const Vector4& a, const Vector4& b)
{
	return Vector4(_mm_add_ps(a.mValue, b.mValue));

}

Vector4 operator-(const Vector4& a, const Vector4& b)
{
	return Vector4(_mm_sub_ps(a.mValue, b.mValue));
}
Vector4 operator*(const Vector4& a, float b)
{
	return Vector4(_mm_mul_ps(a.mValue, _mm_set1_ps(b)));
}
Vector4 operator*(float a, const Vector4& b)
{
	return Vector4(_mm_mul_ps(_mm_set1_ps(a), b.mValue));
}

float Vector4::Length() const
{
	__m128 squared = _mm_mul_ps(mValue, mValue);
	__m128 sum = _mm_hadd_ps(squared, squared);
	sum = _mm_hadd_ps(sum, sum);
	return _mm_cvtss_f32(_mm_sqrt_ss(sum));	
}

float Vector4::LengthSquared() const
{
	__m128 squared = _mm_mul_ps(mValue, mValue);
	__m128 sum = _mm_hadd_ps(squared, squared);
	sum = _mm_hadd_ps(sum, sum);
	return _mm_cvtss_f32(sum);
}

float Vector4::X() const
{
	float x = _mm_cvtss_f32(mValue); // 첫 번째 요소 (x)
	return x;
}

float Vector4::Y() const
{
	float y = _mm_cvtss_f32(_mm_shuffle_ps(mValue, mValue, _MM_SHUFFLE(0, 0, 0, 1))); // 두 번째 요소 (y)
	return y;
}

float Vector4::Z() const
{
	float z = _mm_cvtss_f32(_mm_shuffle_ps(mValue, mValue, _MM_SHUFFLE(0, 0, 1, 2))); // 세 번째 요소 (z)
	return z;
}


Vector3 operator*(const Vector3& a, float b)
{
	return Vector3(_mm_mul_ps(a.mValue, _mm_set1_ps(b)));
}

Vector3 operator*(float a, const Vector3& b)
{
	return Vector3(_mm_mul_ps(_mm_set1_ps(a), b.mValue));
}

void Vector3::Normalize()
{
	__m128 length = _mm_sqrt_ps(_mm_hadd_ps(_mm_hadd_ps(mValue, mValue), _mm_setzero_ps()));
	mValue = _mm_div_ps(mValue, _mm_shuffle_ps(length, length, _MM_SHUFFLE(0, 0, 0, 0)));
}

float Vector3::Length() const
{
	__m128 squared = _mm_mul_ps(mValue, mValue);
	__m128 sum = _mm_hadd_ps(squared, squared);
	sum = _mm_hadd_ps(sum, sum);
	return _mm_cvtss_f32(_mm_sqrt_ss(sum));
}


float Vector3::LengthSquared() const
{
	__m128 squared = _mm_mul_ps(mValue, mValue);
	__m128 sum = _mm_hadd_ps(squared, squared);
	sum = _mm_hadd_ps(sum, sum);
	return _mm_cvtss_f32(sum);
}

Vector3 Ray::GetPointAt(float t) const
{
	return mOrigin + (mDirection * t);
}

Vector3 operator+(const Vector3& a, const Vector3& b)
{
	return Vector3(_mm_add_ps(a.mValue, b.mValue));

}
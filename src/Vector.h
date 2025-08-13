
#pragma once

#include <immintrin.h>
#include <xmmintrin.h>

class alignas(16) Vector4
{
public:

	Vector4() : mValue(_mm_setzero_ps()) {}
	
	Vector4(float x, float y, float z, float w)
		: mValue(_mm_set_ps(w, z, y, x)) {}

	Vector4(const __m128& value) : mValue(value) {}

	friend Vector4 operator+(const Vector4& a, const Vector4& b);
	friend Vector4 operator-(const Vector4& a, const Vector4& b);
	// dot
	friend Vector4 operator*(const Vector4& a, const Vector4& b);

	float Length() const;

	__m128 mValue;
};

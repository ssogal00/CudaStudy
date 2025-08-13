
#pragma once

#include <immintrin.h>
#include <xmmintrin.h>


class alignas(16) Vector4
{
public:

	Vector4() : mValue(_mm_setzero_ps()) {}
	
	Vector4(float x, float y, float z, float w)
		: mValue(_mm_set_ps(w, z, y, x)) {}

	float X() const;
	float Y() const;
	float Z() const;
	float W() const;

	Vector4(const __m128& value) : mValue(value) {}

	friend Vector4 operator+(const Vector4& a, const Vector4& b);
	friend Vector4 operator-(const Vector4& a, const Vector4& b);
	// dot
	friend Vector4 operator*(const Vector4& a, const Vector4& b);
	friend Vector4 operator*(const Vector4& a, float b);
	friend Vector4 operator*(float a, const Vector4& b);

	float Length() const;
	float LengthSquared() const;

private:
	__m128 mValue;
};

class alignas(16) Vector3 
{
public:

	Vector3() : mValue(_mm_setzero_ps()) {}

	Vector3(const Vector3& other) : mValue(other.mValue) {}

	Vector3(float x, float y, float z)
		: mValue(_mm_set_ps(0.0f, z, y, x)) {}

	Vector3(const __m128& value) : mValue(value) {}

	Vector3 Normalize() const;

	void Normalize();
	friend Vector3 operator+(const Vector3& a, const Vector3& b);
	friend Vector3 operator-(const Vector3& a, const Vector3& b);
	// dot
	friend Vector3 operator*(const Vector3& a, const Vector3& b);
	friend Vector3 operator*(const Vector3& a, float b);
	friend Vector3 operator*(float a, const Vector3& b);

	float Length() const;
	float LengthSquared() const;
private:
	__m128 mValue;
};


class Ray
{
public:
	Ray(const Vector3& Origin, const Vector3& Dir)
		: mOrigin(Origin), mDirection(Dir)
	{
	}

	Vector3 GetPointAt(float t) const;

private:
	Vector3 mOrigin;
	Vector3 mDirection;
	
};
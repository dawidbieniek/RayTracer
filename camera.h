#ifndef CAMERA_H
#define CAMERA_H

#include "doubleUtils.h"
#include "vec3.h"
#include "ray.h"

class camera 
{
public:
	__device__ camera() {};

	__device__ camera(vec3 position, vec3 lookAt, vec3 up, float vFov, float aspect)
	{
		origin = position;
		lookAtVec = lookAt;
		upVec = up;
		vFovVal = vFov;
		aspectRatio = aspect;

		updateValues();
	}

	__device__ inline void updatePosition(vec3 translation)
	{
		origin += translation;

		updateValues();
	}

	__device__ inline ray getRay(double u, double v) const
	{
		return ray(origin, lowerLeftCorner + u * horizontal + v * vertical - origin);
	}

private:
	vec3 origin;
	vec3 lookAtVec;
	vec3 upVec;
	float vFovVal;
	float aspectRatio;

	vec3 lowerLeftCorner;
	vec3 horizontal;
	vec3 vertical;

	__device__ inline void updateValues()
	{
		vec3 u;
		vec3 v;
		vec3 w;
		float theta = vFovVal * 3.14159265 / 180;
		float halfHeight = tan(theta / 2);
		float halfWidth = aspectRatio * halfHeight;
		w = unit_vector(origin - lookAtVec);
		u = unit_vector(cross(upVec, w));
		v = cross(w, u);
		lowerLeftCorner = origin - halfWidth * u - halfHeight * v - w;
		horizontal = 2 * halfWidth * u;
		vertical = 2 * halfHeight * v;
	}
};
#endif
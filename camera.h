#ifndef CAMERA_H
#define CAMERA_H

#include "doubleUtils.h"
#include "vec3.h"
#include "ray.h"

class camera 
{
public:
	__device__ __host__ camera() {};

	__device__ __host__ camera(vec3 position, vec3 lookAt, vec3 up, float vFov, float aspect) 
	{
		//float aspectRatio = aspect;
		//float viewportHeight = 2.0;
		//float viewportWidth = aspectRatio * viewportHeight;
		//float focalLength = 1.0;

		//origin = vec3(0.0, 0.0, 0.0);
		//horizontal = vec3(viewportWidth, 0.0, 0.0);
		//vertical = vec3(0.0, viewportHeight, 0.0);
		//lowerLeftCorner = origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focalLength);
		vec3 u;
		vec3 v;
		vec3 w;
		float theta = vFov * 3.14159265 / 180;
		float halfHeight = tan(theta / 2);
		float halfWidth = aspect * halfHeight;
		origin = position;
		w = unit_vector(position - lookAt);
		u = unit_vector(cross(up, w));
		v = cross(w, u);
		lowerLeftCorner = origin - halfWidth * u - halfHeight * v - w;
		horizontal = 2 * halfWidth * u;
		vertical = 2 * halfHeight * v;
	}

	__device__ ray getRay(double u, double v) const
	{
		return ray(origin, lowerLeftCorner + u * horizontal + v * vertical - origin);
	}

private:
	vec3 origin;
	vec3 lowerLeftCorner;
	vec3 horizontal;
	vec3 vertical;
};
#endif
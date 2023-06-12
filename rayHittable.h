#ifndef RAYHITTABLE_H
#define RAYHITTABLE_H

#include "ray.h"

struct hitInfo
{
	vec3 point;
	vec3 normal;
	double t;
	bool frontFace;

	__device__ void setFaceNormal(const ray& r, const vec3& outNormal)
	{
		frontFace = dot(r.direction(), outNormal) < 0;
		normal = frontFace ? outNormal : -outNormal;
	}
};

class rayHittable
{
public:
	__device__ virtual bool hit(const ray& r, double tMin, double tMax, hitInfo& hit) const = 0;
};

#endif // RAYHITTABLE_H

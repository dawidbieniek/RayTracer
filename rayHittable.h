#ifndef RAYHITTABLE_H
#define RAYHITTABLE_H

#include "ray.h"

class material;	// To avoid include cycles

struct hitInfo
{
	vec3 point;
	vec3 normal;
	double t;
	material* matPtr;
};

class rayHittable
{
public:
	__device__ virtual bool hit(const ray& r, double tMin, double tMax, hitInfo& hit) const = 0;
};

#endif // RAYHITTABLE_H

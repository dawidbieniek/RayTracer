#ifndef SPHERE_H
#define SPHERE_H

#include "rayHittable.h"
#include "ray.h"
#include "material.h"

class sphere : public rayHittable 
{
public:
	__device__ sphere() {}
	__device__ sphere(vec3 c, double r, material *mat) : center(c), radius(r), matPtr(mat) {};

	__device__ virtual bool hit(const ray& r, double tMin, double tMax, hitInfo& hit) const override;

private:
	vec3 center;
	double radius;
	material* matPtr;
};

// Casts ray to hit sphere. Returns true if hit and false when missed. rec contains hitInfo
__device__ bool sphere::hit(const ray& r, double tMin, double tMax, hitInfo& hit) const 
{
	vec3 oc = r.origin() - center;
	float a = r.direction().lengthSquared();
	float halfB = dot(oc, r.direction());
	float c = oc.lengthSquared() - radius * radius;

	float discriminant = halfB * halfB - a * c;
	if (discriminant < 0) return false;

	float sqrdDisc = sqrt(discriminant);

	float root = (-halfB - sqrdDisc) / a;
	if (root < tMin || tMax < root) 
	{
		root = (-halfB + sqrdDisc) / a;
		if (root < tMin || tMax < root)
			return false;
	}

	hit.t = root;
	hit.point = r.at(hit.t);
	vec3 outNormal = (hit.point - center) / radius;
	//rec.setFaceNormal(r, outNormal);
	hit.normal = (hit.point - center) / radius;
	hit.matPtr = matPtr;

	return true;
}
#endif // SPHERE_H

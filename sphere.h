#ifndef SPHERE_H
#define SPHERE_H

#include "rayHittable.h"
#include "ray.h"

class sphere : public rayHittable 
{
public:
	__device__ sphere() {}
	__device__ sphere(vec3 c, double r) : center(c), radius(r) {};

	__device__ virtual bool hit(const ray& r, double tMin, double tMax, hitInfo& hit) const override;

private:
	vec3 center;
	double radius;
};

// Casts ray to hit sphere. Returns true if hit and false when missed. rec contains hitInfo
__device__ bool sphere::hit(const ray& r, double tMin, double tMax, hitInfo& rec) const 
{
	vec3 oc = r.origin() - center;
	float a = r.direction().lengthSquared();
	float half_b = dot(oc, r.direction());
	float c = oc.lengthSquared() - radius * radius;

	float discriminant = half_b * half_b - a * c;
	if (discriminant < 0) return false;

	float sqrtd = sqrt(discriminant);

	// Find the nearest root that lies in the acceptable range.
	float root = (-half_b - sqrtd) / a;
	if (root < tMin || tMax < root) {
		root = (-half_b + sqrtd) / a;
		if (root < tMin || tMax < root)
			return false;
	}

	rec.t = root;
	rec.point = r.at(rec.t);    
	vec3 outNormal = (rec.point - center) / radius;
	rec.setFaceNormal(r, outNormal);
	rec.normal = (rec.point - center) / radius;

	return true;
}
#endif // SPHERE_H

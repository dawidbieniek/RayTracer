#ifndef SCENE_H
#define SCENE_H

#include "rayHittable.h"
#include "ray.h"

class scene : public rayHittable
{
public:
	__device__ scene() {};
	__device__ scene(rayHittable** objects, int n) { objectList = objects; objectCount = n; }

	__device__ virtual bool hit(const ray& r, double tMin, double tMax, hitInfo& info) const override;

	rayHittable** objectList;
	int objectCount;
};


__device__ bool scene::hit(const ray& r, double tMin, double tMax, hitInfo& info) const
{
	hitInfo tempInfo;
	bool hitAnything = false;
	float closest = tMax;

	for (int i = 0; i < objectCount; i++) {
		if (objectList[i]->hit(r, tMin, closest, tempInfo)) {
			hitAnything = true;
			closest = tempInfo.t;
			info = tempInfo;
		}
	}
	return hitAnything;
}

#endif // SCENE_H

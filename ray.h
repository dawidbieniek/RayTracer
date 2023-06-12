#ifndef RAY_H
#define RAY_H
#include "vec3.h"

class ray
{
public:
    __device__ ray() {}
    // Creates ray from origin point and direction vector 
    __device__ ray(const vec3& origin, const vec3& dir) { _origin = origin; _dir = dir; }
    __device__ vec3 origin() const { return _origin; }
    __device__ vec3 direction() const { return _dir; }
    // Returns point reached by ray of length t
    __device__ vec3 point_at_parameter(float t) const { return _origin + t * _dir; }

private:
    vec3 _origin;
    vec3 _dir;
};

#endif // RAY_H



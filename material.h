#ifndef MATERIAL_H
#define MATERIAL_H

#include "ray.h"
#include "vec3.h"
#include "rayHittable.h"

__device__ vec3 reflect(const vec3& v, const vec3& n) 
{
    return v - 2.0f * dot(v, n) * n;
}

class material 
{
public:
    __device__ virtual bool scatter(const ray& inputRay, const hitInfo& hit, vec3& attenuation, ray& scattered, curandState localState) const = 0;
};

class lambertian : public material 
{
public:
    __device__ lambertian(const vec3& a) : albedo(a) {}
    __device__ virtual bool scatter(const ray& inputRay, const hitInfo& hit, vec3& attenuation, ray& scattered, curandState localState) const
    {
#ifndef DIFFUSE_HALF_SPHERE
        vec3 target = hit.point + hit.normal + randomVecInSphere(localState);
#else
        vec3 target = hit.point + hit.normal + randomVecInHalfSphere(hit.normal, localState);
#endif
        scattered = ray(hit.point, target - hit.point);
        attenuation = albedo;
        return true;
    }

    vec3 albedo;
};

class metal : public material 
{
public:
    __device__ metal(const vec3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
    __device__ virtual bool scatter(const ray& inputRay, const hitInfo& hit, vec3& attenuation, ray& scattered, curandState localState) const 
    {
        vec3 reflected = reflect(unit_vector(inputRay.direction()), hit.normal);
#ifndef DIFFUSE_HALF_SPHERE
        scattered = ray(hit.point, reflected + fuzz * randomVecInSphere(localState));
#else
        scattered = ray(hit.point, reflected + fuzz * randomVecInHalfSphere(hit.normal, localState));
#endif
        attenuation = albedo;
        return (dot(scattered.direction(), hit.normal) > 0.0f);
    }
    vec3 albedo;
    float fuzz;
};


#endif // MATERIAL_H
#ifndef MATERIAL_H
#define MATERIAL_H

#include "ray.h"
#include "vec3.h"
#include "rayHittable.h"

__device__ inline vec3 reflect(const vec3& v, const vec3& n) 
{
    return v - 2.0f * dot(v, n) * n;
}
__device__ inline bool refract(const vec3& v, const vec3& n, float ni_nt, vec3& refracted) 
{
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_nt * ni_nt * (1 - dt * dt);
    if (discriminant > 0) 
    {
        refracted = ni_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else
        return false;
}

__device__ inline float schlick(float cosine, float refractionIndex) 
{
    float r0 = (1.0f - refractionIndex) / (1.0f + refractionIndex);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

class material 
{
public:
    __device__ inline virtual bool scatter(const ray& inputRay, const hitInfo& hit, vec3& attenuation, ray& scattered, curandState localState) const = 0;
};

class lambertian : public material 
{
public:
    __device__ lambertian(const vec3& a) : albedo(a) {}
    __device__ inline virtual bool scatter(const ray& inputRay, const hitInfo& hit, vec3& attenuation, ray& scattered, curandState localState) const
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
    __device__ inline bool scatter(const ray& inputRay, const hitInfo& hit, vec3& attenuation, ray& scattered, curandState localState) const
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

class dielectric : public material
{
public:
    __device__ dielectric(float ri) : refractionIndex(ri) {}
    __device__ inline bool scatter(const ray& inputRay, const hitInfo& hit, vec3& attenuation, ray& scattered, curandState localState) const
    {
        vec3 outward_normal;
        vec3 reflected = reflect(inputRay.direction(), hit.normal);
        float ni_nt;
        attenuation = vec3(1.0, 1.0, 1.0);
        vec3 refracted;
        float reflectProbe;
        float cosine;

        if (dot(inputRay.direction(), hit.normal) > 0.0f)
        {
            outward_normal = -hit.normal;
            ni_nt = refractionIndex;
            cosine = dot(inputRay.direction(), hit.normal) / inputRay.direction().length();
            cosine = sqrt(1.0f - refractionIndex * refractionIndex * (1 - cosine * cosine));
        }
        else
        {
            outward_normal = hit.normal;
            ni_nt = 1.0f / refractionIndex;
            cosine = -dot(inputRay.direction(), hit.normal) / inputRay.direction().length();
        }
        if (refract(inputRay.direction(), outward_normal, ni_nt, refracted))
            reflectProbe = schlick(cosine, refractionIndex);
        else
            reflectProbe = 1.0f;
        if (curand_uniform(&localState) < reflectProbe)
            scattered = ray(hit.point, reflected);
        else
            scattered = ray(hit.point, refracted);
        return true;
    }

    float refractionIndex;
};



#endif // MATERIAL_H
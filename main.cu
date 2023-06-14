// Standard libs
#include <iostream>
#include <memory>

// GL libs
#include <GL/glut.h>

// CUDA libs
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

// My libs
// Wrapper define
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		// Print message
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) 
			<< " at " << file 
			<< ":" << line 
			<< " '" << func << "' \n" 
			<< cudaGetErrorString(result) << "\n";
		// Reset GPU
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

#include "vec3.h"
#include "ray.h"
#include "scene.h"
#include "rayHittable.h"
#include "sphere.h"
#include "doubleUtils.h"
#include "camera.h"
#include "material.h"

#define USE_GAMMA_CORRECTION
#define DIFFUSE_HALF_SPHERE

static const int TARGET_FPS = 60;
static const unsigned int FPS_DISPLAY_REFRESH_TIME = 500;
static const int SAMPLES_PER_PIXEL = 100;
static const int MAX_DIFFUSE_DEPTH = 20;
static const int SCENE_ELEMENTS = 5;

#define BACKGROUND_START_GRADIENT_COLOR vec3(0.5, 0.7, 1.0)
#define BACKGROUND_END_GRADIENT_COLOR vec3(1.0, 1.0, 1.0)

const int screenWidth = 960;
const int screenHeight = 480;

__device__ camera dCam;

// Returns length of ray from origin to hit point. -1 if not hit
__device__ double sphereHitPoint(const vec3& center, float radius, const ray& r)
{
	vec3 oc = r.origin() - center;
	auto a = r.direction().lengthSquared();
	auto half_b = dot(oc, r.direction());
	auto c = oc.lengthSquared() - radius * radius;
	auto discriminant = half_b * half_b - a * c;

	if (discriminant < 0) return -1.0;
	return (-half_b - sqrt(discriminant)) / a;
}

__device__ vec3 color(const ray& r, scene** dScene, curandState localState)
{
	ray currentRay = r;
	vec3 currentAttenuation = vec3(1.0, 1.0, 1.0);

	// NOTE: Recursion blows up GPU stack, so instead I use iterative recursion
	for (int i = 0; i < MAX_DIFFUSE_DEPTH; i++) 
	{
		hitInfo hit;
		if ((*dScene)->hit(currentRay, 0.001f, INFINITY, hit)) 
		{
			ray scattered;
			vec3 attenuation;
			if (hit.matPtr->scatter(currentRay, hit, attenuation, scattered, localState)) 
			{
				currentAttenuation *= attenuation;
				currentRay = scattered;
			}
			else 
			{
				return vec3(0.0, 0.0, 0.0);
			}
		}
		else 
		{
			vec3 unit_direction = unit_vector(currentRay.direction());
			float t = 0.5f * (unit_direction.y() + 1.0f);
			vec3 c = (1.0f - t) * BACKGROUND_END_GRADIENT_COLOR + t * BACKGROUND_START_GRADIENT_COLOR;
			return currentAttenuation * c;
		}
	}
	// Over depth limit
	return vec3(0.0, 0.0, 0.0);
}

__global__ void createCamera(int width, int height)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		dCam = camera(width, height);
	}
}

__global__ void setupRNG(curandState* globalState, int seed, int screenWidth)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int id = j * screenWidth + i;
	curand_init(seed, id, 0, &globalState[id]);
}

__global__ void createScene(rayHittable** dObjects, scene** dScene)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		*(dObjects) = new sphere(vec3(0, 0, -2), 0.5, new lambertian(vec3(0.8, 0.2, 0.2)));
		*(dObjects + 1) = new sphere(vec3(-2, -1, -5), 1, new lambertian(vec3(0.0, 0.8, 0.8)));
		*(dObjects + 2) = new sphere(vec3(0, -100.5, -1), 100, new lambertian(vec3(0.0, 0.8, 0.0)));
		*(dObjects + 3) = new sphere(vec3(1.5, 0.5, -2), 0.5, new lambertian(vec3(1, 1, 1)));
		*(dObjects + 4) = new sphere(vec3(-1.5, 0.5, -2), 0.5, new lambertian(vec3(0, 0, 0)));
		*dScene = new scene(dObjects, SCENE_ELEMENTS);
	}
}

__global__ void render(vec3* fb, int maxX, int maxY, scene** dScene, curandState* globalState)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= maxX) || (j >= maxY)) return;
	
	int pixelIndex = j * maxX + i;
	curandState localState = globalState[pixelIndex];
	for (int p = 0; p < SAMPLES_PER_PIXEL; p++)	// TODO: Maybe divide this across threads
	{
		float u = (float(i) + curand_uniform(&localState)) / float(maxX);
		float v = (float(j) + curand_uniform(&localState)) / float(maxY);
		ray r = dCam.getRay(u, v);
#ifndef USE_GAMMA_CORRECTION
		fb[pixelIndex] += color(r, dScene, localState);
#else
		vec3 clr = color(r, dScene, localState);
		clr.v3sqrt();
		fb[pixelIndex] += clr;
#endif
	}
	fb[pixelIndex] /= SAMPLES_PER_PIXEL;
}

vec3* fb;

int lastFrameTime = 0;
int deltaTimes[TARGET_FPS];
int deltaTimesIndex = 0;

void refreshFrameCallback(int value)
{
	if (glutGetWindow()) 
	{
		int currentTime = glutGet(GLUT_ELAPSED_TIME);
		int deltaTime = currentTime - lastFrameTime;
		lastFrameTime = currentTime;
		deltaTimes[deltaTimesIndex++] = deltaTime;
		if (deltaTimesIndex >= TARGET_FPS)
			deltaTimesIndex = 0;

		// Refresh window
		glutPostRedisplay();
		// Refresh callback
		glutTimerFunc((unsigned int)(1000.0 / TARGET_FPS), refreshFrameCallback, NULL);
	}
}

void displayFPSCountCallback(int value)
{
	if (glutGetWindow())
	{
		float fpsCount = 0;
		int i;
		for (i = 0; i < TARGET_FPS; i++)
		{
			if (deltaTimes[i] == 0)
				break;
			fpsCount += deltaTimes[i];
		}

		fpsCount  = 1000.0 / (fpsCount / i + 1);

		char titleBuffer[16];
		sprintf(titleBuffer, "FPS: %3.1f", fpsCount);
		glutSetWindowTitle(titleBuffer);

		// Refresh callback
		glutTimerFunc(FPS_DISPLAY_REFRESH_TIME, displayFPSCountCallback, NULL);
	}
}

void draw()
{
	glBegin(GL_POINTS);

	// Draw out texture
	glPointSize(1.0);
	for (int y = 0; y < screenHeight; y++)
	{
		for (int x = 0; x < screenWidth; x++)
		{
			int ind = y * screenWidth + x;

			glColor3f(fb[ind].x(), fb[ind].y(), fb[ind].z());
			glVertex2f(x, y);
		}
	}
	glEnd();
}

void displayCallback()
{
	glClear(GL_COLOR_BUFFER_BIT);
	draw();
	glFlush();
}

void cleanup()
{
	checkCudaErrors(cudaFree(fb));
}

void initGL(int argc, char **args)
{
	glutInit(&argc, args);
	glutInitDisplayMode(GLUT_RGBA);
	glutInitWindowSize(screenWidth, screenHeight);
	glutInitWindowPosition(10, 10);
	glutCreateWindow("OKNO");

	glMatrixMode(GL_PROJECTION);
	glOrtho(0, screenWidth, 0, screenHeight, -1, 1);
	glMatrixMode(GL_MODELVIEW);

	glClearColor(1.0, 0.0, 1.0, 0.0);

	glutDisplayFunc(displayCallback);
	glutTimerFunc((unsigned int)(1000.0 / TARGET_FPS), refreshFrameCallback, NULL);
	glutTimerFunc(FPS_DISPLAY_REFRESH_TIME, displayFPSCountCallback, NULL);
}

int main(int argc, char** args)
{
	// Create scene
	rayHittable** dObjects;
	checkCudaErrors(cudaMallocManaged((void**)&dObjects, SCENE_ELEMENTS * sizeof(rayHittable*)));
	scene** dScene;
	checkCudaErrors(cudaMallocManaged((void**)&dScene, sizeof(scene)));

	createScene <<<1, 1>>> (dObjects, dScene);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	
	// Create camera
	createCamera << <1, 1 >> > (screenWidth, screenHeight);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// Init values for kernels
	int numPixels = screenWidth * screenHeight;
	size_t fbSize = numPixels * sizeof(vec3);
	checkCudaErrors(cudaMallocManaged(&fb, fbSize));

	int tx = 8;
	int ty = 8;
	dim3 blocks(screenWidth / tx + 1, screenHeight / ty + 1);
	dim3 threads(tx, ty);

	// Setup RNG
	curandState* globalState;
	checkCudaErrors(cudaMallocManaged(&globalState, numPixels * sizeof(curandState)));
	setupRNG << <blocks, threads >> > (globalState, time(NULL), screenWidth);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	
	// Setup render kernel
	initGL(argc, args);

	render <<<blocks, threads >>> (fb, screenWidth, screenHeight, dScene, globalState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	glutMainLoop();

	return 0;
}
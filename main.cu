// Standard libs
#include <iostream>
#include <chrono>

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

#include "fpsCounter.h"

#include "vec3.h"
#include "ray.h"
#include "scene.h"
#include "rayHittable.h"
#include "sphere.h"
#include "doubleUtils.h"
#include "camera.h"
#include "material.h"

static const int SAMPLES_PER_PIXEL = 50;
static const int MAX_SCATTER_DEPTH = 20;

#define BACKGROUND_START_GRADIENT_COLOR vec3(0.5, 0.7, 1.0)
#define BACKGROUND_END_GRADIENT_COLOR vec3(1.0, 1.0, 1.0)

const int screenWidth = 960;
const int screenHeight = 480;

__device__ camera dCam;

std::chrono::steady_clock::time_point start, end;

scene** currentScene;
scene** dSceneAll;
scene** dSceneDiffuse;
scene** dSceneMetalic;
scene** dSceneGlass;
scene** dSceneBig;
vec3* fb;
curandState* globalState;

int tx = 8;
int ty = 8;
dim3 blocks(screenWidth / tx + 1, screenHeight / ty + 1);
dim3 threads(tx, ty);

__device__ vec3 color(const ray& r, scene** dScene, curandState localState)
{
	ray currentRay = r;
	vec3 currentAttenuation = vec3(1.0, 1.0, 1.0);

	// NOTE: Recursion blows up GPU stack, so instead I use iterative recursion
	for (int i = 0; i < MAX_SCATTER_DEPTH; i++) 
	{
		hitInfo hit;
		if ((*dScene)->hit(currentRay, 0.001f, 1000, hit)) 
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
		dCam = camera(vec3(0, 0, 1), vec3(0,0,-1), vec3(0,1,0), 45.0, width/height);
	}
}

__global__ void updateCameraPosition(vec3 translation)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		dCam.updatePosition(translation);
	}
}

__global__ void setupRNG(curandState* globalState, int seed, int screenWidth)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int id = j * screenWidth + i;
	curand_init(seed, id, 0, &globalState[id]);
}

__global__ void createScenes(scene** dSceneAll, scene** dSceneDiffuse, scene** dSceneMetalic, scene** dSceneGlass, scene** dSceneBig)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		// All scene
		rayHittable** objects = (rayHittable**)malloc(8 * sizeof(rayHittable*));
		*(objects) = new sphere(vec3(0, 0, -2), 0.5, new lambertian(vec3(0.8, 0.2, 0.2)));
		*(objects + 1) = new sphere(vec3(-2, -1, -5), 1, new lambertian(vec3(0.0, 0.8, 0.8)));
		*(objects + 2) = new sphere(vec3(0, -100.5, -1), 100, new lambertian(vec3(0.0, 0.8, 0.0)));
		*(objects + 3) = new sphere(vec3(1.5, 0.5, -2), 0.5, new lambertian(vec3(1, 1, 1)));
		*(objects + 4) = new sphere(vec3(-1.5, 0.5, -2), 0.5, new lambertian(vec3(0, 0, 0)));
		*(objects + 5) = new sphere(vec3(1.5, 0, -3), 0.5, new metal(vec3(1, 1, 1), 1));
		*(objects + 6) = new sphere(vec3(-2, 0, -2), 0.5, new metal(vec3(1, 0, 0), 0.5));
		*(objects + 7) = new sphere(vec3(-0.5, 0, -1.5), 0.25, new dielectric(0.9));
		*dSceneAll = new scene(objects, 8);

		// Diffuse scene
		objects = (rayHittable**)malloc(5 * sizeof(rayHittable*));
		*(objects) = new sphere(vec3(0, 0, -1), 0.5, new lambertian(vec3(0.8, 0.2, 0.2)));
		*(objects + 1) = new sphere(vec3(1, 0, -1), 0.5, new lambertian(vec3(0.0, 0.8, 0.8)));
		*(objects + 2) = new sphere(vec3(0, -100.5, -1), 100, new lambertian(vec3(0.0, 0.8, 0.0)));
		*(objects + 3) = new sphere(vec3(-1, 0, -1), 0.5, new lambertian(vec3(1, 1, 1)));
		*(objects + 4) = new sphere(vec3(-2, 1, 0), 0.5, new lambertian(vec3(0, 0, 1)));
		*dSceneDiffuse = new scene(objects, 5);

		// Metalic scene
		objects = (rayHittable**)malloc(5 * sizeof(rayHittable*));
		*(objects) = new sphere(vec3(0, 0, -1), 0.5, new metal(vec3(1, 1, 1), 1));
		*(objects + 1) = new sphere(vec3(1, 0, -1), 0.5, new metal(vec3(0.5, 0, 0.5), 0.5));
		*(objects + 2) = new sphere(vec3(0, -100.5, -1), 100, new lambertian(vec3(0.0, 0.8, 0.0)));
		*(objects + 3) = new sphere(vec3(-1, 0, -1), 0.5, new lambertian(vec3(1, 1, 1)));
		*(objects + 4) = new sphere(vec3(-2, 1, 0), 0.5, new metal(vec3(1, 1, 1), 1));
		*dSceneMetalic = new scene(objects, 5);

		// Glass scene
		objects = (rayHittable**)malloc(6 * sizeof(rayHittable*));
		*(objects) = new sphere(vec3(0, 0, -1), 0.5, new dielectric(0.9));
		*(objects + 1) = new sphere(vec3(1, 0, -1), 0.5, new dielectric(0.1));
		*(objects + 2) = new sphere(vec3(0, -100.5, -1), 100, new lambertian(vec3(0.0, 0.8, 0.0)));
		*(objects + 3) = new sphere(vec3(-1, 0, -1), 0.5, new lambertian(vec3(1, 1, 1)));
		*(objects + 4) = new sphere(vec3(-2, 1, 0), 0.5, new dielectric(5));
		*(objects + 5) = new sphere(vec3(0, 0, -3), 0.5, new lambertian(vec3(0, 0, 1)));
		*dSceneGlass = new scene(objects, 6);

		// Big scene
		objects = (rayHittable**)malloc(300 * sizeof(rayHittable*));
		for (int i = 0; i < 10; i++)
		{
			for (int j = 0; j < 10; j++)
			{
				*(objects + i * 10 + j) = new sphere(vec3((i-4)/2.0, (j - 5)/2.0, -4), 0.1, new lambertian(vec3(i / 5.0, j / 5.0, 0.5)));
				*(objects + i * 10 + j + 100) = new sphere(vec3((i - 4) / 2.0+0.25, (j - 4) / 2.0, -5), 0.2, new metal(vec3((i+j)/100.0, 0.2, 0.5), (10-i + j)/100.0));
				*(objects + i * 10 + j + 200) = new sphere(vec3((i - 4) / 2.0+0.25, (j - 4) +0.5, -6), 0.1, new lambertian(vec3(i / 5.0, j / 5.0, 0.5)));
			}
		}
		*dSceneBig = new scene(objects, 300);
	}
}

__global__ void clearFb(vec3* fb, int maxX)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int pixelIndex = j * maxX + i;
	fb[pixelIndex] = vec3(0.0, 0.0, 0.0);
}

__global__ void render(vec3* fb, int maxX, int maxY, scene** dScene, curandState* globalState)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	
	int pixelIndex = j * maxX + i;

	for (int p = 0; p < SAMPLES_PER_PIXEL; p++)	// TODO: Maybe divide this across threads
	{
		float u = (float(i) + curand_uniform(&globalState[pixelIndex])) / float(maxX);
		float v = (float(j) + curand_uniform(&globalState[pixelIndex])) / float(maxY);
		ray r = dCam.getRay(u, v);
#ifndef USE_GAMMA_CORRECTION
		fb[pixelIndex] += color(r, dScene, globalState[pixelIndex]);
#else
		vec3 clr = color(r, dScene, globalState[pixelIndex]);
		clr.v3sqrt();
		fb[pixelIndex] += clr;
#endif
	}
	fb[pixelIndex] /= SAMPLES_PER_PIXEL;
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

void rerender()
{
	start = std::chrono::high_resolution_clock::now();
	clearFb << <blocks, threads >> > (fb, screenWidth);
	render << <blocks, threads >> > (fb, screenWidth, screenHeight, currentScene, globalState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	end = std::chrono::high_resolution_clock::now();
	std::cout << "Render kernel time:\t\t\t" << (end - start) / std::chrono::milliseconds(1) << "\tms" << std::endl;
	lastRenderTime = (end - start) / std::chrono::milliseconds(1);
}

void glutHandleKeyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'w':
		start = std::chrono::high_resolution_clock::now();
		updateCameraPosition <<<1, 1 >>> (vec3(0, 0, -1));
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		std::cout << "Pressed W" << std::endl;
		rerender();
		break;
	case 's':
		start = std::chrono::high_resolution_clock::now();
		updateCameraPosition << <1, 1 >> > (vec3(0, 0, 1));
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		std::cout << "Pressed S" << std::endl;
		rerender();
		break;
	case 'a':
		start = std::chrono::high_resolution_clock::now();
		updateCameraPosition << <1, 1 >> > (vec3(-1, 0, 0));
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		std::cout << "Pressed A" << std::endl;
		rerender();
		break;
	case 'd':
		start = std::chrono::high_resolution_clock::now();
		updateCameraPosition << <1, 1 >> > (vec3(1, 0, 0));
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		std::cout << "Pressed D" << std::endl;
		rerender();
		break;
	case 'r':
		start = std::chrono::high_resolution_clock::now();
		updateCameraPosition << <1, 1 >> > (vec3(0, 1, 0));
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		std::cout << "Pressed R" << std::endl;
		rerender();
		break;
	case 'f':
		start = std::chrono::high_resolution_clock::now();
		updateCameraPosition << <1, 1 >> > (vec3(0, -1, 0));
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		std::cout << "Pressed F" << std::endl;
		rerender();
		break;
	case '1':
		currentScene = dSceneAll;
		std::cout << "Changing to scene 1 (all)" << std::endl;
		rerender();
		break;
	case '2':
		currentScene = dSceneDiffuse;
		std::cout << "Changing to scene 2 (diffuse)" << std::endl;
		rerender();
		break;
	case '3':
		currentScene = dSceneMetalic;
		std::cout << "Changing to scene 3 (metal)" << std::endl;
		rerender();
		break;
	case '4':
		currentScene = dSceneGlass;
		std::cout << "Changing to scene 4 (glass)" << std::endl;
		rerender();
		break;
	case '5':
		currentScene = dSceneBig;
		std::cout << "Changing to scene 5 (big)" << std::endl;
		rerender();
		break;
	}

}

void initGL(int argc, char **args)
{
	glutInit(&argc, args);
	glutInitDisplayMode(GLUT_RGBA);
	glutInitWindowSize(screenWidth, screenHeight);
	glutInitWindowPosition(10, 10);
	glutCreateWindow("RayTracer");

	glMatrixMode(GL_PROJECTION);
	glOrtho(0, screenWidth, 0, screenHeight, -1, 1);
	glMatrixMode(GL_MODELVIEW);

	glClearColor(1.0, 0.0, 1.0, 0.0);

	glutDisplayFunc(displayCallback);
	glutKeyboardFunc(glutHandleKeyboard);
	glutTimerFunc((unsigned int)(1000.0 / TARGET_FPS), refreshFrameCallback, NULL);
	glutTimerFunc(FPS_DISPLAY_REFRESH_TIME, displayFPSCountCallback, NULL);
}

void writeStartInfo()
{
	std::cout << "\tQuality info:" << std::endl
		<< "\t\tSamples per pixel: " << SAMPLES_PER_PIXEL << std::endl
		<< "\t\tScatter depth: " << MAX_SCATTER_DEPTH << std::endl
		<< "\t\tGamma correction: " <<
#ifdef DIFFUSE_HALF_SPHERE
		"Yes"
#else
		"No"
#endif
		<< std::endl
		<< "\t\tDiffuse half sphere: " <<
#ifdef USE_GAMMA_CORRECTION
		"Yes"
#else
		"No"
#endif
		<< std::endl;
}

int main(int argc, char** args)
{
	int numPixels = screenWidth * screenHeight;

	writeStartInfo();

	// Init GL
	initGL(argc, args);
	std::cout << "GL initialized" << std::endl;

	// CUDA mallocs
	start = std::chrono::high_resolution_clock::now();
	checkCudaErrors(cudaMallocManaged((void**)&dSceneAll, sizeof(scene)));
	checkCudaErrors(cudaMallocManaged((void**)&dSceneDiffuse, sizeof(scene)));
	checkCudaErrors(cudaMallocManaged((void**)&dSceneMetalic, sizeof(scene)));
	checkCudaErrors(cudaMallocManaged((void**)&dSceneGlass, sizeof(scene)));
	checkCudaErrors(cudaMallocManaged((void**)&dSceneBig, sizeof(scene)));
	checkCudaErrors(cudaMallocManaged(&fb, numPixels * sizeof(vec3)));
	checkCudaErrors(cudaMallocManaged(&globalState, numPixels * sizeof(curandState)));
	end = std::chrono::high_resolution_clock::now();
	std::cout << "Cuda mallocs time:\t\t\t" << (end - start) / std::chrono::milliseconds(1) << "\tms" << std::endl;

	// Create scene kernel
	start = std::chrono::high_resolution_clock::now();
	createScenes <<<1, 1>>> (dSceneAll, dSceneDiffuse, dSceneMetalic, dSceneGlass, dSceneBig);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	end = std::chrono::high_resolution_clock::now();
	std::cout << "Create scene kernel time:\t\t" << (end - start) / std::chrono::milliseconds(1) << "\tms" << std::endl;
	
	// Create camera kernel
	start = std::chrono::high_resolution_clock::now();
	createCamera << <1, 1 >> > (screenWidth, screenHeight);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	end = std::chrono::high_resolution_clock::now();
	std::cout << "Create camera kernel time:\t\t" << (end - start) / std::chrono::milliseconds(1) << "\tms" << std::endl;

	// Setup RNG kernel
	start = std::chrono::high_resolution_clock::now();
	setupRNG << <blocks, threads >> > (globalState, time(NULL), screenWidth);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	end = std::chrono::high_resolution_clock::now();
	std::cout << "RNG states init kernel time:\t\t" << (end - start) / std::chrono::milliseconds(1) << "\tms" << std::endl;

	// Render kernel
	currentScene = dSceneDiffuse;
	std::cout << "Changing to scene 2 (diffuse)" << std::endl;
	rerender();

	glutMainLoop();

	return 0;
}
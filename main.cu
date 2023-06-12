// Standard libs
#include <iostream>
#include <memory>

// GL libs
#include <GL/glut.h>

// CUDA libs
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// My libs
//#include "cudaHelpers.h"
// TMP
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		// Print message
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n" << cudaGetErrorString(result) << "\n";
		// Reset GPU
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

// Wrapper define
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
// !TMP

#include "vec3.h"
#include "ray.h"
#include "scene.h"
#include "rayHittable.h"
#include "sphere.h"

static const int TARGET_FPS = 60;
static const unsigned int FPS_DISPLAY_REFRESH_TIME = 500;

const int screenWidth = 960;
const int screenHeight = 480;

#define ASPECT_RATIO (float)screenWidth / screenHeight

__global__ void createScene(rayHittable** dObjects, scene** dScene)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		*(dObjects) = new sphere(vec3(0, 0, -2), 0.5);
		*(dObjects + 1) = new sphere(vec3(-2, -1, -5), 1);
		*dScene = new scene(dObjects, 2);
	}
}

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

__device__ vec3 color(const ray& r, scene** dScene)
{
	hitInfo hit;
	if ((*dScene)->hit(r, 0, 100, hit))
	{
		return 0.5 * (hit.normal + vec3(1.0, 1.0, 1.0));
	}

	vec3 dir = unit_vector(r.direction());
	float t = 0.5 * (dir.y() + 1.0);
	return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

__global__ void render(vec3* fb, int maxX, int maxY, vec3 lowerLeftCorner, vec3 horizontal, vec3 vertical, vec3 origin, scene** dScene)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= maxX) || (j >= maxY)) return;

	int pixelIndex = j * maxX + i;
	float u = float(i) / float(maxX);
	float v = float(j) / float(maxY);
	ray r(origin, lowerLeftCorner + u * horizontal + v * vertical);
	fb[pixelIndex] = color(r, dScene);
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
	int numPixels = screenWidth * screenHeight;
	size_t fbSize = numPixels * sizeof(vec3);
	checkCudaErrors(cudaMallocManaged(&fb, fbSize));

	rayHittable** dObjects;
	checkCudaErrors(cudaMallocManaged((void**)&dObjects, 2 * sizeof(rayHittable*)));
	scene** dScene;
	checkCudaErrors(cudaMallocManaged((void**)&dScene, sizeof(scene)));

	createScene <<<1, 1>>> (dObjects, dScene);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	int tx = 8;
	int ty = 8;
	dim3 blocks(screenWidth / tx + 1, screenHeight / ty + 1);
	dim3 threads(tx, ty);

	initGL(argc, args);

	render <<<blocks, threads >>> (fb, screenWidth, screenHeight, vec3(-2.0, -1.0, -1.0), vec3(2 * ASPECT_RATIO, 0.0, 0.0), vec3(0.0, 2.0, 0.0), vec3(0.0, 0.0, 0.0), dScene);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	glutMainLoop();

	return 0;
}
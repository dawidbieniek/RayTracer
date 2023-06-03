// Standard libs
#include <iostream>

// GL libs
#include <GL/glut.h>

// CUDA libs
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// My libs
//#include "cudaHelpers.h"
// TMP
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
// !TMP
#include "vec3.h"
#include "ray.h"

int screenWidth = 960;
int screenHeight = 480;

__device__ bool hit_sphere(const vec3& center, float radius, const ray& r) 
{
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = 2.0f * dot(oc, r.direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - 4.0f * a * c;
	return (discriminant > 0.0f);
}

__device__ vec3 color(const ray& r) 
{
	if (hit_sphere(vec3(0, 0, -1), 0.5, r))
		return vec3(1, 0, 0);
	// Tlo
	vec3 unitDirection = unit_vector(r.direction());
	float t = 0.5f * (unitDirection.y() + 1.0f);
	return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

__global__ void render(vec3* fb, int maxX, int maxY, vec3 lowerLeftCorner, vec3 horizontal, vec3 vertical, vec3 origin)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= maxX) || (j >= maxY)) return;
	int pixel_index = j * maxX + i;
	float u = float(i) / float(maxX);
	float v = float(j) / float(maxY);
	ray r(origin, lowerLeftCorner + u * horizontal + v * vertical);
	fb[pixel_index] = color(r);
}

vec3* fb;

void draw()
{
	glBegin(GL_POINTS);

	glPointSize(1.0);
	for (int y = 0; y < screenHeight; y++)
	{
		for (int x = 0; x < screenWidth; x++)
		{
			int ind = y * screenWidth + x;

			glColor3f(fb[ind].r(), fb[ind].g(), fb[ind].b());
			glVertex2f(x, y);
		}
	}

	glEnd();
}


void display()
{
	glClear(GL_COLOR_BUFFER_BIT);
	draw();
	glFlush();
}

void cleanup()
{
	checkCudaErrors(cudaFree(fb));
}

int main(int argc, char** argv)
{
	// Example
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Good
	int numPixels = screenWidth * screenHeight;
	size_t fbSize = numPixels * sizeof(vec3);
	checkCudaErrors(cudaMallocManaged(&fb, fbSize));

	int tx = 8;
	int ty = 8;
	dim3 blocks(screenWidth / tx + 1, screenHeight / ty + 1);
	dim3 threads(tx, ty);

	render <<<blocks, threads >>> (fb, screenWidth, screenHeight, vec3(-2.0, -1.0, -1.0), vec3(4.0, 0.0, 0.0), vec3(0.0, 2.0, 0.0), vec3(0.0, 0.0, 0.0));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// GL Init
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA);
	glutInitWindowSize(screenWidth, screenHeight);
	glutInitWindowPosition(10, 10);
	glutCreateWindow("OKNO");

	glMatrixMode(GL_PROJECTION);
	glOrtho(0, screenWidth, 0, screenHeight, -1, 1);
	glMatrixMode(GL_MODELVIEW);

	glClearColor(1.0, 0.0, 1.0, 0.0);
	
	glutDisplayFunc(display);
	glutMainLoop();

	//cleanup();
	//std::cout << "End of program" << std::endl;

	//// Add vectors in parallel.
	//cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "addWithCuda failed!");
	//	return 1;
	//}

	//printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
	//	c[0], c[1], c[2], c[3], c[4]);

	//// cudaDeviceReset must be called before exiting in order for profiling and
	//// tracing tools such as Nsight and Visual Profiler to show complete traces.
	//cudaStatus = cudaDeviceReset();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaDeviceReset failed!");
	//	return 1;
	//}

	return 0;
}

#ifndef FPSCOUNTER_H
#define FPSCOUNTER_H

#include <stdio.h>
#include <GL/glut.h>

static const int TARGET_FPS = 60;
static const unsigned int FPS_DISPLAY_REFRESH_TIME = 500;

int lastFrameTime = 0;
int deltaTimes[TARGET_FPS];
int deltaTimesIndex = 0;

long long lastRenderTime;

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

		fpsCount = 1000.0 / (fpsCount / i + 1);

		char titleBuffer[32];
		sprintf(titleBuffer, "FPS: %3.1f | %lldms", fpsCount, lastRenderTime);
		glutSetWindowTitle(titleBuffer);

		// Refresh callback
		glutTimerFunc(FPS_DISPLAY_REFRESH_TIME, displayFPSCountCallback, NULL);
	}
}

#endif // FPSCOUNTER_H
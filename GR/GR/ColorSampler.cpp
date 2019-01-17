#include "ColorSampler.h"

ColorSampler::ColorSampler()
{
	averageColor = Vec3b(0, 0, 0);
	upperBoundColor = Vec3b(0, 0, 0);
	lowerBoundColor = Vec3b(255, 255, 255);

	bCaptureColor = false;
}

ColorSampler::ColorSampler(const Rect& inRect, int inMinWidth)
	:samplingRect(inRect),
	minWidth(inMinWidth)
{
	averageColor = Vec3b(0, 0, 0);
	upperBoundColor = Vec3b(0, 0, 0);
	lowerBoundColor = Vec3b(255, 255, 255);

	bCaptureColor = false;
}

void ColorSampler::examineColor(const Mat& inMat)
{
	averageColor = Vec3b(0, 0, 0);
	upperBoundColor = Vec3b(0, 0, 0);
	lowerBoundColor = Vec3b(255, 255, 255);

	int x = samplingRect.x;
	int y = samplingRect.y;
	int rows = samplingRect.height;
	int cols = samplingRect.width;
	int numElements = rows * cols;

	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			auto& color = inMat.at<Vec3b>(y + row, x + col);

			upperBoundColor = Vec3b(MAX(upperBoundColor[0], color[0]), MAX(upperBoundColor[1], color[1]), MAX(upperBoundColor[2], color[2]));
			lowerBoundColor = Vec3b(MIN(lowerBoundColor[0], color[0]), MIN(lowerBoundColor[1], color[1]), MIN(lowerBoundColor[2], color[2]));
			averageColor += color / numElements;
		}
	}
}

void ColorSampler::moveBy(int dx, int dy)
{
	samplingRect.x += dx;
	samplingRect.y += dy;
}

void ColorSampler::moveTo(int x, int y)
{
	samplingRect.x = x;
	samplingRect.y = y;
}

void ColorSampler::resizeBy(int dw, int dh)
{
	samplingRect.width += dw;
	samplingRect.height += dh;

	samplingRect.width = MAX(samplingRect.width, minWidth);
	samplingRect.height = MAX(samplingRect.height, minWidth);
}

void ColorSampler::resizeTo(int w, int h)
{
	samplingRect.width = w;
	samplingRect.height = h;

	samplingRect.width = MAX(samplingRect.width, minWidth);
	samplingRect.height = MAX(samplingRect.height, minWidth);
}
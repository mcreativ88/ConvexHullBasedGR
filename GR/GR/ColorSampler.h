#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;

class ColorSampler
{
public:
	ColorSampler();
	ColorSampler(const Rect& inRect, int inMinWidth);

	void examineColor(const Mat& inMat);
	void moveBy(int dx, int dy);
	void moveTo(int x, int y);
	void resizeBy(int dw, int dh);
	void resizeTo(int w, int h);
	
	const Rect& GetSampleRect() const
	{
		return samplingRect;
	}

	int GetSampleArea() const
	{
		return samplingRect.area();
	}

	int GetSampleRows() const
	{
		return samplingRect.height;
	}

	int GetSampleColumns() const
	{
		return samplingRect.width;
	}

	Vec3f GetAverageColor() const
	{
		return averageColor;
	}

	Vec3f GetUpperBoundColor() const
	{
		return upperBoundColor;
	}

	Vec3f GetLowerBoundColor() const
	{
		return lowerBoundColor;
	}

public:
	bool bCaptureColor = false;

private:
	Rect samplingRect;
	Vec3b averageColor;
	Vec3b upperBoundColor;
	Vec3b lowerBoundColor;
	int minWidth = 0;
};
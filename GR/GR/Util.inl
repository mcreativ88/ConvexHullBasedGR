#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;

inline void reverseColumns(Mat& inOutFrame)
{
	for (int i = 0; i < inOutFrame.cols / 2.0f; i++)
	{
		int leftIndex = i;
		int rightIndex = inOutFrame.cols - 1 - i;

		if (leftIndex == rightIndex)
		{
			break;
		}

		Mat left = inOutFrame.col(leftIndex);
		Mat right = inOutFrame.col(rightIndex);
		Mat temp;
		left.copyTo(temp);
		right.copyTo(left);
		temp.copyTo(right);
	}
}

inline Point getRectMid(const Rect& inRect)
{
	Point mid;
	mid.x = inRect.x + (int)(0.5f*inRect.width);
	mid.y = inRect.y + (int)(0.5f*inRect.height);

	return mid;
}

inline Rect mergeRect(const Rect& inRect1, const Rect& inRect2)
{
	Point newMid = 0.5f*getRectMid(inRect1) + 0.5f*getRectMid(inRect2);

	int x = MIN(inRect1.x, inRect2.x);
	int y = MIN(inRect1.y, inRect2.y);
	int outerX = MAX(inRect1.x + inRect1.width, inRect2.x + inRect2.width);
	int outerY = MAX(inRect1.y + inRect1.height, inRect2.y + inRect2.height);
	int width = outerX - x;
	int height = outerY - y;
	Rect mergedRect(x, y, width, height);

	return mergedRect;
}

inline bool isPointInRect(const Rect& r, const Point& p, float margin = 0)
{
	return
		p.x >= r.x - margin && p.x <= r.x + r.width + margin &&
		p.y >= r.y - margin && p.y <= r.y + r.height + margin;
}


inline bool isInMergeBound(const Rect& inHost, const Rect& inTarget, float inMinDistance, float inMargin = 0)
{
	int hostArea = inHost.area();
	int targetArea = inTarget.area();

	Point hostMid = getRectMid(inHost);
	Point targetMid = getRectMid(inTarget);

	bool bInBound = false;

	// 포함
	if (hostArea >= targetArea)
	{
		Point targetP0(inTarget.x, inTarget.y);
		Point targetP1(inTarget.x + inTarget.width, inTarget.y);
		Point targetP2(inTarget.x + inTarget.width, inTarget.y + inTarget.height);
		Point targetP3(inTarget.x, inTarget.y + inTarget.height);

		bInBound = isPointInRect(inHost, targetP0, inMargin) || isPointInRect(inHost, targetP1, inMargin) || isPointInRect(inHost, targetP2, inMargin) || isPointInRect(inHost, targetP3, inMargin);
	}
	else
	{
		Point hostP0(inHost.x, inHost.y);
		Point hostP1(inHost.x + inHost.width, inHost.y);
		Point hostP2(inHost.x + inHost.width, inHost.y + inHost.height);
		Point hostP3(inHost.x, inHost.y + inHost.height);

		bInBound = isPointInRect(inTarget, hostP0, inMargin) || isPointInRect(inTarget, hostP1, inMargin) || isPointInRect(inTarget, hostP2, inMargin) || isPointInRect(inTarget, hostP3, inMargin);
	}

	// 최소거리 이내
	if (norm(targetMid - hostMid) <= inMinDistance)
	{
		bInBound = true;
	}

	return bInBound;
}
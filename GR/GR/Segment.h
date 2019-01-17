
#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;

class Segment
{
public:
	Segment(const Rect& inRect);

public:
	// 세그먼트 영역
	Rect rect;

	// 세그먼트 중심
	Point mid;

	// 트래킹 마커
	bool bTracked = false;

	// 현재 유효한 제스쳐에 포함된 세그먼트 마킹
	bool bActive = false;
};

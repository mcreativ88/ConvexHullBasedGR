
#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;

class Segment
{
public:
	Segment(const Rect& inRect);

public:
	// ���׸�Ʈ ����
	Rect rect;

	// ���׸�Ʈ �߽�
	Point mid;

	// Ʈ��ŷ ��Ŀ
	bool bTracked = false;

	// ���� ��ȿ�� �����Ŀ� ���Ե� ���׸�Ʈ ��ŷ
	bool bActive = false;
};

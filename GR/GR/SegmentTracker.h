#pragma once
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "Segment.h"

using namespace std;
using namespace cv;

class SegmentTracker
{
public:
	SegmentTracker(int n);

	void preupdate();
	void addToHistory(Segment& inSegment);
	Point getNLatestSegmentMid(int n);
	Point getNLatestInactiveSegmentMid(int n);

public:
	int deactivateCounter = 0;
	int numTrackingFrames = 0;
	vector<Segment> history;

	bool bActive = false;
	bool bTracking = false;
	bool bUpdated = false;

	Point avgMid;
	Point avgInactiveSegmentMid;
};
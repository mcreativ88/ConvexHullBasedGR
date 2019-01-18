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

private:
	void trackSegment(Segment& inSegmnet);
	void addToHistory(Segment& inSegment);
	Point getNLatestSegmentMid(int n);
	Point getNLatestInactiveSegmentMid(int n);

private:
	int deactivateCounter = 0;
	int numTrackingFrames = 0;
	vector<Segment> history;

	// 이번 프레임에 업데이트 되었는가
	bool bUpdated = false;

	// 추적중인가
	bool bTracking = false;

	// 추적중인 세그먼트가 이동중인가
	bool bActive = false;

private:
	friend class TrackerManager;
};

class TrackerManager
{
public:
	TrackerManager(int inNumTrackingTime, int inDeactivationTime);
	
	void update(vector<Segment>& inSegments);
	void visualize(Mat& inMat);

private:
	void prepare();
	void updateActiveTrackers();
	void updateInactiveTrackers();
	void assignSegments();
	void removeUnusedTrackers();

private:
	vector<Segment>* segments = nullptr;
	vector<SegmentTracker> segmentTrackers;
	int numTrackingFrames = 0;
	int numDeactivationFrames = 0;
};
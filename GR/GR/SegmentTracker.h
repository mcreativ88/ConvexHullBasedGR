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

	// �̹� �����ӿ� ������Ʈ �Ǿ��°�
	bool bUpdated = false;

	// �������ΰ�
	bool bTracking = false;

	// �������� ���׸�Ʈ�� �̵����ΰ�
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
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include "Util.inl"
#include "ColorSampler.h"
#include "SegmentTracker.h"

using namespace cv;
using namespace std;

#define TARGET_FPS				60
#define FIXED_DT				((int)(1000.0/TARGET_FPS))
#define NUM_DEACTIVATION_FRAMES	((int)(0.4*TARGET_FPS))
#define NUM_TRACKING_FRAMES		120

void extractSegments(vector<Segment>& outSegments, const Mat& inMat, const ColorSampler& inSpoid);

int main(int, char**)
{
	VideoCapture capture(0);
	if (!capture.isOpened())
	{
		cout << "Camera open failed.\n";
		return -1;
	}

	// 화면 크기
	int vWidth = (int)capture.get(CAP_PROP_FRAME_WIDTH);
	int vHeight = (int)capture.get(CAP_PROP_FRAME_HEIGHT);

	// 색상 추출기
	ColorSampler spoid;
	{
		int minSpoidWidth = (int)(0.05f*vWidth);
		int x = (int)(0.75f*vWidth);
		int y = (int)(0.5f*vHeight);
		int columns = minSpoidWidth;
		int rows = minSpoidWidth;

		spoid = ColorSampler(Rect(x, y, columns, rows), minSpoidWidth);
	}

	// 샘플 영역 색상 범위
	Vec3b minYcc(0, 0, 0);
	Vec3b maxYcc(0, 0, 0);
	Vec3b minHsv(0, 0, 0);
	Vec3b maxHsv(0, 0, 0);

	// 세그먼트 병합 최소 거리
	float minMergeDistance = 0;

	// 세그먼트 트래커
	vector<SegmentTracker> segmentTrackers;
	TrackerManager trackerManager(NUM_TRACKING_FRAMES, NUM_DEACTIVATION_FRAMES);

	// 매트릭스
	Mat input;
	Mat result;
	Mat hsv;
	Mat yCrCv;
	Mat hsvBinary = Mat(vHeight, vWidth, CV_8U);
	Mat yCrCvBinary = Mat(vHeight, vWidth, CV_8U);
	Mat combinedBinary = Mat(vHeight, vWidth, CV_8U);

	while (capture.read(input))
	{
		if (input.empty())
		{
			cerr << "ERROR! blank frame grabbed\n";
			break;
		}

		// 화면 좌우 반전
		reverseColumns(input);

		input.copyTo(result);

		// 색상 변환
		cvtColor(input, hsv, COLOR_BGR2HSV);
		cvtColor(input, yCrCv, COLOR_BGR2YCrCb);

		// 샘플 색상 추출
		if (spoid.bCaptureColor)
		{
			Rect spoidRect = spoid.GetSampleRect();
			rectangle(result, spoidRect, Scalar(0, 0, 255), 2);

			spoid.examineColor(hsv);
			maxHsv = spoid.GetUpperBoundColor();
			minHsv = spoid.GetLowerBoundColor();

			spoid.examineColor(yCrCv);
			maxYcc = spoid.GetUpperBoundColor();
			minYcc = spoid.GetLowerBoundColor();
		}

		// hsv 마스크 생성
		{
			float margin = 14;
			inRange(hsv, Scalar(minHsv[0], minHsv[1] - 0.2f*margin, 0), Scalar(maxHsv[0], maxHsv[1] + margin, 255), hsvBinary);

			// 블러 처리
			blur(hsvBinary, hsvBinary, Size(15, 15));
			threshold(hsvBinary, hsvBinary, 130, 255, THRESH_BINARY);
			blur(hsvBinary, hsvBinary, Size(10, 10));
			threshold(hsvBinary, hsvBinary, 150, 255, THRESH_BINARY);
		}

		// yCrCv 마스크 생성
		{
			float margin = 3;
			inRange(yCrCv, Scalar(0, minYcc[1] - margin, minYcc[2] - margin), Scalar(255, maxYcc[1] + margin, maxYcc[2] + margin), yCrCvBinary);
			 
			// 블러 처리
			blur(yCrCvBinary, yCrCvBinary, Size(10, 10));
			threshold(yCrCvBinary, yCrCvBinary, 130, 255, THRESH_BINARY);
		}

		// hsv * yCrCv
		{
			for (int row = 0; row < yCrCv.rows; row++)
			{
				for (int col = 0; col < yCrCv.cols; col++)
				{
					auto& h = hsvBinary.at<uchar>(row, col);
					auto& y = yCrCvBinary.at<uchar>(row, col);
					auto& c = combinedBinary.at<uchar>(row, col);

					c = MIN(h * y, 255);
				}
			}

			blur(combinedBinary, combinedBinary, Size(15, 15));
			threshold(combinedBinary, combinedBinary, 130, 255, THRESH_BINARY);
		}

		// 세그먼트 추출
		vector<Segment> segments;
		extractSegments(segments, combinedBinary, spoid);

		// 트래커 업데이트
		trackerManager.update(segments);

		// 시각화
		trackerManager.visualize(result);

		// 매트릭스 출력
		imshow("result", result);
		//imshow("hsv", hsvBinary);
		//imshow("yCrCv", yCrCvBinary);
		imshow("combinedBinary", combinedBinary);
		
		int key = waitKey(FIXED_DT);
		switch (key)
		{
		case 27:
			return 0;
		case '1':
			spoid.bCaptureColor = !spoid.bCaptureColor;
			break;
		}

		if (spoid.bCaptureColor)
		{
			switch (key)
			{
			// 위치 이동
			case 'a':
				spoid.moveBy(-10, 0);
				break;
			case 'd':
				spoid.moveBy(10, 0);
				break;
			case 'w':
				spoid.moveBy(0, -10);
				break;
			case 's':
				spoid.moveBy(0, 10);
				break;

			// 크기조절
			case 'q':
				spoid.resizeBy(-10, 0);
				break;
			case 'e':
				spoid.resizeBy(10, 0);
				break;
			case 'z':
				spoid.resizeBy(0, -10);
				break;
			case 'c':
				spoid.resizeBy(0, 10);
				break;
			}
		}
	}

	return 0;
}

void extractSegments(vector<Segment>& outSegments, const Mat& inMat, const ColorSampler& inSpoid)
{
	vector<vector<Point>> contours;
	vector<Rect> candidates;

	findContours(inMat, contours, RetrievalModes::RETR_EXTERNAL, ContourApproximationModes::CHAIN_APPROX_NONE);

	for (size_t c_i = 0; c_i < contours.size(); c_i++)
	{
		vector<Point> hullPoints;
		convexHull(contours[c_i], hullPoints, true);

		Point min(INT32_MAX, INT32_MAX);
		Point max(0, 0);
		for (size_t p_i = 0; p_i < hullPoints.size(); p_i++)
		{
			min.x = MIN(min.x, hullPoints[p_i].x);
			min.y = MIN(min.y, hullPoints[p_i].y);
			max.x = MAX(max.x, hullPoints[p_i].x);
			max.y = MAX(max.y, hullPoints[p_i].y);
		}

		candidates.push_back(Rect2f(min, max));
	}

	if (candidates.size() > 0)
	{
		// 스포이드 크기 이하는 제거
		for (size_t c_i = 0; c_i < candidates.size(); c_i++)
		{
			if (candidates[c_i].area() < inSpoid.GetSampleArea())
			{
				candidates.erase(candidates.begin() + c_i);
				c_i--;
			}
		}

		// 병합 
		float mergingMinDst = sqrt(4.0f*inSpoid.GetSampleArea());
		float mergingMargin = sqrt(1.5f*inSpoid.GetSampleArea());
		for (size_t c_i = 0; c_i < candidates.size(); c_i++)
		{
			auto& candidate = candidates[c_i];

			if (outSegments.size() == 0)
			{
				outSegments.push_back(Segment(candidates[0]));
				continue;
			}

			bool bMerged = false;
			for (size_t c_j = 0; c_j < outSegments.size(); c_j++)
			{
				auto& segment = outSegments[c_j];

				if (isInMergeBound(segment.rect, candidate, mergingMinDst, mergingMargin))
				{
					segment.rect = mergeRect(segment.rect, candidate);
					bMerged = true;
					break;
				}
			}

			if (!bMerged)
			{
				outSegments.push_back(Segment(candidate));
			}
		}
	}
}
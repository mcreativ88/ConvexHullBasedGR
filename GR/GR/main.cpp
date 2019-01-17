#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include "Util.inl"
#include "SegmentTracker.h"

using namespace cv;
using namespace std;

#define TARGET_FPS			60
#define FIXED_DT			(1000.0/TARGET_FPS)
#define DEACTIVATION_FRAME	(0.4*TARGET_FPS)

int main(int, char**)
{
	VideoCapture capture(0);
	if (!capture.isOpened())
	{
		cout << "Camera open failed.\n";
		return -1;
	}

	int vWidth = (int)capture.get(CAP_PROP_FRAME_WIDTH);
	int vHeight = (int)capture.get(CAP_PROP_FRAME_HEIGHT);

	// 샘플 색상 추출
	bool bCaptureSampleColor = false;
	const int MIN_SPOID_WIDTH = (int)(0.05f*vWidth);
	const int MIN_SPOID_HEIGHT = (int)(0.05f*vWidth);
	int SpoidColumns = MIN_SPOID_WIDTH;
	int SpoidRows = MIN_SPOID_HEIGHT;
	int SpoidX = (int)(0.75f*vWidth);
	int SpoidY = (int)(0.5f*vHeight);
	Vec3f SampleColor(0, 0, 0);
	
	float sampleSegmentArea = SpoidRows*SpoidColumns;
	float criticalDistance = (float)sqrt(sampleSegmentArea);

	// 샘플 yCrCb 색상 범위
	uchar MinCr = 255;
	uchar MaxCr = 0;
	uchar MinCb = 255;
	uchar MaxCb = 0;

	// 샘플 hsv 색상 범위
	uchar MinH = 255;
	uchar MaxH = 0;
	uchar MinS = 255;
	uchar MaxS = 0;

	// 컨벡스
	Mat input;
	Mat workspace;
	Mat result;
	Mat Spoid(SpoidRows, SpoidColumns, CV_8UC3);
	Mat hsv;
	Mat yCrCv;
	Mat hsvBinary = Mat(vHeight, vWidth, CV_8U);
	Mat yCrCvBinary = Mat(vHeight, vWidth, CV_8U);
	Mat combinedBinary = Mat(vHeight, vWidth, CV_8U);
	
	// 세그먼트 트래커
	int numTrackingFrame = 120;
	vector<SegmentTracker> segmentTrackers;

	while (capture.read(input))
	{
		if (input.empty())
		{
			cerr << "ERROR! blank frame grabbed\n";
			break;
		}

		reverseColumns(input);

		input.copyTo(workspace);
		input.copyTo(result);

		// 색상 변환
		cvtColor(workspace, hsv, COLOR_BGR2HSV);
		cvtColor(workspace, yCrCv, COLOR_BGR2YCrCb);

		// 샘플 색상 추출
		if (bCaptureSampleColor)
		{
			Rect SpoidRect = Rect(SpoidX, SpoidY, SpoidColumns, SpoidRows);
			rectangle(result, SpoidRect, Scalar(0, 0, 255), 2);

			MinCr = 255;
			MaxCr = 0;
			MinCb = 255;
			MaxCb = 0;

			MinH = 255;
			MaxH = 0;
			MinS = 255;
			MaxS = 0;

			SampleColor.zeros();
			for (int row = 0; row < SpoidRows; row++)
			{
				for (int col = 0; col < SpoidColumns; col++)
				{
					auto& hsvColor = hsv.at<Vec3b>(SpoidY + row, SpoidX + col);
					auto& yCrCvColor = yCrCv.at<Vec3b>(SpoidY + row, SpoidX + col);
					Spoid.at<Vec3b>(row, col) = yCrCvColor;

					MinCr = MIN(MinCr, yCrCvColor[1]);
					MinCb = MIN(MinCb, yCrCvColor[2]);
					MaxCr = MAX(MaxCr, yCrCvColor[1]);
					MaxCb = MAX(MaxCb, yCrCvColor[2]);

					MinH = MIN(MinH, hsvColor[0]);
					MinS = MIN(MinS, hsvColor[1]);
					MaxH = MAX(MaxH, hsvColor[0]);
					MaxS = MAX(MaxS, hsvColor[1]);
				}
			}
		}

		// hsv 마스크 생성
		{
			float margin = 14;
			inRange(hsv, Scalar(MinH, MinS - 0.2f*margin, 0), Scalar(MaxH, MaxS + margin, 255), hsvBinary);

			blur(hsvBinary, hsvBinary, Size(15, 15));
			threshold(hsvBinary, hsvBinary, 130, 255, THRESH_BINARY);
			blur(hsvBinary, hsvBinary, Size(10, 10));
			threshold(hsvBinary, hsvBinary, 150, 255, THRESH_BINARY);
		}

		// yCrCv 마스크 생성
		{
			float margin = 3;
			inRange(yCrCv, Scalar(0, MinCr - margin, MinCb - margin), Scalar(255, MaxCr + margin, MaxCb + margin), yCrCvBinary);

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

		// 컨벡스 헐 추출
		vector<Segment> segments;
		{
			vector<vector<Point>> contours;
			vector<Rect> candidates;

			findContours(combinedBinary, contours, RetrievalModes::RETR_EXTERNAL, ContourApproximationModes::CHAIN_APPROX_NONE);
			//findContours(yCrCvBinary, contours, RetrievalModes::RETR_EXTERNAL, ContourApproximationModes::CHAIN_APPROX_NONE);
			//findContours(hsvBinary, contours, RetrievalModes::RETR_EXTERNAL, ContourApproximationModes::CHAIN_APPROX_NONE);

			for (size_t c_i = 0; c_i < contours.size(); c_i++)
			{
				vector<Point> hullPoints;
				convexHull(contours[c_i], hullPoints, true);

				Point min(vWidth, vHeight);
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
				// 스포이드 크기보다 작은건 제거
				for (size_t c_i = 0; c_i < candidates.size(); c_i++)
				{
					Rect SpoidRect = Rect(SpoidX, SpoidY, SpoidColumns, SpoidRows);
					if (candidates[c_i].area() < SpoidRect.area())
					{
						candidates.erase(candidates.begin() + c_i);
						c_i--;
					}
				}

				// 타겟 세그먼트 크기 저장
				if (bCaptureSampleColor)
				{
					for (size_t c_i = 0; c_i < candidates.size(); c_i++)
					{
						auto& candidate = candidates[c_i];

						if (isPointInRect(candidate, Point(SpoidX, SpoidY)))
						{
							sampleSegmentArea = (float)candidate.area();
							criticalDistance = (float)sqrt(sampleSegmentArea);
							break;
						}
					}
				}

				// 병합 
				int mergingMargin = MAX(SpoidRows, SpoidColumns);
				for (size_t c_i = 0; c_i < candidates.size(); c_i++)
				{
					auto& candidate = candidates[c_i];

					if (segments.size() == 0)
					{
						segments.push_back(Segment(candidates[0]));
						continue;
					}

					bool bMerged = false;
					for (size_t c_j = 0; c_j < segments.size(); c_j++)
					{
						auto& segment = segments[c_j];

						if (isInMergeBound(segment.rect, candidate, 0.8f*criticalDistance, (float)MAX(SpoidRows, SpoidColumns)))
						{
							segment.rect = mergeRect(segment.rect, candidate);
							bMerged = true;
							break;
						}
					}

					if (!bMerged)
					{
						segments.push_back(Segment(candidate));
					}
				}
			}
		}

		// 트래커 업데이트 //

		// 트래커 추가
		while (segmentTrackers.size() < segments.size())
		{
			segmentTrackers.push_back(SegmentTracker(120));
		}

		// 업데이트 준비
		for (size_t c_i = 0; c_i < segmentTrackers.size(); c_i++)
		{
			auto& tracker = segmentTrackers[c_i];
			tracker.preupdate();
			tracker.bUpdated = false;
		}

		// 업데이트
		if (segmentTrackers.size() > 0)
		{
			// 활성화 상태 트래커 업데이트
			for (size_t c_i = 0; c_i < segmentTrackers.size(); c_i++)
			{
				auto& tracker = segmentTrackers[c_i];

				// 다음 세그먼트 추적
				if (!tracker.bUpdated && tracker.bTracking && tracker.bActive)
				{
					auto prevSegment = tracker.history.front();

					// 가장 가까운 세그먼트를 추적
					float minDistance = MAX(vWidth, vHeight);
					int minIndex = -1;
					for (size_t c_j = 0; c_j < segments.size(); c_j++)
					{
						auto& segment = segments[c_j];
						if (!segment.bTracked && isPointInRect(prevSegment.rect, getRectMid(segment.rect), 0.3f*(float)sqrt(prevSegment.rect.area())))
						{
							float distance = (float)norm(segment.mid - prevSegment.mid);
							if (minDistance > distance)
							{
								minDistance = distance;
								minIndex = c_j;
							}
						}
					}

					if (minIndex >= 0)
					{
						auto& nearestSegment = segments[minIndex];
			
						// 일정 거리 이내일 경우 비활성화 상태로 변경
						float threshold = 0.1f*(float)sqrt(prevSegment.rect.area());
						if (minDistance <= threshold)
						{
							if (++tracker.deactivateCounter > DEACTIVATION_FRAME)
							{
								tracker.bActive = false;
								tracker.history.clear();
								tracker.deactivateCounter = 0;
							}
						}
						else
						{
							tracker.deactivateCounter = 0;
							nearestSegment.bActive = true;
						}

						// 히스토리엔 위치를 보간 후 보관
						auto copy = nearestSegment;
						copy.mid = 0.6f*copy.mid + 0.4f*prevSegment.mid;

						tracker.addToHistory(copy);
						tracker.bUpdated = true;
						nearestSegment.bTracked = true;
					}
				}

			}

			// 비활성화 상태 트래커 업데이트
			for (size_t c_i = 0; c_i < segmentTrackers.size(); c_i++)
			{
				auto& tracker = segmentTrackers[c_i];
				
				// 다음 세그먼트 추적
				if (tracker.bTracking && !tracker.bActive)
				{
					auto prevSegment = tracker.history.front();

					// 가장 가까운 세그먼트를 추적
					float minDistance = (float)MAX(vWidth, vHeight);
					int minIndex = -1;
					for (size_t c_j = 0; c_j < segments.size(); c_j++)
					{
						auto& segment = segments[c_j];
						if (!segment.bTracked && isPointInRect(prevSegment.rect, getRectMid(segment.rect), 0.3f*(float)sqrt(prevSegment.rect.area())))
						{
							float distance = (float)norm(segment.mid - tracker.getNLatestSegmentMid(10));
							if (minDistance > distance)
							{
								minDistance = distance;
								minIndex = c_j;
							}
						}
					}

					if (minIndex >= 0)
					{
						auto& nearestSegment = segments[minIndex];

						// 일정 거리 이내일 경우 계속 비활성화 상태로 간주
						float threshold = 0.3f*(float)sqrt(prevSegment.rect.area());
						if (minDistance <= threshold)
						{
							//tracker.history.clear();
						}

						// 일정 거리를 벗어나면 활성화 상태로 변경
						else
						{
							tracker.history.clear();
							tracker.bActive = true;
							nearestSegment.bActive = true;
						}

						// 히스토리엔 위치를 보간 후 보관
						auto copy = nearestSegment;
						copy.mid = 0.6f*copy.mid + 0.4f*prevSegment.mid;

						tracker.addToHistory(nearestSegment);
						tracker.bUpdated = true;
						nearestSegment.bTracked = true;
					}
				}
			}

			// 트래커에 세그먼트 할당
			for (size_t c_i = 0; c_i < segmentTrackers.size(); c_i++)
			{
				auto& tracker = segmentTrackers[c_i];
	
				if(!tracker.bUpdated && !tracker.bTracking)
				{
					for (size_t c_j = 0; c_j < segments.size(); c_j++)
					{
						auto& segment = segments[c_j];
						if (!segment.bTracked)
						{
							tracker.addToHistory(segment);
							tracker.bTracking = true;
							tracker.bUpdated = true;
							segment.bTracked = true;
							break;
						}
					}
				}
			}

			// 잉여 트래커 삭제
			for (size_t c_i = 0; c_i < segmentTrackers.size(); c_i++)
			{
				auto& tracker = segmentTrackers[c_i];

				if (!tracker.bUpdated)
				{
					cout << "tracker: " << c_i <<  " not updated" << endl;

					segmentTrackers.erase(segmentTrackers.begin() + c_i);
					c_i--;
				}
			}
		}

		// 시각화
		{
			// 디버그
			for (size_t c_i = 0; c_i < segmentTrackers.size(); c_i++)
			{
				auto& segmentTracker = segmentTrackers[c_i];
				if (segmentTracker.bTracking)
				{
					Rect& segmentArea = segmentTracker.history.front().rect;
					if (segmentTracker.bActive)
					{
						rectangle(result, segmentArea, Scalar(50, 255, 50), 5);

						for (size_t c_i = segmentTracker.history.size() - 1; c_i > 0; c_i--)
						{
							auto& segment = segmentTracker.history[c_i];
							auto& nextSegment = segmentTracker.history[c_i - 1];

							Point mid = segment.mid;
							Point nextMid = nextSegment.mid;

							line(result, mid, nextMid, Scalar(255, 0, 255), 5);
						}
					}
					else
					{
						rectangle(result, segmentArea, Scalar(255, 0, 0), 2);
					}
				}
			}
		}

		imshow("result", result);
		//imshow("hsv", hsvBinary);
		//imshow("yCrCv", yCrCvBinary);
		imshow("combinedBinary", combinedBinary);
		
		int key = waitKey(FIXED_DT);
		switch (key)
		{
		case 27:
			return 0;

		case '1': // hand spoid
			bCaptureSampleColor = !bCaptureSampleColor;
			break;

		// 위치 조절
		case 'a':
			if (bCaptureSampleColor)
			{
				SpoidX -= 10;
			}
			break;
		case 'd':
			if (bCaptureSampleColor)
			{
				SpoidX += 10;
			}
			break;
		case 'w':
			if (bCaptureSampleColor)
			{
				SpoidY -= 10;
			}
			break;
		case 's':
			if (bCaptureSampleColor)
			{
				SpoidY += 10;
			}
			break;

		// 크기조절
		case 'q':
			if (bCaptureSampleColor)
			{
				SpoidColumns = MAX(SpoidColumns - 10, MIN_SPOID_WIDTH);
			}
			break;
		case 'e':
			if (bCaptureSampleColor)
			{
				SpoidColumns += 10;
			}
			break;
		case 'z':
			if (bCaptureSampleColor)
			{
				SpoidRows = MAX(SpoidRows - 10, MIN_SPOID_HEIGHT);
			}
			break;
		case 'c':
			if (bCaptureSampleColor)
			{
				SpoidRows += 10;
			}
			break;

		}
	}

	return 0;
}

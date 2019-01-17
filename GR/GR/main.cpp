#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include "Util.inl"
#include "ColorSampler.h"
#include "SegmentTracker.h"

using namespace cv;
using namespace std;

#define TARGET_FPS			60
#define FIXED_DT			((int)(1000.0/TARGET_FPS))
#define DEACTIVATION_FRAME	((int)(0.4*TARGET_FPS))
#define NUM_TRACKING_FRAME	120

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
	bool bCaptureSampleColor = false;
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
		if (bCaptureSampleColor)
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
					Rect SpoidRect = spoid.GetSampleRect();
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
						if (isPointInRect(candidate, getRectMid(spoid.GetSampleRect())))
						{
							float sampleSegmentArea = (float)candidate.area();
							minMergeDistance = (float)sqrt(sampleSegmentArea);
							break;
						}
					}
				}

				// 병합 
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

						if (isInMergeBound(segment.rect, candidate, 0.8f*minMergeDistance, 0))
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
			segmentTrackers.push_back(SegmentTracker(NUM_TRACKING_FRAME));
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
					float minDistance = (float)MAX(vWidth, vHeight);
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
								minIndex = (int)c_j;
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
								minIndex = (int)c_j;
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
				spoid.moveBy(-10, 0);
			}
			break;
		case 'd':
			if (bCaptureSampleColor)
			{
				spoid.moveBy(10, 0);
			}
			break;
		case 'w':
			if (bCaptureSampleColor)
			{
				spoid.moveBy(0, -10);
			}
			break;
		case 's':
			if (bCaptureSampleColor)
			{
				spoid.moveBy(0, 10);
			}
			break;

		// 크기조절
		case 'q':
			if (bCaptureSampleColor)
			{
				spoid.resizeBy(-10, 0);
			}
			break;
		case 'e':
			if (bCaptureSampleColor)
			{
				spoid.resizeBy(10, 0);
			}
			break;
		case 'z':
			if (bCaptureSampleColor)
			{
				spoid.resizeBy(0, -10);
			}
			break;
		case 'c':
			if (bCaptureSampleColor)
			{
				spoid.resizeBy(0, 10);
			}
			break;
		}
	}

	return 0;
}

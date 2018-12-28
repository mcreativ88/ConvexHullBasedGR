#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

class Contour
{
public:
	Contour(const vector<Point>& contour);

	vector<Point> contourPoints;
	Point mid;
	Rect boundary;

private:
};

Contour::Contour(const vector<Point>& contour)
	:contourPoints(contour)
{
}


void reverseColumns(Mat& inOutFrame);

int main(int, char**)
{
	VideoCapture capture(0);
	if (!capture.isOpened())
	{
		cout << "Camera open failed.\n";
		return -1;
	}

	int vWidth = capture.get(CAP_PROP_FRAME_WIDTH);
	int vHeight = capture.get(CAP_PROP_FRAME_HEIGHT);
	cout << vWidth << ", " << vHeight << endl;

	// 샘플 색상 추출
	bool bCaptureSampleColor = false;
	int SpoidColumns = 0.05f*vWidth;
	int SpoidRows = 0.05f*vWidth;
	int SpoidX = 0.75f*vWidth;
	int SpoidY = 0.5f*vHeight;
	Vec3f SampleColor(0, 0, 0);
	

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
	Mat handSpoid(SpoidRows, SpoidColumns, CV_8UC3);
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
					handSpoid.at<Vec3b>(row, col) = yCrCvColor;

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
			float margin = 15;
			inRange(hsv, Scalar(MinH, MinS - 5, 0), Scalar(MaxH, MaxS + margin, 255), hsvBinary);

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
		{
			vector<vector<Point>> contours;
			vector<Rect> candidates;

			findContours(combinedBinary, contours, RetrievalModes::RETR_EXTERNAL, ContourApproximationModes::CHAIN_APPROX_NONE);
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

			for (size_t c_i = 0; c_i < candidates.size(); c_i++)
			{
				Rect SpoidRect = Rect(SpoidX, SpoidY, SpoidColumns, SpoidRows);
				if (candidates[c_i].area() < SpoidRect.area())
				{
					candidates.erase(candidates.begin() + c_i);
					c_i--;
				}
			}

			for (size_t c_i = 0; c_i < candidates.size(); c_i++)
			{
				rectangle(result, candidates[c_i], Scalar(0, 255, 0), 2);
			}
		}

		imshow("result", result);
		imshow("combinedBinary", combinedBinary);
		
		int key = waitKey(16);

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
				SpoidColumns = MAX(SpoidColumns - 10, 0.05f*vWidth);
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
				SpoidRows = MAX(SpoidRows - 10, 0.05f*vWidth);
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

void reverseColumns(Mat& inOutFrame)
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

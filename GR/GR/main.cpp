#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

void reverseColumns(Mat& inOutFrame);

int main(int, char**)
{
	Mat input;
	Mat workspace;

	VideoCapture capture(0);
	if (!capture.isOpened())
	{
		cout << "Camera open failed.\n";
		return -1;
	}

	while (capture.read(input))
	{
		if (input.empty())
		{
			cerr << "ERROR! blank frame grabbed\n";
			break;
		}

		reverseColumns(input);

		input.copyTo(workspace);

		imshow("Output", workspace);

		int key = waitKey(16);
		switch (key)
		{
		case 27:
			return 0;
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

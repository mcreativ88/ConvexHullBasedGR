#include "SegmentTracker.h"

SegmentTracker::SegmentTracker(int n)
	:numTrackingFrames(n)
{

}

void SegmentTracker::preupdate()
{
	avgMid = Point(0, 0);
	avgInactiveSegmentMid = Point(0, 0);
	int InactiveSegments = 0;

	if (history.size() > 0)
	{
		for (size_t c_i = 0; c_i < history.size(); c_i++)
		{
			auto& segment = history[c_i];

			avgMid = avgMid + segment.mid;

			if (!segment.bActive)
			{
				InactiveSegments++;
				avgInactiveSegmentMid = avgInactiveSegmentMid + segment.mid;
			}
		}

		avgMid = (1.0f / (float)history.size())*avgMid;

		if (InactiveSegments > 0)
		{
			avgInactiveSegmentMid = (1.0f / (float)InactiveSegments)*avgInactiveSegmentMid;
		}
	}

	bUpdated = false;
}

void SegmentTracker::addToHistory(Segment& inSegment)
{
	while (history.size() >= numTrackingFrames)
	{
		history.pop_back();
	}

	history.insert(history.begin(), inSegment);
}

Point SegmentTracker::getNLatestSegmentMid(int _n)
{
	int n = MIN(_n, (int)history.size());

	Point mid(0, 0);
	if (n > 0)
	{
		for (size_t c_i = 0; c_i < n; c_i++)
		{
			auto& segment = history[c_i];

			mid = mid + segment.mid;
		}
	}
	mid = (1.0f / (float)n)*mid;

	return mid;
}

Point SegmentTracker::getNLatestInactiveSegmentMid(int _n)
{
	int n = MIN(_n, (int)history.size());
	int count = 0;

	Point mid(0, 0);
	if (n > 0)
	{
		for (size_t c_i = 0; c_i < history.size(); c_i++)
		{
			auto& segment = history[c_i];
			if (!segment.bActive)
			{
				mid = mid + segment.mid;
				count++;
			}

			if (count == n)
			{
				break;
			}
		}
	}
	mid = (1.0f / (float)n)*mid;

	return mid;
}
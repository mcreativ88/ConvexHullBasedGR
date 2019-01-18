#include "SegmentTracker.h"
#include "Util.inl"

SegmentTracker::SegmentTracker(int n)
	:numTrackingFrames(n)
{

}

void SegmentTracker::trackSegment(Segment& inSegmnet)
{
	addToHistory(inSegmnet);
	inSegmnet.bTracked = true;
}

void SegmentTracker::addToHistory(Segment& inSegment)
{
	while (history.size() >= numTrackingFrames)
	{
		history.pop_back();
	}

	history.insert(history.begin(), inSegment);

	bTracking = true;
	bUpdated = true;
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

TrackerManager::TrackerManager(int inNumTrackingTime, int inDeactivationTime)
	:numTrackingFrames(inNumTrackingTime),
	numDeactivationFrames(inDeactivationTime)
{
}

void TrackerManager::update(vector<Segment>& inSegments)
{
	segments = &inSegments;

	prepare();
	updateActiveTrackers();
	updateInactiveTrackers();
	assignSegments();
	removeUnusedTrackers();
}

void TrackerManager::visualize(Mat& inMat)
{
	for (size_t c_i = 0; c_i < segmentTrackers.size(); c_i++)
	{
		auto& segmentTracker = segmentTrackers[c_i];
		if (segmentTracker.bTracking)
		{
			Rect& segmentArea = segmentTracker.history.front().rect;
			if (segmentTracker.bActive)
			{
				rectangle(inMat, segmentArea, Scalar(50, 255, 50), 5);

				for (size_t c_i = segmentTracker.history.size() - 1; c_i > 0; c_i--)
				{
					auto& segment = segmentTracker.history[c_i];
					auto& nextSegment = segmentTracker.history[c_i - 1];

					Point mid = segment.mid;
					Point nextMid = nextSegment.mid;

					line(inMat, mid, nextMid, Scalar(255, 0, 255), 5);
				}
			}
			else
			{
				rectangle(inMat, segmentArea, Scalar(255, 0, 0), 2);
			}
		}
	}
}

void TrackerManager::prepare()
{
	// Ʈ��Ŀ �߰�
	while (segmentTrackers.size() < segments->size())
	{
		segmentTrackers.push_back(SegmentTracker(numTrackingFrames));
	}

	// ������Ʈ �غ�
	for (size_t c_i = 0; c_i < segmentTrackers.size(); c_i++)
	{
		auto& tracker = segmentTrackers[c_i];
		tracker.bUpdated = false;
	}
}

void TrackerManager::updateActiveTrackers()
{
	for (size_t c_i = 0; c_i < segmentTrackers.size(); c_i++)
	{
		auto& tracker = segmentTrackers[c_i];

		// ���� ���׸�Ʈ ����
		if (tracker.bTracking && tracker.bActive)
		{
			auto prevSegment = tracker.history.front();

			// ���� ����� ���׸�Ʈ�� ����
			float minDistance = FLT_MAX;
			int minIndex = -1;
			for (size_t c_j = 0; c_j < segments->size(); c_j++)
			{
				auto& segment = (*segments)[c_j];
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
				auto& nearestSegment = (*segments)[minIndex];

				// ���� �Ÿ� �̳��� ��� ��Ȱ��ȭ ���·� ����
				float threshold = 0.1f*(float)sqrt(prevSegment.rect.area());
				if (minDistance <= threshold)
				{
					if (++tracker.deactivateCounter > numDeactivationFrames)
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

				// �����丮�� ��ġ�� ���� �� ����
				auto copy = nearestSegment;
				copy.mid = 0.6f*copy.mid + 0.4f*prevSegment.mid;

				tracker.addToHistory(copy);
				nearestSegment.bTracked = true;
			}
		}
	}
}

void TrackerManager::updateInactiveTrackers()
{
	// ��Ȱ��ȭ ���� Ʈ��Ŀ ������Ʈ
	for (size_t c_i = 0; c_i < segmentTrackers.size(); c_i++)
	{
		auto& tracker = segmentTrackers[c_i];

		// ���� ���׸�Ʈ ����
		if (tracker.bTracking && !tracker.bActive)
		{
			auto prevSegment = tracker.history.front();

			// ���� ����� ���׸�Ʈ�� ����
			float minDistance = FLT_MAX;
			int minIndex = -1;
			for (size_t c_j = 0; c_j < segments->size(); c_j++)
			{
				auto& segment = (*segments)[c_j];
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
				auto& nearestSegment = (*segments)[minIndex];
				
				// ���� �Ÿ��� ����� Ȱ��ȭ ���·� ����
				float threshold = 0.3f*(float)sqrt(prevSegment.rect.area());
				if (minDistance > threshold)
				{
					tracker.history.clear();
					tracker.bActive = true;
					nearestSegment.bActive = true;
				}

				// �����丮�� ��ġ�� ���� �� ����
				auto copy = nearestSegment;
				copy.mid = 0.6f*copy.mid + 0.4f*prevSegment.mid;

				tracker.addToHistory(nearestSegment);
				nearestSegment.bTracked = true;
			}
		}
	}
}

void TrackerManager::assignSegments()
{
	for (size_t c_i = 0; c_i < segmentTrackers.size(); c_i++)
	{
		auto& tracker = segmentTrackers[c_i];

		if (!tracker.bUpdated && !tracker.bTracking)
		{
			for (size_t c_j = 0; c_j < segments->size(); c_j++)
			{
				auto& segment = (*segments)[c_j];
				if (!segment.bTracked)
				{
					tracker.trackSegment(segment);
					break;
				}
			}
		}
	}
}

void TrackerManager::removeUnusedTrackers()
{
	for (size_t c_i = 0; c_i < segmentTrackers.size(); c_i++)
	{
		auto& tracker = segmentTrackers[c_i];

		if (!tracker.bUpdated)
		{
			//cout << "tracker: " << c_i <<  " unused." << endl;

			segmentTrackers.erase(segmentTrackers.begin() + c_i);
			c_i--;
		}
	}
}
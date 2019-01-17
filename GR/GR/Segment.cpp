#include "Segment.h"
#include "Util.inl"

Segment::Segment(const Rect& inRect)
	:rect(inRect)
{
	mid = getRectMid(rect);
}

#include "Util.inl"
#include "Segment.h"

Segment::Segment(const Rect& inRect)
	:rect(inRect)
{
	mid = getRectMid(rect);
}

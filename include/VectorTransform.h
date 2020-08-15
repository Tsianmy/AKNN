#pragma once
#include "util.h"

namespace aknn {

	class VectorTransform
	{
	public:
		static float * transform(const float * x, idx_t nx, idx_t dx,
			const float * R, idx_t nR, idx_t dR);
	};


}
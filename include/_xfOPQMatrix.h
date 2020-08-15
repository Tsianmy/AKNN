#pragma once
#include <faiss/VectorTransform.h>

namespace faiss {

struct _xfOPQMatrix : OPQMatrix
{
	_xfOPQMatrix(int d = 0, int M = 1, int d2 = -1);
	void train(idx_t n, const float * x);
};

} // namespace faiss
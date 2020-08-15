#include "VectorTransform.h"

using namespace aknn;

extern "C" {

	// this is to keep the clang syntax checker happy
#ifndef FINTEGER
#define FINTEGER int
#endif


	/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

	int sgemm_(
		const char *transa, const char *transb, FINTEGER *m, FINTEGER *
		n, FINTEGER *k, const float *alpha, const float *a,
		FINTEGER *lda, const float *b,
		FINTEGER *ldb, float *beta,
		float *c, FINTEGER *ldc);

}

float * VectorTransform::transform(const float * x, idx_t nx, idx_t dx,
	const float * R, idx_t nR, idx_t dR)
{
	assert_aknn(dx == dR, "[nx * dx] * [dR * nR]");
	// X[nx * dx], R[nR * dR] -> M = nx K = dx N = dR
	// -> m = dR k = dx = nR n = nx -> A[m * k], B[k * n]
	float one = 1, zero = 0;
	FINTEGER m = dR, k = nR, n = nx;
	float * xt = new float[n * dx];
	sgemm_("Transposed", "Not transposed",
		&m, &n, &k,
		&one, R, &k, x, &k, &zero, xt, &m);
	return xt;
}

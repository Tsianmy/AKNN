#include "_xfOPQMatrix.h"
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/IndexPQ.h>

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

	int dgemm_(
		const char *transa, const char *transb, FINTEGER *m, FINTEGER *
		n, FINTEGER *k, const double *alpha, const double *a,
		FINTEGER *lda, const double *b,
		FINTEGER *ldb, double *beta,
		double *c, FINTEGER *ldc);

	int ssyrk_(
		const char *uplo, const char *trans, FINTEGER *n, FINTEGER *k,
		float *alpha, float *a, FINTEGER *lda,
		float *beta, float *c, FINTEGER *ldc);

	/* Lapack functions from http://www.netlib.org/clapack/old/single/ */

	int ssyev_(
		const char *jobz, const char *uplo, FINTEGER *n, float *a,
		FINTEGER *lda, float *w, float *work, FINTEGER *lwork,
		FINTEGER *info);

	int dsyev_(
		const char *jobz, const char *uplo, FINTEGER *n, double *a,
		FINTEGER *lda, double *w, double *work, FINTEGER *lwork,
		FINTEGER *info);

	int sgesvd_(
		const char *jobu, const char *jobvt, FINTEGER *m, FINTEGER *n,
		float *a, FINTEGER *lda, float *s, float *u, FINTEGER *ldu, float *vt,
		FINTEGER *ldvt, float *work, FINTEGER *lwork, FINTEGER *info);


	int dgesvd_(
		const char *jobu, const char *jobvt, FINTEGER *m, FINTEGER *n,
		double *a, FINTEGER *lda, double *s, double *u, FINTEGER *ldu, double *vt,
		FINTEGER *ldvt, double *work, FINTEGER *lwork, FINTEGER *info);

}

namespace faiss {

_xfOPQMatrix::_xfOPQMatrix(int d, int M, int d2) :OPQMatrix(d, M, d2)
{
	niter_pq_0 = 25;
}

void _xfOPQMatrix::train(idx_t n, const float * x)
{
	const float * x_in = x;

	x = fvecs_maybe_subsample(d_in, (size_t*)&n, max_train_points, x, verbose);

	ScopeDeleter<float> del_x(x != x_in ? x : nullptr);

	// To support d_out > d_in, we pad input vectors with 0s to d_out
	size_t d = d_out <= d_in ? d_in : d_out;
	size_t d2 = d_out;

	if (verbose) {
		printf("OPQMatrix::train: training an OPQ rotation matrix "
			"for M=%d from %ld vectors in %dD -> %dD\n",
			M, n, d_in, d_out);
	}

	std::vector<float> xtrain(n * d);
	// center x
	{
		std::vector<float> sum(d);
		const float *xi = x;
		for (size_t i = 0; i < n; i++) {
			for (int j = 0; j < d_in; j++)
				sum[j] += *xi++;
		}
		for (int i = 0; i < d; i++) sum[i] /= n;
		float *yi = xtrain.data();
		xi = x;
		for (size_t i = 0; i < n; i++) {
			for (int j = 0; j < d_in; j++)
				*yi++ = *xi++ - sum[j];
			yi += d - d_in;
		}
	}
	float *rotation;

	if (A.size() == 0) {
		A.resize(d * d);
		rotation = A.data();
		if (verbose)
			printf("  OPQMatrix::train: making %ld*%ld rotation\n",
				d, d);
		//float_randn (rotation, d * d, 1234);
		//matrix_qr (d, d, rotation);

		// use identity matrix
		for (int i = 0; i < d; i++) A[i * d + i] = 1;

		// we use only the d * d2 upper part of the matrix
		A.resize(d * d2);
	}
	else {
		FAISS_THROW_IF_NOT(A.size() == d * d2);
		rotation = A.data();
	}

	std::vector<float>
		xproj(d2 * n), pq_recons(d2 * n), xxr(d * n),
		tmp(d * d * 4);


	ProductQuantizer pq_default(d2, M, 8);
	ProductQuantizer &pq_regular = pq ? *pq : pq_default;
	std::vector<uint8_t> codes(pq_regular.code_size * n);

	double t0 = 0;
	if (verbose) {
		t0 = getmillisecs();
	}
	for (int iter = 0; iter < niter; iter++) {

		{ // torch.mm(xtrain, rotation:t())
			FINTEGER di = d, d2i = d2, ni = n;
			float zero = 0, one = 1;
			sgemm_("Transposed", "Not transposed",
				&d2i, &ni, &di,
				&one, rotation, &di,
				xtrain.data(), &di,
				&zero, xproj.data(), &d2i);
		}

		pq_regular.cp.min_points_per_centroid = 1;
		pq_regular.cp.max_points_per_centroid = 10000;
		pq_regular.cp.niter = iter == 0 ? niter_pq_0 : niter_pq;
		pq_regular.verbose = verbose;
		pq_regular.train(n, xproj.data());

		if (verbose) {
			printf("    encode / decode\n");
		}
		if (pq_regular.assign_index) {
			pq_regular.compute_codes_with_assign_index
			(xproj.data(), codes.data(), n);
		}
		else {
			pq_regular.compute_codes(xproj.data(), codes.data(), n);
		}
		pq_regular.decode(codes.data(), pq_recons.data(), n);


		if (verbose) {
			float pq_err = fvec_L2sqr(pq_recons.data(), xproj.data(), n * d2) / n;
			printf("    Iteration %d (%d PQ iterations):"
				"%.3f s, obj=%g\n", iter, pq_regular.cp.niter,
				(getmillisecs() - t0) / 1000.0, pq_err);
		}

		{
			float *u = tmp.data(), *vt = &tmp[d * d];
			float *sing_val = &tmp[2 * d * d];
			FINTEGER di = d, d2i = d2, ni = n;
			float one = 1, zero = 0;

			if (verbose) {
				printf("    X * recons\n");
			}
			// torch.mm(xtrain:t(), pq_recons)
			sgemm_("Not", "Transposed",
				&d2i, &di, &ni,
				&one, pq_recons.data(), &d2i,
				xtrain.data(), &di,
				&zero, xxr.data(), &d2i);


			FINTEGER lwork = -1, info = -1;
			float worksz;
			// workspace query
			sgesvd_("All", "All",
				&d2i, &di, xxr.data(), &d2i,
				sing_val,
				vt, &d2i, u, &di,
				&worksz, &lwork, &info);

			lwork = int(worksz);
			std::vector<float> work(lwork);
			// u and vt swapped
			sgesvd_("All", "All",
				&d2i, &di, xxr.data(), &d2i,
				sing_val,
				vt, &d2i, u, &di,
				work.data(), &lwork, &info);

			sgemm_("Transposed", "Transposed",
				&di, &d2i, &d2i,
				&one, u, &di, vt, &d2i,
				&zero, rotation, &di);

		}
		pq_regular.train_type = ProductQuantizer::Train_hot_start;
	}

	float pq_err = fvec_L2sqr(pq_recons.data(), xproj.data(), n * d2) / n;
	fprintf(stderr, "    %d OPQ iterations: obj=%g\n", niter, pq_err);


	// revert A matrix
	if (d > d_in) {
		for (long i = 0; i < d_out; i++)
			memmove(&A[i * d_in], &A[i * d], sizeof(A[0]) * d_in);
		A.resize(d_in * d_out);
	}

	is_trained = true;
	is_orthonormal = true;
}

} // namespace faiss
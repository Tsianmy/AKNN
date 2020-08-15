#include "mIndexMQ.h"
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexFlat.h>
#include <faiss/utils/distances.h>
#include <faiss/impl/FaissAssert.h>
#include <chrono>
#include <cstring>
#include <cmath>
using namespace std::chrono;

namespace aknn {

void mIndexMQ::compute_residual(float * q, idx_t klq, idx_t d, float * centroids)
{
	float mindis = INF(), *c = nullptr;
	for (idx_t ci = 0; ci < klq; ci++) {
		float dis = fvec_L2sqr(centroids, q, d);
		if (dis < mindis) {
			mindis = dis;
			c = centroids;
		}
		centroids += d;
	}
	fvec_sub(q, c, d);
}

void mIndexMQ::train(float * xt, idx_t nt, idx_t d, idx_t nbits, PQparameter param)
{
	{
		ScopeDeleter<float> delR(pq.R), delct(pq.centroids), dellqct(lq.centroids);
		ScopeDeleter<code_t> delcd(pq.codes);
		ScopeDeleter<lqc_t> dellqc(lq.codes);
	}
	fprintf(stderr, "Training on %ld vectors\n", nt);

	idx_t M = param.M, klq = param.klq;
	faiss::IndexFlatL2 * coarse_quantizer = new faiss::IndexFlatL2(d);
	faiss::IndexIVFPQ * index_ivfpq = new faiss::IndexIVFPQ(coarse_quantizer, d, klq, M, nbits);
	index_ivfpq->own_fields = true;
	{
		auto start = system_clock::now();
		// train
		index_ivfpq->cp.max_points_per_centroid = 100000;
		index_ivfpq->pq.cp.min_points_per_centroid = 1;
		index_ivfpq->pq.cp.max_points_per_centroid = 100000;
		index_ivfpq->train(nt, xt);

		auto end = system_clock::now();
		fprintf(stderr, "  Time cost: %.3f s\n", duration<float>(end - start).count());
		
		fprintf(stderr, "Evaluate\n");
		// reconstruct
		index_ivfpq->add(nt, xt);
		float * pq_recons = new float[nt * d];
		ScopeDeleter<float> delrecon(pq_recons);
		index_ivfpq->reconstruct_n(0, nt, pq_recons);
		// error
		float pq_err = faiss::fvec_L2sqr(pq_recons, xt, nt * d) / nt;
		fprintf(stderr, "    PQ distortion: %g\n", pq_err);
	}
	{
		if (verbose) fprintf(stderr, "  Copy lqcentroids...");
		lq.centroids = new float[klq * d];
		memcpy(lq.centroids, coarse_quantizer->xb.data(), klq * d * sizeof(*lq.centroids));

		if (verbose) fprintf(stderr, "  Copy lqcodes...");
		lq.codes = new lqc_t[nt];
		{
			faiss::Index::idx_t * assign = new faiss::Index::idx_t[nt];
			coarse_quantizer->assign(nt, xt, assign);
			std::copy(assign, assign + nt, lq.codes);
			delete[] assign;
		}

		if (verbose) fprintf(stderr, "  Copy codes...");
		pq.codes = new code_t[nt * M];
		index_ivfpq->pq.compute_codes(xt, pq.codes, nt);

		if (verbose) fprintf(stderr, "  Copy centroids...");
		idx_t dsub = index_ivfpq->pq.dsub, ksub = index_ivfpq->pq.ksub;
		pq.centroids = new float[M * ksub * dsub];
		memcpy(pq.centroids, index_ivfpq->pq.centroids.data(), M * ksub * dsub * sizeof(*pq.centroids));
		/*if(verbose)
		fprintf(stderr, "  %ld * %ld = %ld centroids.size: %ld\n", M * index_pq->pq.ksub,
		index_pq->pq.dsub, index_pq->pq.ksub * d, index_pq->pq.centroids.size());*/

		if (verbose) fprintf(stderr, "\n");
	}
	delete index_ivfpq;
}

mIndexMQ::~mIndexMQ()
{
	ScopeDeleter<float>(lq.centroids);
	ScopeDeleter<lqc_t>(lq.codes);
}

float *  mIndexMQ1::compute_residuals(float * xq)
{
	float * q = new float[nq * d];
	memcpy(q, xq, nq * d * sizeof(*q));
	float * q0 = q;
	for (idx_t qi = 0; qi < nq; qi++) {
		compute_residual(q0, lq.klq, d, lq.centroids);
		q0 += d;
	}
	return q;
}

void mIndexMQ2::search(Parameter params, idx_t * res)
{
	srand((unsigned int)time(0));
#pragma omp parallel for
	for (int qi = 0; qi < nq; qi++) {
		idx_t initPi = (rand() / (double)RAND_MAX * (double)nb);
		//idx_t initPi = rand();

		idx_t * neighbors = mIndex::get_ptr(res, qi, params.K);
		mIndexG::search_neighbors(qi, initPi, params.K, params.L, params.E, neighbors);
	}
}

float mIndexMQ2::compute_distance(idx_t pi, idx_t qi)
{
	float * cen = mIndex::get_ptr(lq.centroids, lq.codes[pi], d);
	float * p = new float[d];
	memcpy(p, cen, d * sizeof(*p));
	float * q = mIndex::get_ptr(xq, qi, d);

	float * p0 = p;
	code_t * code = get_ptr(pq.codes, pi, pq.M);
	for (idx_t m = 0; m < pq.M; m++) {
		cen = mIndex::get_ptr(pq.centroids, m * pq.ksub + code[m], pq.dsub);
		fvec_add(p0, cen, pq.dsub);
		p0 += pq.dsub;
	}

	float dis = fvec_L2sqr(p, q, d);
	delete[] p;
	return sqrt(dis);
}

void mIndexMQ3::search(Parameter params, idx_t * res)
{
	srand((unsigned int)time(0));
#pragma omp parallel for
	for (int qi = 0; qi < nq; qi++) {
		idx_t initPi = (rand() / (double)RAND_MAX * (double)nb);
		//idx_t initPi = rand();

		idx_t * neighbors = mIndex::get_ptr(res, qi, params.K);
		mIndexG::search_neighbors(qi, initPi, params.K, params.L, params.E, neighbors);
	}
}

float mIndexMQ3::compute_distance(idx_t pi, idx_t qi)
{
	float * cen = mIndex::get_ptr(lq.centroids, lq.codes[pi], d);
	float * p = new float[d];
	memcpy(p, cen, d * sizeof(*p));
	float * q = mIndex::get_ptr(xq, qi, d);

	float * p0 = p;
	code_t * code = get_ptr(pq.codes, pi, pq.M);
	for (idx_t m = 0; m < pq.M; m++) {
		cen = mIndex::get_ptr(pq.centroids, m * pq.ksub + code[m], pq.dsub);
		fvec_mul(p0, cen, pq.dsub);
		p0 += pq.dsub;
	}

	float dis = fvec_L2sqr(p, q, d);
	delete[] p;
	return sqrt(dis);
}

void mIndexMQ3::train(float * xt, idx_t nt, idx_t d, idx_t nbits, PQparameter param)
{
	{
		ScopeDeleter<float> delR(pq.R), delct(pq.centroids), dellqct(lq.centroids);
		ScopeDeleter<code_t> delcd(pq.codes);
		ScopeDeleter<lqc_t> dellqc(lq.codes);
	}
	fprintf(stderr, "Training on %ld vectors\n", nt);

	idx_t M = param.M, klq = param.klq;
	faiss::IndexFlatL2 * coarse_quantizer = new faiss::IndexFlatL2(d);
	faiss::IndexIVFPQ * index_ivfpq = new faiss::IndexIVFPQ(coarse_quantizer, d, klq, M, nbits);
	index_ivfpq->own_fields = true;
	{
		auto start = system_clock::now();
		// train
		index_ivfpq->cp.max_points_per_centroid = 1000000;
		index_ivfpq->pq.cp.min_points_per_centroid = 1;
		index_ivfpq->pq.cp.max_points_per_centroid = 100000;
		index_ivfpq->train_q1(nt, xt, false, index_ivfpq->metric_type);

		lq.codes = new lqc_t[nt];
		{
			faiss::Index::idx_t * assign = new faiss::Index::idx_t[nt];
			coarse_quantizer->assign(nt, xt, assign);
			std::copy(assign, assign + nt, lq.codes);
			delete[] assign;
		}

		float * beta = new float[nt * d];
		ScopeDeleter<float> delb(beta);
		for (idx_t i = 0; i < nt; i++) {
			for (idx_t j = 0; j < d; j++) {
				beta[i * d + j] = xt[i * d + j] / coarse_quantizer->xb[lq.codes[i] * d + j];
			}
		}
		index_ivfpq->pq.train(nt, beta);
		pq.codes = new code_t[nt * M];
		index_ivfpq->pq.compute_codes(beta, pq.codes, nt);

		auto end = system_clock::now();
		fprintf(stderr, "  Time cost: %.3f s\n", duration<float>(end - start).count());
		
		fprintf(stderr, "Evaluate\n");
		// reconstruct
		float * pq_recons = new float[nt * d];
		ScopeDeleter<float> delrecon(pq_recons);

		index_ivfpq->pq.decode(pq.codes, pq_recons, nt);
		lq.centroids = new float[klq * d];
		memcpy(lq.centroids, coarse_quantizer->xb.data(), klq * d * sizeof(*lq.centroids));

		float * p0 = pq_recons;
		for (idx_t i = 0; i < nt; i++) {
			float * y = mIndex::get_ptr(lq.centroids, lq.codes[i], d);
			fvec_mul(p0, y, d);
			p0 += d;
		}
		// error
		float pq_err = faiss::fvec_L2sqr(pq_recons, xt, nt * d) / nt;
		fprintf(stderr, "    PQ distortion: %g\n", pq_err);
		
	}
	{
		if (verbose) fprintf(stderr, "  Copy lqcentroids...");
		// ...

		if (verbose) fprintf(stderr, "  Copy lqcodes...");
		// ...

		if (verbose) fprintf(stderr, "  Copy codes...");
		// ...

		if (verbose) fprintf(stderr, "  Copy centroids...");
		idx_t dsub = index_ivfpq->pq.dsub, ksub = index_ivfpq->pq.ksub;
		pq.centroids = new float[M * ksub * dsub];
		memcpy(pq.centroids, index_ivfpq->pq.centroids.data(), M * ksub * dsub * sizeof(*pq.centroids));
		/*if(verbose)
		fprintf(stderr, "  %ld * %ld = %ld centroids.size: %ld\n", M * index_pq->pq.ksub,
		index_pq->pq.dsub, index_pq->pq.ksub * d, index_pq->pq.centroids.size());*/

		if (verbose) fprintf(stderr, "\n");
	}
	delete index_ivfpq;
}

}	// namespace aknn
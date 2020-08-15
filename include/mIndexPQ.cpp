#include "mIndexPQ.h"
#include "_xfOPQMatrix.h"
#include <faiss/IndexPQ.h>
#include <faiss/utils/distances.h>
#include <faiss/impl/FaissAssert.h>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>
using namespace std::chrono;

namespace aknn {

void mIndexPQ::search(Parameter params, idx_t * res)
{
	srand((unsigned int)time(0));
#pragma omp parallel for
	for (int qi = 0; qi < nq; qi++) {
		// Mod *******
		float * distance_table = new float[pq.M * pq.ksub];
		ScopeDeleter<float> del(distance_table);
		compute_dis_table(distance_table, qi);
		// *******

		idx_t initPi = (rand() / (double)RAND_MAX * (double)nb);
		//idx_t initPi = rand();

		idx_t * neighbors = mIndex::get_ptr(res, qi, params.K);
		search_neighbors(qi, distance_table, initPi, params.K, params.L, params.E, neighbors);
	}
}

void mIndexPQ::compute_dis_table(float * distance_table, idx_t qi)
{
	float * q = mIndex::get_ptr(xq, qi, d);
	for (idx_t m = 0; m < pq.M; m++) {
		for (idx_t cid = 0; cid < pq.ksub; cid++) {
			float * c = mIndex::get_ptr(pq.centroids, m * pq.ksub + cid, pq.dsub);
			distance_table[cid] = fvec_L2sqr(q, c, pq.dsub);
		}
		distance_table += pq.ksub;
		q += pq.dsub;
	}
}

void mIndexPQ::search_neighbors(idx_t qi, float * distance_table, idx_t initPi,
	idx_t K, idx_t L, idx_t E, idx_t * res)
{
	// Mod *******
	float initDis = compute_distance(initPi, distance_table);
	// *******
	Neighbor * S = new Neighbor[L + 1];
	ScopeDeleter<Neighbor> dpool(S);
	S[0] = Neighbor(initPi, initDis);
	size_t end = 1;

	std::vector<bool> vis(nb), in(nb);

	idx_t i = 0;
	while (i < L) {
		// find the index of the first unchecked node in S
		idx_t j;
		for (j = 0; j < end; j++) {
			if (!vis[S[j].id]) {
				i = j;
				break;
			}
		}
		if (j == end) break;

		idx_t id = S[i].id;
		vis[id] = true;

		// add neighbors to the candidate pool and reorder by distance
		idx_t * neighbors = mIndex::get_ptr(xg, id, k);
		for (j = 0; j < E; j++) {
			if (vis[neighbors[j]] || in[neighbors[j]]) continue;
			// Mod *******
			float dis = compute_distance(neighbors[j], distance_table);
			// *******
			insert_pool(S, Neighbor(neighbors[j], dis), end);
			end = std::min(end + 1, (size_t)L);
			in[neighbors[j]] = true;
		}
	}
	for (i = 0; i < K && i < end; i++) res[i] = S[i].id;
}

float mIndexPQ::compute_distance(idx_t pi, float * distance_table)
{
	code_t * code = get_ptr(pq.codes, pi, pq.M);
	float dis = 0;
	idx_t blockn = pq.M / 4, rem = pq.M % 4, m = 0;
	for (idx_t i = 0; i < blockn; i++) {
		dis += distance_table[m * pq.ksub + code[m]]; m++;
		dis += distance_table[m * pq.ksub + code[m]]; m++;
		dis += distance_table[m * pq.ksub + code[m]]; m++;
		dis += distance_table[m * pq.ksub + code[m]]; m++;
	}
	for (idx_t i = 0; i < rem; i++) {
		dis += distance_table[m * pq.ksub + code[m]]; m++;
	}

	return sqrt(dis);
}

code_t * mIndexPQ::get_ptr(code_t * begin, const idx_t idx, const idx_t d)
{
	return begin + idx * d;
}

void mIndexPQ::train(float * xt, idx_t nt, idx_t d, idx_t nbits, PQparameter param)
{
	{
		ScopeDeleter<float> delR(pq.R), delct(pq.centroids);
		ScopeDeleter<code_t> delcd(pq.codes);
	}

	fprintf(stderr, "Training on %ld vectors\n", nt);

	fprintf(stderr, "  Preparing index PQ d=%ld\n", d);
	idx_t M = param.M;
	faiss::IndexPQ * index_pq = new faiss::IndexPQ(d, M, nbits, faiss::MetricType::METRIC_L2);
	faiss::VectorTransform * vt = use_opq ? new faiss::_xfOPQMatrix(d, M) : nullptr;
	{
		auto start = system_clock::now();
		// train
		if (use_opq) vt->train(nt, xt);
		const float *xpt = use_opq ? vt->apply(nt, xt) : xt;
		faiss::ScopeDeleter<float> del(xpt == xt ? nullptr : xpt);

		index_pq->pq.cp.min_points_per_centroid = 1;
		index_pq->pq.cp.max_points_per_centroid = 10000;
		index_pq->train(nt, xpt);

		auto end = system_clock::now();
		fprintf(stderr, "  Time cost: %.3f s\n", duration<float>(end - start).count());
		
		fprintf(stderr, "Evaluate\n");
		// reconstruct
		index_pq->add(nt, xpt);
		float * pq_recons = new float[nt * d];
		ScopeDeleter<float> delrecon(pq_recons);
		index_pq->pq.decode(index_pq->codes.data(), pq_recons, nt);
		// error
		float pq_err = faiss::fvec_L2sqr(pq_recons, xpt, nt * d) / nt;
		fprintf(stderr, "    PQ distortion: %g\n", pq_err);
	}
	{
		if (use_opq) {
			pq.R = new float[d * d];
			faiss::OPQMatrix * opqm = dynamic_cast<faiss::OPQMatrix *>(vt);
			if (verbose) fprintf(stderr, "Copy R...");
			memcpy(pq.R, opqm->A.data(), d * d * sizeof(*pq.R));
		}

		if (verbose) fprintf(stderr, "  Copy codes...");
		pq.codes = new code_t[nt * M];
		memcpy(pq.codes, index_pq->codes.data(), nt * M * sizeof(*pq.codes));

		if (verbose) fprintf(stderr, "  Copy centroids...");
		idx_t dsub = index_pq->pq.dsub, ksub = index_pq->pq.ksub;
		pq.centroids = new float[M * ksub * dsub];
		memcpy(pq.centroids, index_pq->pq.centroids.data(), M * ksub * dsub * sizeof(*pq.centroids));

		/*if(verbose)
		fprintf(stderr, "  %ld * %ld = %ld centroids.size: %ld\n", M * index_pq->pq.ksub,
		index_pq->pq.dsub, index_pq->pq.ksub * d, index_pq->pq.centroids.size());*/

		if (verbose) fprintf(stderr, "\n");
	}
	delete vt;
	delete index_pq;
}

mIndexPQ::~mIndexPQ()
{
	ScopeDeleter<float>(pq.centroids);
	ScopeDeleter<code_t>(pq.codes);
}

}	// namespace aknn
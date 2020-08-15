#include "mIndexG.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>
#include <ctime>

namespace aknn {

void mIndexG::search(Parameter params, idx_t * res)
{
	srand((unsigned int)time(0));
#pragma omp parallel for
	for (int qi = 0; qi < nq; qi++) {
		idx_t initPi = (rand() / (double)RAND_MAX * (double)nb);
		//idx_t initPi = rand();

		idx_t * neighbors = get_ptr(res, qi, params.K);
		search_neighbors(qi, initPi, params.K, params.L, params.E, neighbors);
	}
}

float mIndexG::search_R(Parameter params, idx_t * res)
{
	srand((unsigned int)time(0));
	idx_t sum = 0;

#pragma omp parallel for reduction(+:sum)
	for (int qi = 0; qi < nq; qi++) {
		idx_t * tempn = new idx_t[params.K];
		ScopeDeleter<idx_t> del(tempn);
		int maxc = -1;
		for (idx_t r = 0; r < params.R; r++) {
			idx_t initPi = (rand() * 1.0 / RAND_MAX * nb);
			//idx_t initPi = rand();

			search_neighbors(qi, initPi, params.K, params.L, params.E, tempn);

			int cnt = evaluate_s(tempn, qi, gt, params.K);
			if (cnt > maxc) {
				maxc = cnt;
				idx_t * neighbors = get_ptr(res, qi, params.K);
				memcpy(neighbors, tempn, params.K * sizeof(*neighbors));
			}
		}
		sum += maxc;
	}

	float acc = sum / (1.0 * nq * params.K);
	return acc;
}

void mIndexG::search_neighbors(idx_t qi, idx_t initPi, idx_t K, idx_t L, idx_t E, idx_t * res)
{
	float initDis = compute_distance(initPi, qi);

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
		idx_t * neighbors = get_ptr(xg, id, k);
		for (j = 0; j < E; j++) {
			if (vis[neighbors[j]] || in[neighbors[j]]) continue;
			float dis = compute_distance(neighbors[j], qi);
			insert_pool(S, Neighbor(neighbors[j], dis), end);
			end = std::min(end + 1, (size_t)L);
			in[neighbors[j]] = true;
		}
	}
	for (i = 0; i < K && i < end; i++) res[i] = S[i].id;
}

float mIndexG::compute_distance(idx_t pi, idx_t qi)
{
	float * x = get_ptr(xb, pi, d);
	float * y = get_ptr(xq, qi, d);
	float dis = fvec_L2sqr(x, y, d);
	return sqrt(dis);
}

void mIndexG::insert_pool(Neighbor * pool, Neighbor p, idx_t end)
{
	size_t i = end;
	while (i > 0 && p.distance < pool[i - 1].distance) {
		pool[i] = pool[i - 1];
		i--;
	}
	pool[i] = p;
}

}

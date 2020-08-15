#include "mIndex.h"

namespace aknn {

void mIndex::search(Parameter params, idx_t * res)
{
	// do nothing
}

float mIndex::search_R(Parameter params, idx_t * res)
{
	// do nothing
	return 0.0f;
}

void mIndex::search_neighbors(idx_t qi, idx_t initPi, idx_t K, idx_t L, idx_t E, idx_t * res)
{
	// do nothing
}

float * mIndex::get_ptr(float * begin, const idx_t idx, const idx_t d)
{
	return begin + idx * d;
}

idx_t * mIndex::get_ptr(idx_t * begin, const idx_t idx, const idx_t d)
{
	return begin + idx * d;
}

float mIndex::compute_distance(idx_t, idx_t)
{
	// do nothing
	return 0;
}

void mIndex::train(float * xt, idx_t nt, idx_t d, idx_t nbits, PQparameter param)
{
	// do nothing
}

float mIndex::evaluate(idx_t * res, std::unordered_set<idx_t> * gt, idx_t nq, idx_t K)
{
	idx_t cnt = 0;
	for (idx_t qi = 0; qi < nq; qi++) {
		for (idx_t j = 0; j < K; j++) {
			if (gt[qi].count(res[j])) cnt++;
		}
		res += K;
	}
	return cnt / (1.0 * nq * K);
}

int mIndex::evaluate_s(idx_t * res, idx_t qi, std::unordered_set<idx_t>* gt, idx_t K)
{
	int cnt = 0;
	for (idx_t i = 0; i < K; i++) {
		if (gt[qi].count(res[i])) cnt++;
	}
	return cnt;
}

} // namespace aknn
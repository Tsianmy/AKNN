#pragma once
#include "mIndex.h"

namespace aknn {

class mIndexG : public mIndex
{
public:
	mIndexG() {}

	void search(Parameter params, idx_t * res) override;
	float search_R(Parameter params, idx_t * res) override;
	void search_neighbors(idx_t qi, idx_t initPi, idx_t K, idx_t L, idx_t E, idx_t * res) override;

	float compute_distance(idx_t pi, idx_t qi) override;
	static void insert_pool(Neighbor * pool, Neighbor p, idx_t end);
};

} // namespace aknn
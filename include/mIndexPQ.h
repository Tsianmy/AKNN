#pragma once
#include "mIndexG.h"
#include "pq.h"

namespace aknn {

class mIndexPQ : public mIndexG
{
public:
	ProductQuantizer pq;
	bool use_opq = false;

	mIndexPQ() {}
	mIndexPQ(bool _use_opq): use_opq(_use_opq){}

	void search(Parameter params, idx_t * res) override;
	void compute_dis_table(float * distance_table, idx_t qi);

	void search_neighbors(idx_t qi, float * distance_table, idx_t initPi,
		idx_t K, idx_t L, idx_t E, idx_t * res);
	float compute_distance(idx_t pi, float * distance_table);
	static code_t * get_ptr(code_t * begin, const idx_t idx, const idx_t d);

	void train(float * xt, idx_t nt, idx_t d, idx_t nbits, PQparameter param) override;
	~mIndexPQ();
};

}	// namespace aknn
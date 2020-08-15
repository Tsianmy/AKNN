#pragma once
#include "AKNN.h"
#include "mIndexPQ.h"
#include "mIndexMQ.h"

namespace aknn {

class AKNN_EX :	public AKNN
{
public:
	const char * Rname, *cenname, *codename;
	const char * lqcenname, *lqcodename;
	bool use_opq = false;
	bool lack_mem = true;

	AKNN_EX() {}
	AKNN_EX(
		const char * base, const char * query, const char * knngraph, const char * groundtruth,
		const char * R = "", const char * centroids = "", const char * codes = "",
		const char * lqcentroids = "", const char * lqcodes = ""):
		AKNN(base, query, knngraph, groundtruth),Rname(R), cenname(centroids), codename(codes),
		lqcenname(lqcentroids), lqcodename(lqcodes)
	{}

	void refine(PQparameter * params, size_t pmsize, size_t pmnum, Parameter search_params);
	void search(Parameter * search_params, size_t pmsize, const char * outname = "");
	void train(float * xt, idx_t nt, idx_t d, idx_t nbits, PQparameter param);
	
	void prepare_pq(mIndexPQ * index_pq,
		const char * Rname, const char * cenname, const char * codename);
	void prepare_mq(mIndexMQ * index_mq, const char * cenname, const char * codename);

	void clear();
};

}	// namespace aknn
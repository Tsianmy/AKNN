#include "../include/AKNN_EX.h"
#include <omp.h>
#include <vector>
using namespace aknn;

#ifdef GIST
const char
*basename = "../data/gist_base.fvecs",
*queryname = "../data/gist_query.fvecs",
*graphname = "../data/gist_100NN_100.graph",
*gtname = "../data/gist_groundtruth.ivecs",
*Rname = "../data/gist_R.fvecs",
*codename = "../data/gist_pqcodes.codes",
*cenname = "../data/gist_centab.fvecs",
*outname = "../data/searchRes.ivecs";
#else
const char
*basename = "../data/sift_base.fvecs",
*queryname = "../data/sift_query.fvecs",
*graphname = "../data/sift_100NN_100.graph",
*gtname = "../data/sift_groundtruth.ivecs",
*Rname = "../data/sift_R.fvecs",
*codename = "../data/sift_pqcodes.codes",
*cenname = "../data/sift_centab.fvecs",
*outname = "../data/searchRes.ivecs";
#endif

int main(int argc, char** argv)
{
	omp_set_num_threads(4);
	freopen("../log.txt", "w", stdout);
#if 1
	AKNN_EX aknn(basename, queryname, graphname, gtname, Rname, cenname, codename);
#else
	AKNN_EX aknn(basename, queryname, graphname, gtname);
#endif
	aknn.index = new mIndexPQ;
	aknn.verbose = true;
	aknn.use_opq = true;
#ifndef GIST
	aknn.lack_mem = false;
#endif
	const idx_t K = 100;
	idx_t L = K, E = K, R = 1;

#ifndef plot

	Parameter params(K, L, E);
	PQparameter pq_params[] = { {64}};
	aknn.refine(pq_params, sizeof(pq_params) / sizeof(pq_params[0]), 1, params);

#else

	std::vector<Parameter> params;
	for (E = 10; E <= K; E += 10) params.emplace_back(Parameter(K, L, E));
	E = K;
	for (L = 200; L <= 1200; L += 100) params.emplace_back(Parameter(K, L, E));
	aknn.search(params.data(), params.size());

#endif

	return 0;
}
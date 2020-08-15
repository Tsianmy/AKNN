#include <omp.h>
#include "../include/AKNN.h"
#include "../include/mIndexHeap.h"
using namespace aknn;

#define SIFT
#ifdef GIST
const char
*basename = "../data/gist_base.fvecs",
*queryname = "../data/gist_query.fvecs",
*graphname = "../data/gist_100NN_100.graph",
*gtname = "../data/gist_groundtruth.ivecs",
*outname = "../data/searchRes.ivecs";
#else
const char
*basename = "../data/sift_base.fvecs",
*queryname = "../data/sift_query.fvecs",
*graphname = "../data/sift_100NN_100.graph",
*gtname = "../data/sift_groundtruth.ivecs",
*outname = "../data/searchRes.ivecs";
#endif

int main(int argc, char** argv)
{
	omp_set_num_threads(4);
	freopen("../log.txt", "w", stdout);
	AKNN aknn(basename, queryname, graphname, gtname);
	aknn.index = new mIndexHeap;
	aknn.verbose = true;
	const idx_t K = 100;
	idx_t L = K, E = K, R = 1;

#ifndef plot

	Parameter params(K, L, E);
	aknn.search(&params, 1);

#else
	std::vector<Parameter> params;
	for (E = 10; E <= K; E += 10) params.emplace_back(Parameter(K, L, E));
	E = K;
	for (L = 200; L <= 1200; L += 100) params.emplace_back(Parameter(K, L, E));
	aknn.search(params.data(), params.size());

#endif

	return 0;
}
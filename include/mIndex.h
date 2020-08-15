#pragma once

#include "util.h"
#include "pq.h"
#include <unordered_set>

namespace aknn {

class mIndex
{
public:
	// base
	float * xb = nullptr;
	// query
	float * xq = nullptr;
	// knn graph
	idx_t * xg = nullptr;
	// groundtruth
	std::unordered_set<idx_t> * gt = nullptr;

	idx_t nb, nq, d, k;
	bool verbose = false;

	mIndex() {}

	/* query n vectors of dimension d to the index
	*
	* return params.K nearest neighbors for each vectors
	*
	* @param params		 input search parameters
	* @param res         output index vectors of the NNs, size n * params.K
	*/
	virtual void search(Parameter params, idx_t * res);

	/* query n vectors of dimension d to the index
	* restart R times for each query
	*
	* return recall and params.K nearest neighbors for each vectors
	*
	* @param params		 input search parameters
	* @param res         output index vectors of the NNs, size n * params.K
	*/
	virtual float search_R(Parameter params, idx_t * res);

	/* return the indexes of the K neighbors closest to the query q
	*
	* @param qi				input index of q
	* @param iniPi			input index of init point for search
	* @param L				input size of candidate pool
	* @param E				input expanding number of each point
	* @param res			output index vector of the NNs, size params.K
	*/
	virtual void search_neighbors(idx_t qi, idx_t initPi, idx_t K, idx_t L, idx_t E, idx_t * res);

	// get the pointer at the position in data
	static float * get_ptr(float * begin, const idx_t idx, const idx_t d);
	static idx_t * get_ptr(idx_t * begin, const idx_t idx, const idx_t d);

	// compute the distance between two points given by index
	virtual float compute_distance(idx_t, idx_t);

	/** Perform training on a representative set of vectors
	*
	* @param xt			training vecors, size n * d
	* @param nt			nb of training vectors
	* @param nbits		number of bits per subvector
	* @param param		PQparameter
	*/
	virtual void train(float * xt, idx_t nt, idx_t d, idx_t nbits, PQparameter param);

	/* return the correct numbers of neighbors of queries
	*
	* @param res			input index vector of the NNs, size K
	* @param gt				input groundtruth
	*/
	static float evaluate(idx_t * res, std::unordered_set<idx_t> * gt, idx_t nq, idx_t K);

	/* return the correct numbers of neighbors of a single query
	*
	* @param res			input index vector of the NNs, size K
	* @param qi				input index of q
	* @param gt				input groundtruth
	*/
	static int evaluate_s(idx_t * res, idx_t qi, std::unordered_set<idx_t> * gt, idx_t K);
	virtual ~mIndex() {}
};


}	// namespace aknn
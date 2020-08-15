#pragma once
#include "mIndexPQ.h"

namespace aknn {

class mIndexMQ : public mIndexPQ
{
public:
	LevelQuantizer lq;

	mIndexMQ(){}
	static void compute_residual(float * q, idx_t klq, idx_t d, float * centroids);
	void train(float * xt, idx_t nt, idx_t d, idx_t nbits, PQparameter param) override;

	~mIndexMQ();
};

class mIndexMQ1 : public mIndexMQ
{
public:
	mIndexMQ1() {}

	float * compute_residuals(float * xq);

};

class mIndexMQ2 : public mIndexMQ
{
public:
	mIndexMQ2() {}

	void search(Parameter params, idx_t * res) override;

	float compute_distance(idx_t pi, idx_t qi) override;
};

class mIndexMQ3 : public mIndexMQ
{
public:
	mIndexMQ3() {}

	void search(Parameter params, idx_t * res) override;
	float compute_distance(idx_t pi, idx_t qi) override;

	void train(float * xt, idx_t nt, idx_t d, idx_t nbits, PQparameter param) override;
};

}	// namespace aknn
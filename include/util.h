#pragma once

#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <immintrin.h>
#include <cfloat>

namespace aknn {

using idx_t = unsigned int;

static float INF() { return FLT_MAX; }

struct Parameter
{
	idx_t K, L, E, R;
	Parameter() {}
	Parameter(idx_t _K, idx_t _L, idx_t _E, idx_t _R = 1) : K(_K), L(_L), E(_E), R(_R) {}
};

struct Neighbor
{
	idx_t id;
	float distance;
	Neighbor(idx_t _id = 0, float _d = 0) : id(_id), distance(_d) {}
	bool operator < (const Neighbor & n) const {
		return distance < n.distance;
	}
	bool operator > (const Neighbor & n) const {
		return distance > n.distance;
	}
};

template<class T>
struct ScopeDeleter {
	T * & ptr;
	explicit ScopeDeleter(T * & ptr) : ptr(ptr) {}
	~ScopeDeleter() {
		delete[] ptr;
		ptr = nullptr;
	}
};


// squared L2 distance between two vectors
static float fvec_L2sqr(float * x, float * y, const idx_t d)
{
	int nBlockWidth = 8;
	int cntBlock = d / nBlockWidth;
	int cntRem = d % nBlockWidth;

	__m256 mload1, mload2,
		mSub = _mm256_setzero_ps(),
		mSum = _mm256_setzero_ps();
	float *p1 = x, *p2 = y;
	for (int i = 0; i < cntBlock; i++)
	{
		mload1 = _mm256_loadu_ps(p1);
		mload2 = _mm256_loadu_ps(p2);
		mSub = _mm256_sub_ps(mload1, mload2);
		mSum = _mm256_fmadd_ps(mSub, mSub, mSum);
		p1 += nBlockWidth;
		p2 += nBlockWidth;
	}
	mSum = _mm256_hadd_ps(mSum, mSum);
	mSum = _mm256_hadd_ps(mSum, mSum);

	float sum = 0;
#ifdef __linux__
	sum += mSum[0];
	sum += mSum[4];
#else
	sum += mSum.m256_f32[0];
	sum += mSum.m256_f32[4];
#endif
	for (int i = 0; i < cntRem; i++) sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);

	return sum;
}

// subtraction between two vectors (x - y)
static void fvec_sub(float * x, float * y, const idx_t d)
{
	int nBlockWidth = 8;
	int cntBlock = d / nBlockWidth;
	int cntRem = d % nBlockWidth;

	__m256 mload1, mload2, mSub;
	float *p1 = x, *p2 = y;
	for (int i = 0; i < cntBlock; i++) {
		mload1 = _mm256_loadu_ps(p1);
		mload2 = _mm256_loadu_ps(p2);
		mSub = _mm256_sub_ps(mload1, mload2);
		_mm256_storeu_ps(p1, mSub);
		p1 += nBlockWidth;
		p2 += nBlockWidth;
	}
	for (int i = 0; i < cntRem; i++) p1[i] -= p2[i];
}

// add vector y to vector x
static void fvec_add(float * x, float * y, const idx_t d)
{
	int nBlockWidth = 8;
	int cntBlock = d / nBlockWidth;
	int cntRem = d % nBlockWidth;

	__m256 mload1, mload2, mSum;
	float *p1 = x, *p2 = y;
	for (int i = 0; i < cntBlock; i++) {
		mload1 = _mm256_loadu_ps(p1);
		mload2 = _mm256_loadu_ps(p2);
		mSum = _mm256_add_ps(mload1, mload2);
		_mm256_storeu_ps(p1, mSum);
		p1 += nBlockWidth;
		p2 += nBlockWidth;
	}
	for (int i = 0; i < cntRem; i++) p1[i] += p2[i];
}

// multiply vector x by vector y for each element
static void fvec_mul(float * x, float * y, const idx_t d)
{
	int nBlockWidth = 8;
	int cntBlock = d / nBlockWidth;
	int cntRem = d % nBlockWidth;

	__m256 mload1, mload2, mProd;
	float *p1 = x, *p2 = y;
	for (int i = 0; i < cntBlock; i++) {
		mload1 = _mm256_loadu_ps(p1);
		mload2 = _mm256_loadu_ps(p2);
		mProd = _mm256_mul_ps(mload1, mload2);
		_mm256_storeu_ps(p1, mProd);
		p1 += nBlockWidth;
		p2 += nBlockWidth;
	}
	for (int i = 0; i < cntRem; i++) p1[i] *= p2[i];
}

static void assert_aknn(bool expr, const char * msg = "")
{
	if (!expr) {
		fprintf(stderr, "%s\n", msg);
		exit(-1);
	}
}

}
#pragma once

#include "util.h"
#include <cstdint>

namespace aknn {

struct PQparameter
{
	idx_t M;
	idx_t klq;
};

using code_t = uint8_t;

struct ProductQuantizer
{
	idx_t M, dsub, ksub;
	float * R = nullptr;			// d * d
	float * centroids = nullptr;	// (M * ksub) * dsub
	code_t * codes = nullptr;		// n * M
};

using lqc_t = uint16_t;

struct LevelQuantizer
{
	idx_t klq;
	float * centroids = nullptr;	// klq * d
	lqc_t * codes = nullptr;		// n * 1
};

}	// namespace aknn
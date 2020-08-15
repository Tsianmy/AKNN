#pragma once

#include "util.h"
#include <cstdint>

namespace aknn {

class AKNN_IO
{
public:
	static void fvecs_read(const char *fname, float * & x, idx_t & n, idx_t & d);
	static void ivecs_read(const char *fname, idx_t * & x, idx_t & n, idx_t & d);
	static void codes_read(const char *fname, uint8_t * & x, idx_t & n, idx_t & d);
	static void codes_read(const char *fname, uint16_t * & x, idx_t & n, idx_t & d);

	static void fvecs_write(const char *fname, float * x, idx_t n, idx_t d);
	static void ivecs_write(const char *fname, idx_t * x, idx_t n, idx_t d);
	static void codes_write(const char *fname, uint8_t * x, idx_t n, idx_t d);
	static void codes_write(const char *fname, uint16_t * x, idx_t n, idx_t d);

	static void fvecs_mmap(const char *fname, float * & x, idx_t & n, idx_t & d);
};

}	// namespace aknn
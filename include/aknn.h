#pragma once

#include "AKNN_IO.h"
#include "mIndex.h"

namespace aknn {

class AKNN
{
public:
	const char *bname, *qname, *knngname, *gtname;
	mIndex * index = nullptr;
	bool verbose = false, use_mmap = false;

	AKNN() {}
	AKNN(const char * base, const char * query,
		const char * knngraph, const char * groundtruth) :
		bname(base), qname(query), knngname(knngraph), gtname(groundtruth) {}

	void search(Parameter * params, size_t pmsize, const char * outname = "");
	void search_o(Parameter params, const char * outname = "");
	void prepare(mIndex * index, const char * base, const char * query,
		const char * knngraph, const char * groundtruth);

	static void display(float * x, idx_t n, idx_t d);
	static void display(idx_t * x, idx_t n, idx_t d);
	static void display(uint8_t * x, idx_t n, idx_t d);
	static void display(uint16_t * x, idx_t n, idx_t d);

	void save_res(const char * resname, idx_t * x, idx_t n, idx_t d);

	void clear();
};

}	// namespace aknn
#include "AKNN.h"
#include <cstring>
#include <algorithm>
#include <chrono>

namespace aknn {

void AKNN::search(Parameter * params, size_t pmsize, const char * outname)
{
	assert_aknn(index, "No index");
	prepare(index, bname, qname, knngname, gtname);

	fprintf(stderr, "search...\n");
	for (size_t i = 0; i < pmsize; i++) {
		fprintf(stderr, "E: %d	L: %d\n", params[i].E, params[i].L);
		search_o(params[i], outname);
	}
	clear();
}

void AKNN::search_o(Parameter params, const char * outname)
{
	idx_t * res = new idx_t[index->nq * params.K];
	ScopeDeleter<idx_t> del(res);
	{
		auto start = std::chrono::system_clock::now();
		index->search(params, res);
		auto end = std::chrono::system_clock::now();
		float qtime = std::chrono::duration<float>(end - start).count();
		fprintf(stderr, "\nquery time: %.5f\n"
			"QPS: %f\n",
			qtime, index->nq * 1.0 / qtime);
	}
	float recall = index->evaluate(res, index->gt, index->nq, params.K);
	fprintf(stderr, "average accuracy: %f\n\n", recall);
	if (strlen(outname) > 0) save_res(outname, res, index->nq, index->d);
}

void AKNN::prepare(mIndex * index, const char * base, const char * query, const char * knngraph, const char * groundtruth)
{
	idx_t dq, kgt, ng, ngt;
	if (strlen(base) > 0) {
		if (use_mmap) AKNN_IO::fvecs_mmap(base, index->xb, index->nb, index->d);
		else AKNN_IO::fvecs_read(base, index->xb, index->nb, index->d);
		fprintf(stderr, "[%dx%d] read base: %s\n", index->nb, index->d, base);
		if (verbose) {
			printf("\nbase\n");
			display(index->xb, index->nb, index->d);
		}
	}
	if (strlen(query) > 0) {
		AKNN_IO::fvecs_read(query, index->xq, index->nq, dq);
		fprintf(stderr, "[%dx%d] read query: %s\n", index->nq, dq, query);
		if (verbose) {
			printf("\nquery\n");
			display(index->xq, index->nq, dq);
		}
		assert_aknn(index->d == dq, "weird data size");
	}
	if (strlen(knngraph) > 0) {
		AKNN_IO::ivecs_read(knngraph, index->xg, ng, index->k);
		fprintf(stderr, "[%dx%d] read graph: %s\n", ng, index->k, knngraph);
		if (verbose) {
			printf("\ngraph\n");
			display(index->xg, ng, index->k);
		}
		assert_aknn(index->nb == ng, "weird data size");
	}
	if (strlen(groundtruth) > 0) {
		idx_t * xgt = nullptr;
		AKNN_IO::ivecs_read(groundtruth, xgt, ngt, kgt);
		ScopeDeleter<idx_t> del(xgt);
		fprintf(stderr, "[%dx%d] read groundtruth: %s\n", ngt, kgt, groundtruth);
		if (verbose) {
			printf("\ngroundtruth\n");
			display(xgt, ngt, kgt);
		}
		assert_aknn(index->nq == ngt && index->k <= kgt, "weird data size");
		index->gt = new std::unordered_set<idx_t>[ngt];
		for (idx_t i = 0; i < ngt; i++) {
			for (idx_t j = 0; j < index->k; j++) index->gt[i].insert(xgt[i * kgt + j]);
		}
	}
}

void AKNN::display(float * x, idx_t n, idx_t d)
{
	idx_t num = std::min((idx_t)3, n);
	for (idx_t i = 0; i < num; i++) {
		for (idx_t j = 0; j < d; j++) {
			if (j > 0 && j % 17 == 0) printf("\n");
			printf(" %6.3g", x[i * d + j]);
		}
		printf("\n\n");
	}
}

void AKNN::display(idx_t * x, idx_t n, idx_t d)
{
	idx_t num = std::min((idx_t)3, n);
	for (idx_t i = 0; i < num; i++) {
		for (idx_t j = 0; j < d; j++) {
			if (j > 0 && j % 17 == 0) printf("\n");
			printf(" %6d", x[i * d + j]);
		}
		printf("\n\n");
	}
}

void AKNN::display(uint8_t * x, idx_t n, idx_t d)
{
	idx_t num = std::min((idx_t)3, n);
	for (idx_t i = 0; i < num; i++) {
		for (idx_t j = 0; j < d; j++) {
			if (j > 0 && j % 16 == 0) printf("\n");
			printf(" %6d", x[i * d + j]);
		}
		printf("\n\n");
	}
}

void AKNN::display(uint16_t * x, idx_t n, idx_t d)
{
	idx_t num = std::min((idx_t)3, n);
	for (idx_t i = 0; i < num; i++) {
		for (idx_t j = 0; j < d; j++) {
			if (j > 0 && j % 16 == 0) printf("\n");
			printf(" %6d", x[i * d + j]);
		}
		printf("\n");
	}
}

void AKNN::save_res(const char * resname,  idx_t * x, idx_t n, idx_t d)
{
	AKNN_IO::ivecs_write(resname, x, n, d);
}

void AKNN::clear()
{
	ScopeDeleter<float>(index->xb);
	ScopeDeleter<float>(index->xq);
	ScopeDeleter<idx_t>(index->xg);
	ScopeDeleter<std::unordered_set<idx_t>>(index->gt);
}

}	// namespace aknn
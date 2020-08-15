#include "AKNN_EX.h"
#include "VectorTransform.h"
#include <cstring>
#include <chrono>
using namespace std::chrono;
using namespace aknn;


void AKNN_EX::refine(PQparameter * params, size_t pmsize, size_t pmnum, Parameter search_params)
{
	assert_aknn(index, "No index");
	assert_aknn(pmnum > 0 && pmsize > 0, "weird params size");

	if (lack_mem) prepare(index, bname, qname, "", "");
	else prepare(index, bname, qname, knngname, gtname);
	float * xq = index->xq;

	mIndexMQ * index_mq = dynamic_cast<mIndexMQ *>(index);
	ProductQuantizer & pq = dynamic_cast<mIndexPQ *>(index)->pq;
	idx_t nbits = sizeof(code_t) * 8;
	idx_t ksub = 1 << nbits;
	idx_t bestM = params[0].M, bestKlq = params[0].klq;
	float max_recall = -1;
	fprintf(stderr, "nbits: %d\n", nbits);

	for (size_t i = 0; i < pmsize; i++) {
		idx_t M = params[i].M, klq = params[i].klq;
		fprintf(stderr, "M: %d", M);
		if (index_mq) {
			if (pmnum == 1) klq = 128;
			fprintf(stderr, " klq: %d", klq);
		}
		fprintf(stderr, "\n");

		if (index_mq) index_mq->lq.klq = klq;
		pq.M = M;
		pq.dsub = index->d / M;
		pq.ksub = ksub;

		train(index->xb, index->nb, index->d, nbits, params[i]);

		if (lack_mem) prepare(index, "", "", knngname, gtname);

		if (use_opq) {
			fprintf(stderr, "pretransform\n");
			index->xq = VectorTransform::transform(xq, index->nq, index->d, pq.R, index->d, index->d);
		}

		if (mIndexMQ1 * index_mq1 = dynamic_cast<mIndexMQ1 *>(index)) {
			fprintf(stderr, "compute residuals\n");
			index->xq = index_mq1->compute_residuals(index->xq);
		}

		fprintf(stderr, "\nsearch...\n");
		idx_t * res = new idx_t[index->nq * search_params.K];
		{
			auto start = system_clock::now();
			index->search(search_params, res);
			auto end = system_clock::now();
			float qtime = duration<float>(end - start).count();
			fprintf(stderr, "\nquery time: %.5f\n"
				"QPS: %f\n",
				qtime, index->nq * 1.0 / qtime);
		}
		float recall = index->evaluate(res, index->gt, index->nq, search_params.K);
		fprintf(stderr, "average accuracy: %f\n\n", recall);

		if (recall > max_recall) {
			max_recall = recall;
			bestM = M;
			bestKlq = klq;
		}

		if (xq != index->xq) {
			delete[] index->xq;
			index->xq = xq;
		}
		if (lack_mem) {
			delete[] index->xg;
			delete[] index->gt;
		}
		delete[] res;
	}
	fprintf(stderr, "best M: %d", bestM);
	if (index_mq) fprintf(stderr, " bestKlq: %d", bestKlq);
	fprintf(stderr, "\n");
	clear();
}


void AKNN_EX::search(Parameter * search_params, size_t pmsize, const char * outname)
{
	assert_aknn(index, "No index");
	index->verbose = verbose;

	mIndexMQ * index_mq = dynamic_cast<mIndexMQ *>(index);
	prepare(index, bname, qname, knngname, gtname);
	prepare_pq(dynamic_cast<mIndexPQ *>(index), Rname, cenname, codename);
	if (index_mq) {
		prepare_mq(index_mq, lqcenname, lqcodename);
	}

	ProductQuantizer & pq = dynamic_cast<mIndexPQ *>(index)->pq;
	idx_t nbits = sizeof(code_t) * 8;
	pq.ksub = 1 << nbits;
	fprintf(stderr, "  M: %d dsub: %d ksub: %d", pq.M, pq.dsub, pq.ksub);
	if (index_mq) {
		fprintf(stderr, " klq: %d", index_mq->lq.klq);
	}
	fprintf(stderr, "\n");
	assert_aknn(index->d == pq.M * pq.dsub, "weird PQ data");

	if (use_opq) {
		fprintf(stderr, "pretransform\n");
		float * xq = index->xq;
		ScopeDeleter<float> del(xq);
		index->xq = VectorTransform::transform(xq, index->nq, index->d, pq.R, index->d, index->d);
	}

	if (mIndexMQ1 * index_mq1 = dynamic_cast<mIndexMQ1 *>(index)) {
		fprintf(stderr, "compute residuals\n");
		float * xq = index->xq;
		ScopeDeleter<float> del(xq);
		index->xq = index_mq1->compute_residuals(xq);
	}
	
	fprintf(stderr, "\nsearch...\n");
	for (size_t i = 0; i < pmsize; i++) {
		fprintf(stderr, "E: %d	L: %d\n", search_params[i].E, search_params[i].L);
		search_o(search_params[i], outname);
	}
	clear();
}

void aknn::AKNN_EX::train(float * xt, idx_t nt, idx_t d, idx_t nbits, PQparameter param)
{
	index->verbose = verbose;
	mIndexPQ * index_pq = dynamic_cast<mIndexPQ *>(index);
	if (index_pq) index_pq->use_opq = use_opq;

	index->train(xt, nt, d, nbits, param);

	idx_t M = param.M, klq = param.klq;
	if (index_pq) {
		if (use_opq) {
			if (strlen(Rname) > 0) AKNN_IO::fvecs_write(Rname, index_pq->pq.R, d, d);
			if (verbose) {
				printf("R\n");
				display(index_pq->pq.R, index->d, index->d);
			}
		}
		if (strlen(codename) > 0)
			AKNN_IO::codes_write(codename, index_pq->pq.codes, nt, M);
		if (verbose) {
			printf("codes\n");
			display(index_pq->pq.codes, nt, M);
		}
		if (strlen(cenname) > 0)
			AKNN_IO::fvecs_write(cenname, index_pq->pq.centroids, M * index_pq->pq.ksub, index_pq->pq.dsub);
		if (verbose) {
			printf("centroids\n");
			display(index_pq->pq.centroids, M * index_pq->pq.ksub, index_pq->pq.dsub);
		}
	}
	if (mIndexMQ * index_mq = dynamic_cast<mIndexMQ *> (index)) {
		if (strlen(lqcenname) > 0)
			AKNN_IO::fvecs_write(lqcenname, index_mq->lq.centroids, klq, d);
		if (strlen(lqcodename) > 0)
			AKNN_IO::codes_write(lqcodename, index_mq->lq.codes, nt, 1);
	}
}

void AKNN_EX::prepare_pq(mIndexPQ * index_pq,
	const char * Rname, const char * cenname, const char * codename)
{
	ProductQuantizer & pq = index_pq->pq;
	idx_t mksub, n;
	if (use_opq && strlen(Rname) > 0) {
		idx_t nR, dR;
		AKNN_IO::fvecs_read(Rname, pq.R, nR, dR);
		fprintf(stderr, "[%dx%d] read R: %s\n", nR, dR, Rname);
		if (verbose) {
			printf("\nR\n");
			display(pq.R, nR, dR);
		}
		assert_aknn(index_pq->d == dR, "weird R size");
	}
	if (strlen(cenname) > 0) {
		AKNN_IO::fvecs_read(cenname, pq.centroids, mksub, pq.dsub);
		fprintf(stderr, "[%dx%d] read centroids: %s\n", mksub, pq.dsub, cenname);
		if (verbose) {
			printf("\ncentroids\n");
			display(pq.centroids, mksub, pq.dsub);
		}
	}
	if (strlen(codename) > 0) {
		AKNN_IO::codes_read(codename, pq.codes, n, pq.M);
		fprintf(stderr, "[%dx%d] read codes: %s\n", n, pq.M, codename);
		if (verbose) {
			printf("\ncodes\n");
			display(pq.codes, n, pq.M);
		}
	}
}

void AKNN_EX::prepare_mq(mIndexMQ * index_mq, const char * cenname, const char * codename)
{
	LevelQuantizer & lq = index_mq->lq;
	idx_t mksub, n, d;
	if (strlen(cenname) > 0) {
		AKNN_IO::fvecs_read(cenname, lq.centroids, lq.klq, d);
		fprintf(stderr, "[%dx%d] read lqcentroids: %s\n", lq.klq, d, cenname);
		if (verbose) {
			printf("\ncentroids\n");
			display(lq.centroids, lq.klq, d);
		}
		assert_aknn(d == index_mq->d, "weird lqcentroids size");
	}
	if (strlen(codename) > 0) {
		AKNN_IO::codes_read(codename, lq.codes, n, d);
		fprintf(stderr, "[%dx%d] read codes: %s\n", n, d, codename);
		if (verbose) {
			printf("\ncodes\n");
			display(lq.codes, n, d);
		}
		assert_aknn(n == index_mq->nb, "weird lqcentroids size");
	}
}

void AKNN_EX::clear()
{
	AKNN::clear();
	ProductQuantizer & pq = dynamic_cast<mIndexPQ *>(index)->pq;
	ScopeDeleter<float>(pq.R);
	ScopeDeleter<float>(pq.centroids);
	ScopeDeleter<code_t>(pq.codes);
}


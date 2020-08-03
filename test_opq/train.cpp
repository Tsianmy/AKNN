#include <iostream>
#include <fstream>
#include <cstring>
#include <sys/stat.h>
#include <faiss/utils/utils.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexPQ.h>
#include <faiss/VectorTransform.h>
#include "aknn_test.h"
using namespace faiss;
using namespace std;

namespace sift {
	const char * basename = "../data/sift_base.fvecs",
		*queryname = "../data/sift_query.fvecs",
		*graphname = "../data/sift_100NN_100.graph",
		*gtname = "../data/sift_groundtruth.ivecs",
		*Rname = "../data/sift_R.fvecs",
		*codename = "../data/sift_pqcodes.ivecs",
		*centroidname = "../data/sift_centab.fvecs",
		*outname = "../data/searchRes.ivecs";
}
namespace gist {
	const char * basename = "../data/gist_base.fvecs",
		*queryname = "../data/gist_query.fvecs",
		*graphname = "../data/gist_100NN_100.graph",
		*gtname = "../data/gist_groundtruth.ivecs",
		*Rname = "../data/gist_R.fvecs",
		*codename = "../data/gist_pqcodes.ivecs",
		*centroidname = "../data/gist_centab.fvecs",
		*outname = "../data/searchRes.ivecs";
}

float * fvecs_read(const char *fname, size_t *d_out, size_t *n_out)
{
	FILE *f = fopen(fname, "r");
	if (!f) {
		fprintf(stderr, "could not open %s\n", fname);
		perror("");
		abort();
	}
	int d;
	fread(&d, 1, sizeof(int), f);
	fseek(f, 0, SEEK_SET);
	struct stat st;
	fstat(fileno(f), &st);
	size_t sz = st.st_size;
	size_t n = sz / ((d + 1) * 4);

	*d_out = d; *n_out = n;
	float *x = new float[n * (d + 1)];
	size_t nr = fread(x, sizeof(float), n * (d + 1), f);

	// shift array to remove row headers
	for (size_t i = 0; i < n; i++)
		memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

	fclose(f);
	return x;
}

void save_fvecs(const char * filename, const float * x, int rows, int cols)
{
	std::ofstream out(filename, std::ios::binary);
	if (!out.is_open()) {
		printf("open file error\n");
		return;
	}
	for (int i = 0; i < rows; i++) {
		out.write((char*)& cols, 4);
		out.write((char *)(x + i * cols), cols * 4);
	}
	out.close();
}

void save_codes(const char * filename, const uint8_t * x, int rows, int cols)
{
	std::ofstream out(filename, std::ios::binary);
	if (!out.is_open()) {
		printf("open file error\n");
		return;
	}
	for (int i = 0; i < rows; i++) {
		out.write((char*)& cols, 4);
		out.write((char *)(x + i * cols), cols * sizeof(uint8_t));
	}
	out.close();
}

float test()
{
	AKNN_T aknn(gist::basename, gist::queryname, gist::graphname, gist::gtname);
	aknn.load_train(gist::Rname, gist::codename, gist::centroidname);
	//aknn.display();
	const uint K = 100;
	uint L = K, E = 100, R = 1;
	Param params(K, L, E);
	aknn.init_params(params);
	cerr << "search...\n";
	aknn.search();
	return aknn.curracc;
}

void train()
{
	fprintf(stderr, "Loading train set\n");
	size_t d, nt;
	float * xt = fvecs_read(gist::basename, &d, &nt);

	int _M = 64, _nbits = 8, bestM = 4;
	float maxacc = 0;
	IndexPreTransform * index = nullptr;

	for (int M = 4; M <= _M; M *= 2) {
		for (int nbits = _nbits; nbits <= _nbits; nbits *= 2) {
			fprintf(stderr, "M: %d	nbits: %d\n", M, nbits);
			fprintf(stderr, "  Preparing index OPQ d=%ld\n", d);
			IndexPreTransform * index_pt = nullptr;
			{
				VectorTransform * vt = new OPQMatrix(d, M);
				IndexPQ *index_pq = new IndexPQ(d, M, nbits, MetricType::METRIC_L2);
				index_pt = new IndexPreTransform(index_pq);
				index_pt->own_fields = true;
				index_pt->prepend_transform(vt);
			}

			fprintf(stderr, "  Training on %ld vectors\n", nt);
			double t0 = getmillisecs();
			float err = index_pt->mytrain(nt, xt);
			fprintf(stderr, "  Time cost: %.3f s\n", (getmillisecs() - t0) / 1000.);
			fprintf(stderr, "  PQ Distortion: obj=%g\n", err);

			OPQMatrix * opqm = dynamic_cast<faiss::OPQMatrix *>(index->chain[0]);
			float * r = opqm->A.data();
			save_fvecs(gist::Rname, r, d, d);

			faiss::IndexPQ * index_pq = dynamic_cast<faiss::IndexPQ *>(index->index);
			index_pq->add(nt, xt);
			uint8_t * codes = index_pq->codes.data();
			//printf("  %d * %d = %d codes.size: %d\n", nt, bestM, nt * bestM, index_pq->codes.size());
			save_codes(gist::codename, codes, nt, bestM);

			float * centab = index_pq->pq.centroids.data();
			//printf("  %d * %d = %d centroids.size: %d\n", bestM * index_pq->pq.ksub,
			//	index_pq->pq.dsub, index_pq->pq.ksub * d, index_pq->pq.centroids.size());
			save_fvecs(gist::centroidname, centab, bestM * index_pq->pq.ksub, index_pq->pq.dsub);

			float acc = test();

			if (acc > maxacc) {
				maxacc = acc;
				bestM = M;
			}

			delete index_pt;
		}
	}
	printf("bestM: %d maxacc: %f\n", bestM, maxacc);
}
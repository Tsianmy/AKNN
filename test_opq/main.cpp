#include <iostream>
#include <fstream>
#include <cstring>
#include <sys/stat.h>
#include <faiss/utils/utils.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexPQ.h>
#include <faiss/VectorTransform.h>
using namespace faiss;

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

void save_codes(const char * filename, uint8_t * x, int rows, int cols)
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

#define opq

int main(int argc, char** argv)
{
	printf("Loading train set\n");
	size_t d, nt;
	float * xt = fvecs_read("../data/gist_base.fvecs", &d, &nt);
	//float * xt = fvecs_read("../data/sift_learn.fvecs", &d, &nt);

	int _M = 4, _nbits = 8, bestM = 8;
	if (argc >= 2) sscanf(argv[1], "%d", &_M);
	if (argc >= 3) sscanf(argv[2], "%d", &_nbits);
	float minerr = 0x3f3f3f3f;
	IndexPreTransform * index = nullptr;

	for (int M = _M; M <= _M; M *= 2) {
		for (int nbits = _nbits; nbits <= _nbits; nbits *= 2) {
			printf("M: %d	nbits: %d\n", M, nbits);
			printf("  Preparing index OPQ d=%ld\n", d);
			IndexPreTransform * index_pt = nullptr;
			{
				IndexPQ *index_pq = new IndexPQ(d, M, nbits, MetricType::METRIC_L2);
				index_pt = new IndexPreTransform(index_pq);
				index_pt->own_fields = true;
#ifdef opq
				VectorTransform * vt = new OPQMatrix(d, M);
				index_pt->prepend_transform(vt);
#endif
			}

			printf("  Training on %ld vectors\n", nt);
			double t0 = getmillisecs();
			float err = index_pt->mytrain(nt, xt);
			printf("  Time cost: %.3f s\n", (getmillisecs() - t0) / 1000.);
			printf("  PQ Distortion: obj=%g\n", err);

			if (err < minerr) {
				minerr = err;
				bestM = M;
				index = index_pt;
			}
			else delete index_pt;
		}
	}

	printf("bestM: %d minerr: %f\n", bestM, minerr);

#ifdef opq
	OPQMatrix * opqm = dynamic_cast<faiss::OPQMatrix *>(index->chain[0]);
	float * r = opqm->A.data();
	save_fvecs("../data/gist_R.fvecs", r, d, d);
#endif
	/*{
	std::vector<double> tmpR(opqm->A.begin(), opqm->A.end());
	dynamic_cast<faiss::LinearTransform *>(opqm)->verbose = true;
	opqm->print_if_verbose("R", tmpR, d, d);
	dynamic_cast<faiss::LinearTransform *>(opqm)->verbose = false;
	}*/

	IndexPQ * index_pq = dynamic_cast<IndexPQ *>(index->index);
	index_pq->add(nt, xt);
	uint8_t * codes = index_pq->codes.data();
	printf("  %d * %d = %d codes.size: %d\n", nt, bestM, nt * bestM, index_pq->codes.size());

	int num = 3;
	for (int i = 0; i < num; i++) {
		for (int j = 0; j < bestM; j++) {
			printf("%d ", codes[i * bestM + j]);
		}
		printf("\n");
	}
	save_codes("../data/gist_pqcodes.ivecs", codes, nt, bestM);

	float * centab = index_pq->pq.centroids.data();
	printf("  %d * %d = %d centroids.size: %d\n", bestM * index_pq->pq.ksub,
		index_pq->pq.dsub, index_pq->pq.ksub * d, index_pq->pq.centroids.size());
	save_fvecs("../data/gist_centab.fvecs", centab, bestM * index_pq->pq.ksub, index_pq->pq.dsub);

	return 0;
}

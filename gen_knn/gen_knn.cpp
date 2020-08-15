#include "../include/AKNN_IO.cpp"
#include <algorithm>
using namespace aknn;

void gen_knn(const char * fname, idx_t k, idx_t * xg, idx_t n, idx_t d)
{
	FILE *f = fopen(fname, "wb");
	if (!f) {
		fprintf(stderr, "could not open %s\n", fname);
		return;
	}
	for (idx_t i = 0; i < n; i++) {
		fwrite(&k, 4, 1, f);
		fwrite(xg + i * d, sizeof(*xg), k, f);
	}

	idx_t num = std::min((idx_t)3, n);
	for (idx_t i = 0; i < num; i++) {
		for (idx_t j = 0; j < k; j++) {
			if (j > 0 && j % 17 == 0) printf("\n");
			printf(" %6d", xg[i * d + j]);
		}
		printf("\n\n");
	}

	fclose(f);
}

#define SIFT
#ifdef SIFT
const char *prefix = "../data/sift";
const char *graphname = "../data/sift_100NN_100.graph";
#elif defined GIST
const char *prefix = "../data/gist";
const char *graphname = "../data/gist_100NN_100.graph";
#endif

int main()
{
	freopen("../log.txt", "w", stdout);
	idx_t lk[] = { 50, 30, 10 };
	char buf[30];
	idx_t * xg, n, d;
	AKNN_IO io;
	io.ivecs_read(graphname, xg, n, d);
	for (idx_t i = 0; i < sizeof(lk) / sizeof(idx_t); i++) {
		fprintf(stderr, "k: %d\n\n", lk[i]);
		sprintf(buf, "_%dNN_%d.graph", lk[i], lk[i]);
		char filename[50];
		strcpy(filename, prefix);
		strcat(filename, buf);
		gen_knn(filename, lk[i], xg, n, d);
	}
	return 0;
}
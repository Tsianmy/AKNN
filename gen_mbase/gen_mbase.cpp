#include "../include/AKNN_IO.cpp"
#include <algorithm>
using namespace aknn;

void gen_mbase(const char * fname, float * x, idx_t n, idx_t d)
{
	FILE *f = fopen(fname, "wb");
	if (!f) {
		fprintf(stderr, "could not open %s\n", fname);
		return;
	}
	fwrite(&d, 4, 1, f);
	fwrite(x, 4, n * d, f);

	fclose(f);
}

namespace sift {
	const char
		*basename = "../data/sift_base.fvecs",
		*mbasename = "../data/sift_mbase.fvecs";
}
namespace gist {
	const char
		*basename = "../data/gist_base.fvecs",
		*mbasename = "../data/gist_mbase.fvecs";
}

int main()
{
	freopen("../log.txt", "w", stdout);
	AKNN_IO io;
	{//	sift
		fprintf(stderr, "read sift\n");
		float * x;
		idx_t n, d;
		io.fvecs_read(sift::basename, x, n, d);
		fprintf(stderr, "generate mbase\n");
		gen_mbase(sift::mbasename, x, n, d);
		delete[] x;
	}
	{// gist
		fprintf(stderr, "read gist\n");
		float * x;
		idx_t n, d;
		io.fvecs_read(gist::basename, x, n, d);
		fprintf(stderr, "generate mbase\n");
		gen_mbase(gist::mbasename, x, n, d);
		delete[] x;
	}
	return 0;
}
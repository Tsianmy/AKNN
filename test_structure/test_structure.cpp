#include "../include/AKNN_IO.cpp"
#include "../include/mIndex.cpp"
#include <iostream>
#include <cmath>
using namespace std;
using namespace aknn;

#define SIFT
#ifdef SIFT
const char *bname = "../data/sift_base.fvecs",
*queryname = "../data/sift_query.fvecs",
*graphname = "../data/sift_100NN_100.graph",
*gtname = "../data/sift_groundtruth.ivecs",
*outname = "../data/searchRes.ivecs";
#elif defined GIST
const char *bname = "../data/gist_base.fvecs",
*queryname = "../data/gist_query.fvecs",
*graphname = "../data/gist_100NN_100.graph",
*gtname = "../data/gist_groundtruth.ivecs",
*outname = "../data/searchRes.ivecs";
#endif

void test_groundtruth(float * xb, float * xq, idx_t * xgt, idx_t nb, idx_t k, idx_t d);
void test_knng(float * xb, idx_t * xg, idx_t nb, idx_t k, idx_t d);

int main()
{
	freopen("../log.txt", "w", stdout);
	float * xb, * xq;
	idx_t * xgt, * xg;
	idx_t nb, n, d, k;
	AKNN_IO io;
	io.fvecs_read(bname, xb, nb, d);
	io.fvecs_read(queryname, xq, n, d);
	io.ivecs_read(gtname, xgt, n, k);
	io.ivecs_read(graphname, xg, n, k);
	test_groundtruth(xb, xq, xgt, nb, k, d);
	test_knng(xb, xg, nb, k, d);
	cerr << "write to log\n";
	return 0;
}




void test_groundtruth(float * xb, float * xq, idx_t * xgt, idx_t nb, idx_t k, idx_t d)
{
	cout << "test_groundtruth\n";
	cout << "\ndistance(query0, point0..99):\n";
	unsigned v1 = 0;
	float * qv0 = xq;
	for (unsigned v2 = 0; v2 < 100; v2++) {
		float * vec = mIndex::get_ptr(xb, v2, d);
		cout << "d(q0, " << v2 << "):" << sqrt(fvec_L2sqr(qv0, vec, d)) << " ";
	}

	unordered_set<unsigned> s;

	cout << "\n\ndistance(query0, groundtruth0..9):\n";
	for (size_t i = 0; i < k; i++) {
		unsigned v2 = xgt[v1 * k + i];
		s.insert(v2);
		float * vec = mIndex::get_ptr(xb, v2, d);
		cout << "d(q0, " << v2 << "):" << sqrt(fvec_L2sqr(qv0, vec, d)) << " ";
	}
	cout << endl;

	float * gt99 = mIndex::get_ptr(xb, xgt[99], d);
	bool flag = true;
	for (unsigned v2 = 0; v2 < nb; v2++) {
		if (s.count(v2)) continue;
		float * vec = mIndex::get_ptr(xb, v2, d);
		if (sqrt(fvec_L2sqr(qv0, vec, d)) < sqrt(fvec_L2sqr(qv0, gt99, d))) {
			flag = false;
			break;
		}
	}
	if (flag == true) cout << "\ntruly nearest neighbors!\n";
	else cout << "not nearest neighbors!\n";
}

void test_knng(float * xb, idx_t * xg, idx_t nb, idx_t k, idx_t d)
{
	cout << "test_knng\n";

	cout << "\ndistance(base0, point1..99):\n";
	unsigned v1 = 0;
	float * vec0 = xb;
	for (unsigned v2 = 1; v2 < 100; v2++) {
		float * vec = mIndex::get_ptr(xb, v2, d);
		cout << "d(b0, " << v2 << "):" << sqrt(fvec_L2sqr(vec0, vec, d)) << " ";
	}
	cout << endl;

	cout << "\n\ndistance(base0, knn0..99):\n";
	unordered_set<unsigned> s;
	for (size_t i = 0; i < k; i++) {
		unsigned v2 = xg[v1 * k + i];
		s.insert(v2);
		float * vec = mIndex::get_ptr(xb, v2, d);
		cout << "d(0, " << v2 << "):" << sqrt(fvec_L2sqr(vec0, vec, d)) << " ";
	}
	cout << endl;

	float * gt99 = mIndex::get_ptr(xb, xg[99], d);
	bool flag = true;
	for (unsigned v2 = 1; v2 < nb; v2++) {
		if (s.count(v2)) continue;
		float * vec = mIndex::get_ptr(xb, v2, d);
		if (sqrt(fvec_L2sqr(vec0, vec, d)) < sqrt(fvec_L2sqr(vec0, gt99, d))) {
			flag = false;
			break;
		}
	}
	if (flag == true) cout << "\ntruly nearest neighbors!\n";
	else cout << "not nearest neighbors!\n";

}

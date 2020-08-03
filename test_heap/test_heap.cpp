#include <iostream>
#include <omp.h>
#include "aknn_test.h"
using namespace std;

namespace sift {
	char * basename = "../sift/sift_base.fvecs",
		*queryname = "../sift/sift_query.fvecs",
		*graphname = "../sift/sift_100NN_100.graph",
		*gtname = "../sift/sift_groundtruth.ivecs",
		*outname = "../sift/searchRes.ivecs";
}
namespace gist {
	char * basename = "../gist/gist_base.fvecs",
		*queryname = "../gist/gist_query.fvecs",
		*graphname = "../gist/gist_100NN_100.graph",
		*gtname = "../gist/gist_groundtruth.ivecs",
		*outname = "../gist/searchRes.ivecs";
}

int main(int argc, char** argv)
{
	omp_set_num_threads(4);
	freopen("../log.txt", "w", stdout);
	AKNN_T aknn(gist::basename, gist::queryname, gist::graphname, gist::gtname);
	aknn.display();
	const uint K = 100;
	uint L = K, E = K, R = 1;
	Param params(K, L, E);
	aknn.init_params(params);
	cerr << "search...\n";
	//aknn.search();
	for (E = 10; E <= K; E += 10) {
		cerr << "E: " << E << " L: " << L << endl;
		aknn.set_E(E);
		aknn.search();
	}
	E = K;
	aknn.set_E(E);
	for (L = 200; L <= 2500; L += 100) {
		cerr << "E: " << E << " L: " << L << endl;
		aknn.set_L(L);
		aknn.search();
	}
	/*L = 1200;
	aknn.set_L(L);
	for (R = 2; R <= 5; R++) {
	cerr << "E: " << E << " R: " << R << " L: " << L << endl;
	aknn.set_R(R);
	aknn.search();
	}*/

	/*cerr << "save...\n";
	aknn.save(sift::outname);*/

	/*int n = 100;
	fixedMinHeap<int> heap(n);
	for (int i = 0; i < n; i++) heap.push(i * 2);
	heap.display();
	cout << endl;
	for (int i = n - 1; i >= 0; i--) {
		heap.push(i * 2 + 1);
		heap.display();
		cout << endl;
	}*/
	return 0;
}
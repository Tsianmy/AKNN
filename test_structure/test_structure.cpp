#include <iostream>
#include "aknn_test.h"
using namespace std;

int main(int argc, char** argv)
{
	//freopen("../log.txt", "w", stdout);
	char * basename = "../sift/sift_base.fvecs",
		* queryname = "../sift/sift_query.fvecs",
		* graphname = "../sift/sift_100NN_100.graph",
		* gtname = "../sift/sift_groundtruth.ivecs";
	const unsigned K = 100, L = 10;
	Param params(basename, queryname, graphname, gtname, K, L);
	AKNN_T aknn(params);
	aknn.load();
	freopen("../log.txt", "w", stdout);
	aknn.test_groundtruth();
	aknn.test_knng();
	return 0;
}
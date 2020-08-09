#include <iostream>
#include "aknn_test.h"
using namespace std;

int main(int argc, char** argv)
{
	//freopen("../log.txt", "w", stdout);
	char * basename = "../data/sift_base.fvecs",
		* queryname = "../data/sift_query.fvecs",
		* graphname = "../data/sift_100NN_100.graph",
		* gtname = "../data/sift_groundtruth.ivecs",
		*outname = "../data/searchRes.ivecs";
	AKNN_T aknn(basename, queryname, graphname, gtname);
	freopen("../log.txt", "w", stdout);
	aknn.test_groundtruth();
	aknn.test_knng();
	return 0;
}
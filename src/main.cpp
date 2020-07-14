#include <iostream>
#include "../include/aknn.h"
using namespace std;

int main(int argc, char** argv)
{
	//freopen("../log.txt", "w", stdout);
	char * basename = "../sift/sift_base.fvecs",
		* queryname = "../sift/sift_query.fvecs",
		* graphname = "../sift/sift_100NN_100.graph",
		* gtname = "../sift/sift_groundtruth.ivecs",
		* outname = "../sift/searchRes.ives";
	const unsigned K = 100, L = 100, E = 10, I = 1;
	Param params(basename, queryname, graphname, gtname, outname, K, L, E, I);
	AKNN aknn(params);
	aknn.load();
	cout << "search...\n";
	aknn.search();
	cout << "save...\n";
	freopen("../log.txt", "w", stdout);
	aknn.save();
	return 0;
}
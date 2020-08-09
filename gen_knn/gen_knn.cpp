#include <iostream>
#include <cstring>
#include "../include/aknn.h"
using namespace std;

int main(int argc, char** argv)
{
	char * basename = "",
		*queryname = "",
		*graphname = "../data/gist_100NN_100.graph",
		*gtname = "",
		*outname = "";
	AKNN aknn(basename, queryname, graphname, gtname);
	uint lk[] = { 50, 30, 10 };
	char buf[30];
	freopen("../log.txt", "w", stdout);
	for (uint i = 0; i < sizeof(lk) / sizeof(uint); i++) {
		cerr << "k: " << lk[i] << endl;
		sprintf(buf, "%dNN_%d.graph", lk[i], lk[i]);
		char filename[50] = "../data/gist_";
		strcat(filename, buf);
		aknn.gen_lknn(lk[i], filename);
		cerr << endl;
	}
	aknn.display();

	return 0;
}
#ifndef AKNN_TEST_H
#define AKNN_TEST_H

#include "../include/aknn.h"

class AKNN_T : public AKNN {
public:
	AKNN_T(char *basename, char *queryname, char *graphname, char *gtname) : AKNN(basename, queryname, graphname, gtname) {}
protected:
	void get_neighbors(std::vector<uint> & res, uint q, uint initPoint, uint K, uint L, uint E);
};

#endif

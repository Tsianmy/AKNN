#ifndef AKNN_TEST_H
#define AKNN_TEST_H

#include "../include/aknn.h"

class AKNN_T: public AKNN {
public:
	AKNN_T(char *basename, char *queryname, char *graphname, char *gtname) : AKNN(basename, queryname, graphname, gtname) {}
	void test_groundtruth();
	void test_knng();
};

#endif

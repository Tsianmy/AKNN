#ifndef AKNN_TEST_H
#define AKNN_TEST_H

#include "../include/aknn.h"

class AKNN_T: public AKNN {
public:
	explicit AKNN_T(Param _params) : AKNN(_params) {}
	void test_groundtruth();
	void test_knng();
};

#endif

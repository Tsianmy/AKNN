#ifndef AKNN_TEST_H
#define AKNN_TEST_H

#include "../include/aknn.h"

class AKNN_T : public AKNN {
public:
	AKNN_T(const char *basename, const char *queryname, const char *graphname, const char *gtname);
	void search();
	void display();
	void load(const char *basename, const char *queryname, const char *graphname, const char *gtname);
	~AKNN_T();
protected:
	void get_neighbors(std::vector<uint> & res, uint q, uint initPoint, uint K, uint L, uint E);
	void load_base(const char * basename);
	float * get_bptr(uint index);
};

#endif

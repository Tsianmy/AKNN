#ifndef AKNN_TEST_H
#define AKNN_TEST_H

#include "../include/aknn.h"

class AKNN_T : public AKNN {
public:
	AKNN_T(char *basename, char *queryname, char *graphname, char *gtname);
	void search();
	void display();
	void load(char *basename, char *queryname, char *graphname, char *gtname);
	~AKNN_T();
protected:
	void get_neighbors(std::vector<uint> & res, uint q, uint initPoint, uint K, uint L, uint E);
	void load_base(char * basename);
	float * get_bptr(uint index);
};

#endif

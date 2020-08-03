#ifndef AKNN_TEST_H
#define AKNN_TEST_H

#include "../include/aknn.h"
#include "../include/pq.h"

class AKNN_T : public AKNN {
public:
	float curracc;

	AKNN_T(const char *basename, const char *queryname, const char *graphname, const char *gtname);
	void load(const char *basename, const char *queryname, const char *graphname, const char *gtname);
	void load_train(const char * Rname, const char * codename, const char * centroidname);
	void search();

protected:
	ProductQuantizer pq;

	void load_data(const char* filename, uint8_t *& data, uint& num, uint& dim);
	void display_data(uint8_t* data, uint num, uint dim);
	void get_neighbors(std::vector<uint> & res, std::vector<float> & distance_table,
		uint q, uint initPoint, uint K, uint L, uint E);
	void get_code(std::vector<uint8_t> & code, uint id);
	float compute_PQdistance(uint pqid, std::vector<float> & distance_table);
	void compute_dis_table(std::vector<float> & distance_table, uint q);
};

#endif

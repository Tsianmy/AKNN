#ifndef AKNN_TEST_H
#define AKNN_TEST_H

#include "../include/aknn.h"
#include "../include/pq.h"

class AKNN_T : public AKNN {
public:
	float curracc;

	AKNN_T(const char *basename, const char *queryname, const char *graphname, const char *gtname);
	void load(const char *basename, const char *queryname, const char *graphname, const char *gtname);
	void load_train(const char * Rname, const char * coarsename, const char * idxname, const char * codename, const char * centroidname);
	void search();
	void train(const char * basename, const char * coarsename, const char * idxname, const char * centroidname, const char * codename);

protected:
	ProductQuantizerResi pq;

	void load_data(const char* filename, uint8_t *& data, uint& num, uint& dim);
	void load_data(const char* filename, uint16_t *& data, uint& num, uint& dim);
	void display_data(uint8_t* data, uint num, uint dim);
	void save_fvecs(const char * filename, const float * x, int rows, int cols);
	void save_codes(const char * filename, uint8_t * x, int rows, int cols);
	void save_codes(const char * filename, uint16_t * x, int rows, int cols);

	void get_neighbors(std::vector<uint> & res, std::vector<float> & distance_table,
		uint q, uint initPoint, uint K, uint L, uint E);
	void get_neighbors2(std::vector<uint> & res, uint q, uint initPoint,
		uint K, uint L, uint E);
	void get_code(std::vector<uint8_t> & code, uint id);

	void residual(float * vec1, float * vec2, uint dim);
	void compute_residual(float * ptr_q);
	void add_residual(float *vec1, float *vec2, uint dim);
	void mul_beta(float *vec1, float *vec2, uint dim);

	float compute_PQdistance(uint pqid, std::vector<float> & distance_table);
	void compute_dis_table(std::vector<float> & distance_table, uint q);
	void compute_dis_table2(std::vector<float> & distance_table, uint q);
	float compute_newdistance(uint pqid, uint q);

	static float L2_sqr(float * vec1, float * vec2, uint dim);
};

#endif

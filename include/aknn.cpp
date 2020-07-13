#include "aknn.h"
#include <fstream>
#include <iostream>
#include <set>
using namespace std;

void AKNN::load()
{
	if (strlen(params.basename) > 0) {
		cout << "read base " << params.basename << endl;
		load_float(params.basename, base.data, base.num, base.dim);
		cout << "dimention: " << base.dim << endl
			<< "points' num: " << base.num << endl;
	}
	if (strlen(params.queryname) > 0) {
		cout << "read query " << params.queryname << endl;
		load_float(params.queryname, query.data, query.num, query.dim);
		cout << "dimention: " << query.dim << endl
			<< "points' num: " << query.num << endl;
	}
	if (strlen(params.gtname) > 0) {
		cout << "read groundtruth " << params.gtname << endl;
		load_int(params.gtname, groundtruth.data, groundtruth.num, groundtruth.dim);
		cout << "dimention: " << groundtruth.dim << endl
			<< "points' num: " << groundtruth.num << endl;
	}
	if (strlen(params.graphname) > 0) {
		cout << "read graph " << params.graphname << endl;
		load_int(params.graphname, graph.data, graph.num, graph.dim);
		cout << "dimention: " << graph.dim << endl
			<< "points' num: " << graph.num << endl;
	}
}

void AKNN::load_float(char * filename, float *& data, unsigned & num, unsigned & dim)
{
	ifstream in(filename, ios::binary);
	if (!in.is_open()) {
		cout << "open file error" << endl;
		exit(-1);
	}
	in.read((char*)&dim, 4);
	in.seekg(0, ios::end);
	ios::pos_type ss = in.tellg();
	size_t fsize = (size_t)ss;
	num = (unsigned)(fsize / (dim + 1) / 4);
	data = new float[(size_t)num * (size_t)dim];

	in.seekg(0, ios::beg);
	for (size_t i = 0; i < num; i++) {
		in.seekg(4, ios::cur);
		in.read((char*)(data + i * dim), dim * 4);
	}
	in.close();
}

void AKNN::load_int(char * filename, int *& data, unsigned & num, unsigned & dim)
{
	ifstream in(filename, ios::binary);
	if (!in.is_open()) {
		cout << "open file error" << endl;
		exit(-1);
	}
	in.read((char*)&dim, 4);
	in.seekg(0, ios::end);
	ios::pos_type ss = in.tellg();
	size_t fsize = (size_t)ss;
	num = (unsigned)(fsize / (dim + 1) / 4);
	data = new int[(size_t)num * (size_t)dim];

	in.seekg(0, ios::beg);
	for (size_t i = 0; i < num; i++) {
		in.seekg(4, ios::cur);
		in.read((char*)(data + i * dim), dim * 4);
	}
	in.close();
}

float AKNN::distance(float * vec1, float * vec2, unsigned dim)
{
	float sum = 0;
	for (size_t i = 0; i < dim; i++) {
		sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
	}
	return sqrtf(sum / dim);
}

float * AKNN::get_ptr(float * begin, const unsigned index, const unsigned dim)
{
	return begin + index * dim;
}

AKNN::~AKNN()
{
	if (base.data != nullptr) {
		delete base.data;
		base.data = nullptr;
	}
	if (query.data != nullptr) {
		delete query.data;
		query.data = nullptr;
	}
	if (graph.data != nullptr) {
		delete graph.data;
		graph.data = nullptr;
	}
	if (groundtruth.data != nullptr) {
		delete groundtruth.data;
		groundtruth.data = nullptr;
	}
}
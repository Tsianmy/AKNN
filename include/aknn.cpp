#include "aknn.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <unordered_set>
#include <ctime>
#include <assert.h>
using namespace std;

void AKNN::load()
{
	if (strlen(params.basename) > 0) {
		cout << "read base " << params.basename << endl;
		load_data(params.basename, base.data, base.num, base.dim);
		cout << "dimention: " << base.dim << endl
			<< "points' num: " << base.num << endl;
	}
	if (strlen(params.queryname) > 0) {
		cout << "read query " << params.queryname << endl;
		load_data(params.queryname, query.data, query.num, query.dim);
		cout << "dimention: " << query.dim << endl
			<< "points' num: " << query.num << endl;
	}
	if (strlen(params.gtname) > 0) {
		cout << "read groundtruth " << params.gtname << endl;
		load_data(params.gtname, groundtruth.data, groundtruth.num, groundtruth.dim);
		cout << "dimention: " << groundtruth.dim << endl
			<< "points' num: " << groundtruth.num << endl;
	}
	if (strlen(params.graphname) > 0) {
		cout << "read graph " << params.graphname << endl;
		load_data(params.graphname, graph.data, graph.num, graph.dim);
		cout << "dimention: " << graph.dim << endl
			<< "points' num: " << graph.num << endl;
	}
}

void AKNN::load_data(char * filename, float *& data, uint & num, uint & dim)
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

void AKNN::load_data(char * filename, int *& data, uint & num, uint & dim)
{
	ifstream in(filename, ios::binary);
	if (!in.is_open()) {
		cout << "open file error" << endl;
		exit(-1);
	}
	in.read((char*)&dim, 4);
	in.seekg(0, ios::end);
	ios::pos_type ss = in.tellg();
	uint fsize = (uint)ss;
	num = fsize / (dim + 1) / 4;
	data = new int[(size_t)num * (size_t)dim];

	in.seekg(0, ios::beg);
	for (uint i = 0; i < num; i++) {
		in.seekg(4, ios::cur);
		in.read((char*)(data + i * dim), dim * 4);
	}
	in.close();
}

float AKNN::distance(float * vec1, float * vec2, uint dim)
{
	float sum = 0;
	for (uint i = 0; i < dim; i++) {
		sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
	}
	return sqrtf(sum / dim);
}

float * AKNN::get_ptr(float * begin, const uint index, const uint dim)
{
	return begin + index * dim;
}

int * AKNN::get_ptr(int * begin, const uint index, const uint dim)
{
	return begin + index * dim;
}

void AKNN::search()
{
	if (searchRes.data != nullptr) {
		delete[]searchRes.data;
		searchRes.data = nullptr;
	}
	searchRes.dim = params.K;
	searchRes.num = query.num;
	
	vector<unordered_set<uint>> gtset(groundtruth.num);
	for (uint i = 0; i < groundtruth.num; i++) {
		for (uint j = i * groundtruth.dim; j < (i + 1) * groundtruth.dim; j++) {
			gtset[i].insert(groundtruth.data[j]);
		}
	}

	float qtime = 0;
	srand((int)time(0));
	uint maxAcc = 0;
	for (uint iter = 0; iter < params.I; iter++) {
		cout << "iteration: " << iter << endl;

		int * tempdata = new int[params.K * query.num];
		uint avgAcc = 0;
		uint initPoint = rand() % searchRes.num;
		//cout << "  init: " << initPoint << endl;

		// search neighbors
		clock_t start = clock();
		vector<uint> neighbors;
		for (uint i = 0; i < searchRes.num; i++) {
			//cout << ".";
			get_neighbors(neighbors, i, initPoint, params.K, params.L, params.E);
			for (uint j = 0; j < params.K && j < neighbors.size(); j++) {
				tempdata[i * params.K + j] = neighbors[j];
			}
		}
		qtime += (clock() - start) * 1.0 / CLOCKS_PER_SEC;

		// accuracy
		for (uint i = 0; i < searchRes.num; i++) {
			for (uint j = 0; j < params.K; j++) {
				uint id = tempdata[i * params.K + j];
				if (gtset[i].count(id)) avgAcc++;
			}
		}

		cout << "avg acc: " << avgAcc / (1.0 * searchRes.num * searchRes.dim) << endl;

		if (avgAcc > maxAcc) {
			maxAcc = avgAcc;
			if (searchRes.data != nullptr) delete[]searchRes.data;
			searchRes.data = tempdata;
		}
		else delete[]tempdata;
	}

	cout << "\nquery time: " << qtime << "\nQPS: " << params.I * 5 * 1.0 / qtime
		<< "\nmax average accuracy: " << maxAcc / (1.0 * searchRes.num * searchRes.dim) << endl << endl;
}

void AKNN::get_neighbors(vector<uint> & res, uint q, uint initPoint, uint K, uint L, uint E)
{
	float * ptr_q = get_ptr(query.data, q, query.dim);
	float initDis = distance(get_ptr(base.data, initPoint, base.dim), ptr_q, base.dim);

	vector<Neighbor> S;
	S.push_back(Neighbor(initPoint, initDis));
	//cout << "start p: " << initPoint << " dis: " << initDis << endl;

	vector<bool> vis(base.num, false), in(base.num, false);
	uint i = 0;
	while (i < L) {
		// find the index of the first unchecked node in S
		uint j;
		for (j = 0; j < S.size(); j++) {
			if (!vis[S[j].id]) {
				i = j;
				break;
			}
		}
		if (j == S.size()) break;

		uint id = S[i].id;
		vis[id] = true;
		//cout << "first unchecked i: " << i << " id: " << id << " d(q, " << id << "): " << S[i].distance << endl;
		
		int * neighbors = get_ptr(graph.data, id, graph.dim);
		assert(E <= graph.dim);
		for (j = 0; j < E; j++) {
			if (vis[neighbors[j]] || in[neighbors[j]]) continue;
			float * pj = get_ptr(base.data, neighbors[j], base.dim);
			float dis = distance(pj, ptr_q, base.dim);
			S.push_back(Neighbor(neighbors[j], dis));
			in[neighbors[j]] = true;
			//cout << "  neighbor: " << neighbors[j] << " dis: " << distance(pj, ptr_q, base.dim) << endl;
		}
		sort(S.begin(), S.end());
		if (S.size() > L) S.resize(L);
	}
	res.clear();
	for (i = 0; i < K && i < S.size(); i++) res.push_back(S[i].id);
}

void AKNN::save()
{
	ofstream out(params.outputname, ios::binary);
	if (!out.is_open()) {
		cout << "open file error" << endl;
		exit(-1);
	}
	cout << "search result:\n";
	for (uint i = 0; i < searchRes.num; i++) {
		out.write((char*)& searchRes.dim, 4);
		out.write((char *)(searchRes.data + i * searchRes.dim), searchRes.dim * 4);
		for (uint j = i * searchRes.dim; j < (i + 1) * searchRes.dim; j++) {
			if ((j - i * searchRes.dim) && (j - i * searchRes.dim) % 17 == 0) cout << endl;
			cout << setw(7) << searchRes.data[j];
		}
		cout << endl << endl;
	}
	out.close();

	cout << "groundtruth:\n";
	for (uint i = 0; i < searchRes.num; i++) {
		for (uint j = i * groundtruth.dim; j < (i + 1) * groundtruth.dim; j++) {
			if ((j - i * groundtruth.dim) && (j - i * groundtruth.dim) % 17 == 0) cout << endl;
			cout << setw(7) << groundtruth.data[j];
		}
		cout << endl << endl;
	}
}

AKNN::~AKNN()
{
	if (base.data != nullptr) {
		delete []base.data;
		base.data = nullptr;
	}
	if (query.data != nullptr) {
		delete []query.data;
		query.data = nullptr;
	}
	if (graph.data != nullptr) {
		delete []graph.data;
		graph.data = nullptr;
	}
	if (groundtruth.data != nullptr) {
		delete []groundtruth.data;
		groundtruth.data = nullptr;
	}
}
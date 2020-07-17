#include "aknn.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include <cstring>
#include <cmath>
#include <assert.h>
#include <omp.h>
using namespace std;

AKNN::AKNN(char * basename, char * queryname, char * graphname, char * gtname)
{
	ready = false;
	load(basename, queryname, graphname, gtname);
}

void AKNN::init_params(Param _params)
{
	if (_params.K > groundtruth.dim || _params.E > graph.dim || _params.L < _params.K) {
		cerr << "params out of range\n";
		exit(1);
	}
	params = _params;
	ready = true;
	gtset.resize(groundtruth.num);
	for (uint i = 0; i < groundtruth.num; i++) {
		for (uint j = i * groundtruth.dim; j < i * groundtruth.dim + params.K; j++) {
			gtset[i].insert(groundtruth.data[j]);
		}
	}
}

void AKNN::set_E(uint E)
{
	if (E > graph.dim) exit(1);
	params.E = E;
}

void AKNN::set_R(uint R)
{
	params.R = R;
}

void AKNN::set_L(uint L)
{
	if (L < params.K) exit(1);
	params.L = L;
}

void AKNN::display()
{
	if (base.data != nullptr) {
		cout << "base\n";
		display_data(base.data, base.num, base.dim);
	}
	if (query.data != nullptr) {
		cout << "query\n";
		display_data(query.data, query.num, query.dim);
	}
	if (groundtruth.data != nullptr) {
		cout << "groundtruth\n";
		display_data(groundtruth.data, groundtruth.num, groundtruth.dim);
	}
	if (graph.data != nullptr) {
		cout << "graph\n";
		display_data(graph.data, graph.num, graph.dim);
	}
}

void AKNN::display_data(float * data, uint num, uint dim)
{
	num = 3;
	for (uint i = 0; i < num; i++) {
		for (uint j = i * dim; j < (i + 1) *dim; j++) {
			if ((j - i * dim) && (j - i * dim) % 17 == 0) cout << endl;
			cout << setw(7) << data[j];
		}
		cout << endl << endl;
	}
	cout << "...\n\n";
}

void AKNN::display_data(int * data, uint num, uint dim)
{
	num = 3;
	for (uint i = 0; i < num; i++) {
		for (uint j = i * dim; j < (i + 1) *dim; j++) {
			if ((j - i * dim) && (j - i * dim) % 17 == 0) cout << endl;
			cout << setw(7) << data[j];
		}
		cout << endl << endl;
	}
	cout << "...\n\n";
}

void AKNN::load(char *basename, char *queryname, char *graphname, char *gtname)
{
	clock_t start = clock();
	if (strlen(basename) > 0) {
		cerr << "read base " << basename << endl;
		load_data(basename, base.data, base.num, base.dim);
		cerr << "dimention: " << base.dim << endl
			<< "points' num: " << base.num << endl;
	}
	if (strlen(queryname) > 0) {
		cerr << "read query " << queryname << endl;
		load_data(queryname, query.data, query.num, query.dim);
		cerr << "dimention: " << query.dim << endl
			<< "points' num: " << query.num << endl;
	}
	if (strlen(gtname) > 0) {
		cerr << "read groundtruth " << gtname << endl;
		load_data(gtname, groundtruth.data, groundtruth.num, groundtruth.dim);
		//delete[]groundtruth.data;
		cerr << "dimention: " << groundtruth.dim << endl
			<< "points' num: " << groundtruth.num << endl;
	}
	if (strlen(graphname) > 0) {
		cerr << "read graph " << graphname << endl;
		load_data(graphname, graph.data, graph.num, graph.dim);
		cerr << "dimention: " << graph.dim << endl
			<< "points' num: " << graph.num << endl;
	}
	cerr << "cost " << (clock() - start) * 1.0 / CLOCKS_PER_SEC << " s\n";
}

void AKNN::load_data(char * filename, float *& data, uint & num, uint & dim)
{
	ifstream in(filename, ios::binary);
	if (!in.is_open()) {
		cerr << "open file error" << endl;
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

void AKNN::insert_pool(vector<Neighbor>& pool, Neighbor p, size_t end)
{
	size_t i = end;
	while (i > 0 && p.distance < pool[i - 1].distance) {
		pool[i] = pool[i - 1];
		i--;
	}
	pool[i] = p;
}

float AKNN::distance(float * vec1, float * vec2, uint dim)
{
	float sum = 0;
	for (uint i = 0; i < dim; i++) {
		sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
	}
	return sqrt(sum / dim);
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
	if (!ready) exit(1);
	if (searchRes.data != nullptr) delete[]searchRes.data;
	searchRes.data = new int[params.K * query.num];
	searchRes.dim = params.K;
	searchRes.num = query.num;

	float qtime = 0;
	srand((uint)time(0));
	uint acc = 0;
	//vector<bool> complete(searchRes.num);

	clock_t start = clock();
#pragma omp parallel for reduction(+:acc)
	for (int q = 0; q < searchRes.num; q++) {
		vector<uint> neighbors;
		int maxc = -1;
		for (uint r = 0; r < params.R; r++) {
			vector<uint> tempn(params.K);
			uint initPoint = rand() % base.num;
			get_neighbors(tempn, q, initPoint, params.K, params.L, params.E);
			int cnt = 0;
			for (uint i = 0; i < params.K; i++) {
				if (gtset[q].count(tempn[i])) cnt++;
			}
			if (cnt > maxc) {
				maxc = cnt;
				neighbors = tempn;
			}
		}
		acc += maxc;
		for (uint i = 0; i < params.K; i++) {
			searchRes.data[q * params.K + i] = neighbors[i];
		}
	}
	qtime += (clock() - start) * 1.0 / CLOCKS_PER_SEC;

	cerr << "\nquery time: " << qtime << "\nQPS: " << searchRes.num * 1.0 / qtime
		<< "\naverage accuracy: " << acc / (1.0 * searchRes.num * searchRes.dim) << endl << endl;
}

void AKNN::get_neighbors(vector<uint> & res, uint q, uint initPoint, uint K, uint L, uint E)
{
	float * ptr_q = get_ptr(query.data, q, query.dim);
	float initDis = distance(get_ptr(base.data, initPoint, base.dim), ptr_q, base.dim);

	vector<Neighbor> S;
	S.push_back(Neighbor(initPoint, initDis));

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
		
		int * neighbors = get_ptr(graph.data, id, graph.dim);
		size_t end = min(S.size(), (size_t)L), sz = min(S.size() + E, (size_t)L + 1);
		S.resize(sz);
		for (j = 0; j < E; j++) {
			if (vis[neighbors[j]] || in[neighbors[j]]) continue;
			float * pj = get_ptr(base.data, neighbors[j], base.dim);
			float dis = distance(pj, ptr_q, base.dim);
			insert_pool(S, Neighbor(neighbors[j], dis), end);
			end = min(end + 1, sz - 1);
			//S.push_back(Neighbor(neighbors[j], dis));
			in[neighbors[j]] = true;
		}
		//sort(S.begin(), S.end());
		//if (S.size() > L) S.resize(L);
	}
	for (i = 0; i < K && i < S.size(); i++) res[i] = S[i].id;
}

void AKNN::save(char *outputname)
{
	ofstream out(outputname, ios::binary);
	if (!out.is_open()) {
		cerr << "open file error" << endl;
		exit(-1);
	}
	cout << "search result:\n";
	uint num = 3;
	for (uint i = 0; i < num; i++) {
		out.write((char*)& searchRes.dim, 4);
		out.write((char *)(searchRes.data + i * searchRes.dim), searchRes.dim * 4);
		for (uint j = i * searchRes.dim; j < (i + 1) * searchRes.dim; j++) {
			if ((j - i * searchRes.dim) && (j - i * searchRes.dim) % 17 == 0) cout << endl;
			cout << setw(7) << searchRes.data[j];
		}
		cout << endl << endl;
	}
	cout << "...\n\n";
	out.close();

	/*cout << "groundtruth:\n";
	for (uint i = 0; i < searchRes.num; i++) {
		for (uint j = i * groundtruth.dim; j < i * groundtruth.dim + searchRes.dim; j++) {
			if ((j - i * groundtruth.dim) && (j - i * groundtruth.dim) % 17 == 0) cout << endl;
			cout << setw(7) << groundtruth.data[j];
		}
		cout << endl << endl;
	}*/
}

void AKNN::gen_lknn(uint K, char * lknnName)
{
	assert(K <= graph.dim);
	ofstream out(lknnName, ios::binary);
	if (!out.is_open()) {
		cerr << "open file error" << endl;
		exit(-1);
	}
	for (uint i = 0; i < graph.num; i++) {
		out.write((char*)& K, 4);
		out.write((char *)(graph.data + i * graph.dim), K * 4);
		if (i < 3) {
			for (uint j = i * graph.dim; j < i * graph.dim + K; j++) {
				if ((j - i * graph.dim) && (j - i * graph.dim) % 17 == 0) cout << endl;
				cout << setw(7) << graph.data[j];
			}
			cout << endl << endl;
		}
	}
	out.close();
	cout << "...\n\n";
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
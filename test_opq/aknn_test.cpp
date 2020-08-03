#include "aknn_test.h"
#include "../include/aknn.cpp"

AKNN_T::AKNN_T(const char *basename, const char *queryname, const char *graphname, const char *gtname)
{
	cerr << "call aknn_t.\n";
	ready = false;
	load(basename, queryname, graphname, gtname);
}

void AKNN_T::load_data(const char * filename, uint8_t *& data, uint & num, uint & dim)
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
	num = (unsigned)(fsize / (dim * sizeof(uint8_t) + 4));
	data = new uint8_t[(size_t)num * (size_t)dim];

	in.seekg(0, ios::beg);
	for (size_t i = 0; i < num; i++) {
		in.seekg(4, ios::cur);
		in.read((char*)(data + i * dim), dim * sizeof(uint8_t));
	}
	in.close();
}

void AKNN_T::display_data(uint8_t * data, uint num, uint dim)
{
	num = 3;
	for (uint i = 0; i < num; i++) {
		for (uint j = i * dim; j < (i + 1) *dim; j++) {
			if ((j - i * dim) && (j - i * dim) % 17 == 0) cout << endl;
			cout << setw(7) << (int)data[j];
		}
		cout << "\n\n";
	}
	cout << "...\n\n";
}

void AKNN_T::load(const char *basename, const char *queryname, const char *graphname, const char *gtname)
{
	cerr << "call aknn_t load.\n";
	clock_t start = clock();
	if (strlen(queryname) > 0) {
		cerr << "read query " << queryname << endl;
		AKNN::load_data(queryname, query.data, query.num, query.dim);
		cerr << "dimention: " << query.dim << endl
			<< "points' num: " << query.num << endl;
	}
	if (strlen(gtname) > 0) {
		cerr << "read groundtruth " << gtname << endl;
		AKNN::load_data(gtname, groundtruth.data, groundtruth.num, groundtruth.dim);
		//delete[]groundtruth.data;
		cerr << "dimention: " << groundtruth.dim << endl
			<< "points' num: " << groundtruth.num << endl;
	}
	if (strlen(graphname) > 0) {
		cerr << "read graph " << graphname << endl;
		AKNN::load_data(graphname, graph.data, graph.num, graph.dim);
		cerr << "dimention: " << graph.dim << endl
			<< "points' num: " << graph.num << endl;
	}
	cerr << "cost " << (clock() - start) * 1.0 / CLOCKS_PER_SEC << " s\n";
}

void AKNN_T::load_train(const char * Rname, const char * codename, const char * centroidname)
{
	{
		float * p = nullptr;
		AKNN::load_data(Rname, p, pq.d, pq.d);
		vector<float> tv(p, p + pq.d * pq.d);
		pq.R.swap(tv);
		delete[] p;
	}
	{
		uint8_t * p = nullptr;
		load_data(codename, p, pq.n, pq.M);
		vector<uint8_t> tv(p, p + pq.n * pq.M);
		pq.codes.swap(tv);
		delete[] p;
		cout << "codes\n";
		display_data(pq.codes.data(), pq.n, pq.M);
	}
	{
		float * p = nullptr;
		uint n;
		AKNN::load_data(centroidname, p, n, pq.dsub);
		pq.ksub = n / pq.M;
		vector<float> tv(p, p + n * pq.dsub);
		pq.centroids.swap(tv);
		delete[] p;
		cout << "centroids\n";
		AKNN::display_data(pq.centroids.data(), n, pq.dsub);
	}
	base.num = pq.n;
	base.dim = pq.d;
	cerr << "M: " << pq.M << " dsub: " << pq.dsub << " d: " << pq.d << " ksub: " << pq.ksub << endl;
	if (pq.dsub * pq.M != pq.d) {
		cerr << "pq.dsub * pq.M != pq.d\n";
		exit(-1);
	}
}

void AKNN_T::compute_dis_table(std::vector<float>& distance_table, uint q)
{
	float * ptr_q = get_ptr(query.data, q, query.dim);
	for (uint m = 0; m < pq.M; m++) {
		float * subp1 = ptr_q + m * pq.dsub;
		for (uint cid = 0; cid < pq.ksub; cid++) {
			float * subp2 = pq.centroids.data() + (m * pq.ksub + cid) * pq.dsub;
			distance_table[m * pq.ksub + cid] = distance(subp1, subp2, pq.dsub);
		}
	}
}

void AKNN_T::search()
{
	cerr << "call aknn_t search.\n";
	if (!ready) exit(1);
	if (searchRes.data != nullptr) delete[]searchRes.data;
	searchRes.data = new int[params.K * query.num];
	searchRes.dim = params.K;
	searchRes.num = query.num;

	float qtime = 0;
	srand((uint)time(0));
	uint acc = 0;

	auto start = chrono::system_clock::now();
#pragma omp parallel for reduction(+:acc)
	for (int q = 0; q < searchRes.num; q++) {
		// Mod *******
		vector<float> distance_table(pq.M * pq.ksub);
		compute_dis_table(distance_table, q);
		// *******

		vector<uint> neighbors;
		int maxc = -1;
		for (uint r = 0; r < params.R; r++) {
			vector<uint> tempn(params.K);
			uint initPoint = rand() % base.num;
			get_neighbors(tempn, distance_table, q, initPoint, params.K, params.L, params.E);
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
	auto end = chrono::system_clock::now();
	qtime += chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;

	curracc = acc / (1.0 * searchRes.num * searchRes.dim);
	cerr << "\nquery time: " << qtime << "\nQPS: " << searchRes.num * 1.0 / qtime
		<< "\naverage accuracy: " << acc / (1.0 * searchRes.num * searchRes.dim) << endl << endl;
}

void AKNN_T::get_code(std::vector<uint8_t>& code, uint id)
{
	code.assign(pq.codes.data() + id * pq.M, pq.codes.data() + id * pq.M + pq.M);
}

float AKNN_T::compute_PQdistance(uint pqid, std::vector<float> & distance_table)
{
	vector<uint8_t> code;
	get_code(code, pqid);
	float dis = 0;
	uint blockn = pq.M / 4, rem = pq.M % 4, m = 0;
	for (uint i = 0; i < blockn; i++) {
		dis += distance_table[m * pq.ksub + code[m]]; m++;
		dis += distance_table[m * pq.ksub + code[m]]; m++;
		dis += distance_table[m * pq.ksub + code[m]]; m++;
		dis += distance_table[m * pq.ksub + code[m]]; m++;
	}
	for (uint i = 0; i < rem; i++) {
		dis += distance_table[m * pq.ksub + code[m]]; m++;
	}

	return dis;
}

void AKNN_T::get_neighbors(std::vector<uint> & res, std::vector<float> & distance_table,
	uint q, uint initPoint, uint K, uint L, uint E)
{
	// Mod *******
	float initDis = compute_PQdistance(initPoint, distance_table);
	// *******
	vector<Neighbor> S(L + 1);
	S[0] = Neighbor(initPoint, initDis);
	size_t end = 0;

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

		// add neighbors to the candidate pool
		int * neighbors = get_ptr(graph.data, id, graph.dim);
		for (j = 0; j < E; j++) {
			if (vis[neighbors[j]] || in[neighbors[j]]) continue;
			// Mod *******
			float dis = compute_PQdistance(neighbors[j], distance_table);
			// *******
			insert_pool(S, Neighbor(neighbors[j], dis), end);
			end = min(end + 1, (size_t)L);
			in[neighbors[j]] = true;
		}
	}
	for (i = 0; i < K && i < S.size(); i++) res[i] = S[i].id;
}

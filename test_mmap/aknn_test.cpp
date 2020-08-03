#include "aknn_test.h"
#include "../include/aknn.cpp"
#ifdef __linux__
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#else
#include <windows.h>
#endif

AKNN_T::AKNN_T(char *basename, char *queryname, char *graphname, char *gtname)
{
	load(basename, queryname, graphname, gtname);
}

void AKNN_T::display()
{
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


void AKNN_T::load_base(char *basename)
{
	cerr << "load_base called\n";
#ifdef __linux__
	int fd = open(basename, O_RDONLY);
	if(fd == -1){
		cerr << "open file error" << endl;
		exit(-1);
	}
	struct stat statbuf;
	stat(basename,&statbuf);
	int64_t size = statbuf.st_size;
	base.data = (float *)mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
	cerr << base.data[1] << endl;
	memcpy(&base.dim, base.data, 4);
	base.num = size / (4 * base.dim + 4);
#else
	HANDLE dumpFileDescriptor = CreateFileA(basename, GENERIC_READ | GENERIC_WRITE,
		FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	DWORD size = GetFileSize(dumpFileDescriptor, NULL);
	HANDLE fileMappingObject = CreateFileMapping(dumpFileDescriptor, NULL, PAGE_READWRITE,
		0, 0, NULL);
	base.data =(float *)MapViewOfFile(fileMappingObject, FILE_MAP_READ, 0, 0, 0);
	memcpy(&base.dim, base.data, 4);
	base.num = size / (4 * base.dim + 4);
#endif
}

void AKNN_T::load(char *basename, char *queryname, char *graphname, char *gtname)
{
	clock_t start = clock();
	if (strlen(basename) > 0) {
		cerr << "read base " << basename << endl;
		load_base(basename);
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

float * AKNN_T::get_bptr(uint index)
{
	return base.data + index * (base.dim + 1) + 1;
}

void AKNN_T::search()
{
	cerr << "aknnt search called\n";
	if (!ready) exit(1);
	if (searchRes.data != nullptr) delete[]searchRes.data;
	searchRes.data = new int[params.K * query.num];
	searchRes.dim = params.K;
	searchRes.num = query.num;

	float qtime = 0;
	srand((uint)time(0));
	uint acc = 0;

	//clock_t start = clock();
	auto start = chrono::system_clock::now();
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
	auto end = chrono::system_clock::now();
	qtime += chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;

	cerr << "\nquery time: " << qtime << "\nQPS: " << searchRes.num * 1.0 / qtime
		<< "\naverage accuracy: " << acc / (1.0 * searchRes.num * searchRes.dim) << endl << endl;
}

void AKNN_T::get_neighbors(vector<uint> & res, uint q, uint initPoint, uint K, uint L, uint E)
{
	//cerr << "get_neighbors called\n";
	float * ptr_q = get_ptr(query.data, q, query.dim);
	float initDis = distance(get_bptr(initPoint), ptr_q, base.dim);

	//vector<Neighbor> S;
	//S.push_back(Neighbor(initPoint, initDis));
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

		//if (q == 0) cout << i << endl;
		
		int * neighbors = get_ptr(graph.data, id, graph.dim);
		for (j = 0; j < E; j++) {
			if (vis[neighbors[j]] || in[neighbors[j]]) continue;
			float * pj = get_bptr(neighbors[j]);
			float dis = distance(pj, ptr_q, base.dim);
			insert_pool(S, Neighbor(neighbors[j], dis), end);
			end = min(end + 1, (size_t)L);
			//S.push_back(Neighbor(neighbors[j], dis));
			in[neighbors[j]] = true;
		}
		//sort(S.begin(), S.end());
		//if (S.size() > L) S.resize(L);
		/*if (q == 0) {
			cout << "size: " << S.size() << endl;
			for (j = 0; j < S.size(); j++) cout << S[j].id << ": " << S[j].distance << endl;
			cout << endl;
		}*/
	}
	for (i = 0; i < K && i < S.size(); i++) res[i] = S[i].id;
}

AKNN_T::~AKNN_T()
{
#ifdef __linux__
	munmap(base.data, 4 * (base.dim + 1) * base.num);
#else
	UnmapViewOfFile(base.data);
#endif
	base.data = nullptr;
}
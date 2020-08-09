#include "aknn_test.h"
#include "../include/aknn.cpp"
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>

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

void AKNN_T::load_data(const char * filename, uint16_t *& data, uint & num, uint & dim)
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
	num = (unsigned)(fsize / (dim * sizeof(uint16_t) + 4));
	data = new uint16_t[(size_t)num * (size_t)dim];

	in.seekg(0, ios::beg);
	for (size_t i = 0; i < num; i++) {
		in.seekg(4, ios::cur);
		in.read((char*)(data + i * dim), dim * sizeof(uint16_t));
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

void AKNN_T::load_train(const char * Rname, const char * coarsename, const char * idxname, const char * codename, const char * centroidname)
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
	if (pq.dsub * pq.M != pq.d) {
		cerr << "pq.dsub * pq.M != pq.d\n";
		exit(-1);
	}
	{
		uint16_t * p = nullptr;
		uint n, m;
		load_data(idxname, p, n, m);
		vector<uint16_t> tv(p, p + n);
		pq.idx.swap(tv);
		delete[] p;
	}
	{
		float * p = nullptr;
		uint d;
		AKNN::load_data(coarsename, p, pq.cn, d);
		vector<float> tv(p, p + pq.cn * d);
		pq.clscentroids.swap(tv);
		delete[] p;
		cout << "coarse\n";
		AKNN::display_data(pq.clscentroids.data(), pq.cn, d);
	}
	cerr << "M: " << pq.M << " dsub: " << pq.dsub << " d: " << pq.d
		 << " ksub: " << pq.ksub << " cn: " << pq.cn << endl;
}

void AKNN_T::save_fvecs(const char * filename, const float * x, int rows, int cols)
{
	std::ofstream out(filename, std::ios::binary);
	if (!out.is_open()) {
		printf("open file error\n");
		return;
	}
	for (int i = 0; i < rows; i++) {
		out.write((char*)&cols, 4);
		out.write((char *)(x + i * cols), cols * 4);
	}
	out.close();
}

void AKNN_T::save_codes(const char * filename, uint8_t * x, int rows, int cols)
{
	std::ofstream out(filename, std::ios::binary);
	if (!out.is_open()) {
		printf("open file error\n");
		return;
	}
	for (int i = 0; i < rows; i++) {
		out.write((char*)&cols, 4);
		out.write((char *)(x + i * cols), cols * sizeof(uint8_t));
	}
	out.close();
}

void AKNN_T::save_codes(const char * filename, uint16_t * x, int rows, int cols)
{
	std::ofstream out(filename, std::ios::binary);
	if (!out.is_open()) {
		printf("open file error\n");
		return;
	}
	for (int i = 0; i < rows; i++) {
		out.write((char*)&cols, 4);
		out.write((char *)(x + i * cols), cols * sizeof(uint16_t));
	}
	out.close();
}

//#define RESI

void AKNN_T::train(const char * basename, const char * coarsename, const char * idxname, const char * centroidname, const char * codename)
{
	if (strlen(basename) == 0) return;
	cerr << "read base " << basename << endl;
	AKNN::load_data(basename, base.data, base.num, base.dim);
	cerr << "dimention: " << base.dim << endl
		<< "points' num: " << base.num << endl;

	size_t nt = base.num, d = base.dim, nlist = 1;
	const float * xt = base.data;
	int _M = 32, _nbits = 8, bestM = _M;
	float minerr = 0x3f3f3f3f;

using namespace faiss;
#ifdef RESI
	IndexPreTransform * index = nullptr;

	for (int M = _M; M <= _M; M *= 2) {
		for (int nbits = _nbits; nbits <= _nbits; nbits *= 2) {
			fprintf(stderr, "M: %d	nbits: %d\n", M, nbits);
			fprintf(stderr, "  Preparing index OPQ d=%ld\n", d);
			IndexPreTransform * index_pt = nullptr;
			{
				IndexFlatL2 * coarse_quantizer = new IndexFlatL2(d);
				IndexIVFPQ * index_ivfpq = new IndexIVFPQ(coarse_quantizer, d,
					nlist, M, nbits);
				index_pt = new IndexPreTransform(index_ivfpq);
				index_pt->own_fields = true;
#ifdef opq
				VectorTransform * vt = new OPQMatrix(d, M);
				index_pt->prepend_transform(vt);
#endif
			}

			fprintf(stderr, "  Training on %ld vectors\n", nt);
			index_pt->index->verbose = true;
			//dynamic_cast<IndexIVFPQ *>(index_pt->index)->verbose = true;
			auto start = chrono::system_clock::now();
			float err = index_pt->mytrain(nt, xt);
			auto end = chrono::system_clock::now();
			fprintf(stderr, "  Time cost: %.3f s\n", chrono::duration<float>(end - start).count());
			fprintf(stderr, "  PQ Distortion: obj=%g\n", err);

			if (err < minerr) {
				minerr = err;
				bestM = M;
				index = index_pt;
			}
			else delete index_pt;
		}
	}

	fprintf(stderr, "bestM: %d minerr: %f\n", bestM, minerr);

#ifdef opq
	OPQMatrix * opqm = dynamic_cast<OPQMatrix *>(index->chain[0]);
	float * r = opqm->A.data();
	save_fvecs("../data/gist_R.fvecs", r, d, d);
#endif

	IndexIVFPQ * index_ivfpq = dynamic_cast<IndexIVFPQ *>(index->index);
	IndexFlatL2 * index_flat = dynamic_cast<IndexFlatL2 *>(index_ivfpq->quantizer);

	fprintf(stderr, "Get coarse\n");
	float * coarse = index_flat->xb.data();
	fprintf(stderr, "  %d * %d = %d coarse.size:%d \n", nlist, d, nlist * d, index_flat->xb.size());
	save_fvecs(coarsename, coarse, index_ivfpq->nlist, d);
	fprintf(stderr, "Get assign\n");
	uint16_t * idx = new uint16_t[nt];
	{
		Index::idx_t * assign = new Index::idx_t[nt];
		index_flat->assign(nt, xt, assign);
		copy(assign, assign + nt, idx);
		delete[] assign;
	}
	save_codes(idxname, idx, nt, 1);
	
	{
		int num = 3;
		for (int i = 0; i < num; i++) {
			printf("assign[%d]:%d ", i, idx[i]);
		}
		printf("\n\ncentroids\n");
		for (int i = 0; i < num; i++) {
			for (int j = 0; j < d; j++) {
				printf("%f ", coarse[i * d + j]);
			}
			printf("\n");
		}
	}

	fprintf(stderr, "Compute codes\n");
	vector<uint8_t> codes(nt * bestM);
	index_ivfpq->pq.compute_codes(xt, codes.data(), nt);
	fprintf(stderr, "  %d * %d = %d codes.size: %d\n", nt, bestM, nt * bestM, codes.size());
	save_codes(codename, codes.data(), nt, bestM);

	float * resi = index_ivfpq->pq.centroids.data();
	fprintf(stderr, "  %d * %d = %d centroids.size: %d\n", bestM * index_ivfpq->pq.ksub,
		index_ivfpq->pq.dsub, index_ivfpq->pq.ksub * d, index_ivfpq->pq.centroids.size());
	save_fvecs(centroidname, resi, bestM * index_ivfpq->pq.ksub, index_ivfpq->pq.dsub);

	{
		printf("\ncodes\n");
		int num = 3;
		for (int i = 0; i < num; i++) {
			for (int j = 0; j < bestM; j++) {
				printf("%d ", codes[i * bestM + j]);
			}
			printf("\n");
		}
		printf("\nresidual centroids\n");
		size_t dsub = index_ivfpq->pq.dsub;
		for (int i = 0; i < num; i++) {
			for (int j = 0; j < dsub; j++) {
				printf("%f ", resi[i * dsub + j]);
			}
			printf("\n");
		}
	}
#else
	IndexIVFPQ * index = nullptr;
	uint16_t * idx = nullptr;
	vector<uint8_t> codes(nt * bestM);

	for (int M = _M; M <= _M; M *= 2) {
		for (int nbits = _nbits; nbits <= _nbits; nbits *= 2) {
			fprintf(stderr, "M: %d	nbits: %d\n", M, nbits);
			IndexFlatL2 * coarse_quantizer = new IndexFlatL2(d);
			IndexIVFPQ * index_ivfpq = new IndexIVFPQ(coarse_quantizer, d, nlist, M, nbits);
			
			fprintf(stderr, "Training on %ld vectors\n", nt);
			auto start = chrono::system_clock::now();
			index_ivfpq->cp.max_points_per_centroid = 1000000;
			index_ivfpq->train_q1(nt, xt, true, index_ivfpq->metric_type);
			idx = new uint16_t[nt];
			{
				Index::idx_t * assign = new Index::idx_t[nt];
				coarse_quantizer->assign(nt, xt, assign);
				copy(assign, assign + nt, idx);
				delete[] assign;
			}
			vector<float> beta(nt * d);
			for (uint i = 0; i < nt; i++){
				for (uint j = 0; j < d; j++){
					beta[i * d + j] = xt[i * d + j] / coarse_quantizer->xb[idx[i] * d + j];
				}
			}
			index_ivfpq->pq.train(nt, beta.data());
			index_ivfpq->pq.compute_codes(beta.data(), codes.data(), nt);
			auto end = chrono::system_clock::now();
			fprintf(stderr, "  Time cost: %.3f s\n", chrono::duration<float>(end - start).count());
			index = index_ivfpq;
		}
	}
	IndexIVFPQ * index_ivfpq = index;
	IndexFlatL2 * index_flat = dynamic_cast<IndexFlatL2 *>(index_ivfpq->quantizer);

	float * coarse = index_flat->xb.data();
	fprintf(stderr, "  %d * %d = %d coarse.size:%d \n", nlist, d, nlist * d, index_flat->xb.size());
	save_fvecs(coarsename, coarse, index_ivfpq->nlist, d);
	save_codes(idxname, idx, nt, 1);
	
	{
		int num = 3;
		for (int i = 0; i < num; i++) {
			printf("assign[%d]:%d ", i, idx[i]);
		}
		printf("\n\ncentroids\n");
		for (int i = 0; i < num; i++) {
			for (int j = 0; j < d; j++) {
				printf("%f ", coarse[i * d + j]);
			}
			printf("\n");
		}
	}

	fprintf(stderr, "  %d * %d = %d codes.size: %d\n", nt, bestM, nt * bestM, codes.size());
	save_codes(codename, codes.data(), nt, bestM);

	float * resi = index_ivfpq->pq.centroids.data();
	fprintf(stderr, "  %d * %d = %d centroids.size: %d\n", bestM * index_ivfpq->pq.ksub,
		index_ivfpq->pq.dsub, index_ivfpq->pq.ksub * d, index_ivfpq->pq.centroids.size());
	save_fvecs(centroidname, resi, bestM * index_ivfpq->pq.ksub, index_ivfpq->pq.dsub);

	{
		printf("\ncodes\n");
		int num = 3;
		for (int i = 0; i < num; i++) {
			for (int j = 0; j < bestM; j++) {
				printf("%d ", codes[i * bestM + j]);
			}
			printf("\n");
		}
		printf("\nresidual centroids\n");
		size_t dsub = index_ivfpq->pq.dsub;
		for (int i = 0; i < num; i++) {
			for (int j = 0; j < dsub; j++) {
				printf("%f ", resi[i * dsub + j]);
			}
			printf("\n");
		}
	}
#endif

	delete[] base.data;
	base.data = nullptr;
	delete[] idx;
	delete index;
}

float AKNN_T::L2_sqr(float * vec1, float * vec2, uint dim)
{
	int nBlockWidth = 8;
	int cntBlock = dim / nBlockWidth;
	int cntRem = dim % nBlockWidth;

	__m256 mload1, mload2,
		mSub = _mm256_setzero_ps(),
		mSum = _mm256_setzero_ps();
	float *p1 = vec1, *p2 = vec2;
	for (int i = 0; i < cntBlock; i++)
	{
		mload1 = _mm256_loadu_ps(p1);
		mload2 = _mm256_loadu_ps(p2);
		mSub = _mm256_sub_ps(mload1, mload2);
		mSum = _mm256_fmadd_ps(mSub, mSub, mSum);
		p1 += nBlockWidth;
		p2 += nBlockWidth;
	}
	mSum = _mm256_hadd_ps(mSum, mSum);
	mSum = _mm256_hadd_ps(mSum, mSum);

	float sum = 0;
#ifdef __linux__
	sum += mSum[0];
	sum += mSum[4];
#else
	sum += mSum.m256_f32[0];
	sum += mSum.m256_f32[4];
#endif
	for(int i = 0; i < cntRem; i++) sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);

	return sum;
}

void AKNN_T::residual(float *vec1, float *vec2, uint dim)
{
	int nBlockWidth = 8;
	int cntBlock = dim / nBlockWidth;
	int cntRem = dim % nBlockWidth;

	__m256 mload1, mload2, mSub;
	float *p1 = vec1, *p2 = vec2;
	for (int i = 0; i < cntBlock; i++){
		mload1 = _mm256_loadu_ps(p1);
		mload2 = _mm256_loadu_ps(p2);
		mSub = _mm256_sub_ps(mload1, mload2);
		_mm256_storeu_ps(p1, mSub);
		p1 += nBlockWidth;
		p2 += nBlockWidth;
	}
	for (int i = 0; i < cntRem; i++) p1[i] -= p2[i];
}

void AKNN_T::compute_residual(float *ptr_q)
{
	float mindis = 0x3f3f3f3f, * cls = nullptr;
	for (uint i = 0; i < pq.cn; i++){
		float * xb = get_ptr(pq.clscentroids.data(), i, pq.d);
		float dis = distance(ptr_q, xb, pq.d);
		if (dis < mindis){
			mindis = dis;
			cls = xb;
		}
	}
	residual(ptr_q, cls, pq.d);
}

void AKNN_T::add_residual(float *vec1, float *vec2, uint dim)
{
	int nBlockWidth = 8;
	int cntBlock = dim / nBlockWidth;
	int cntRem = dim % nBlockWidth;

	__m256 mload1, mload2, mSum;
	float *p1 = vec1, *p2 = vec2;
	for (int i = 0; i < cntBlock; i++){
		mload1 = _mm256_loadu_ps(p1);
		mload2 = _mm256_loadu_ps(p2);
		mSum = _mm256_add_ps(mload1, mload2);
		_mm256_storeu_ps(p1, mSum);
		p1 += nBlockWidth;
		p2 += nBlockWidth;
	}
	for (int i = 0; i < cntRem; i++) p1[i] += p2[i];
}

void AKNN_T::compute_dis_table(std::vector<float>& distance_table, uint q)
{
	float * ptr_q = get_ptr(query.data, q, query.dim);
	compute_residual(ptr_q);
	for (uint m = 0; m < pq.M; m++) {
		for (uint cid = 0; cid < pq.ksub; cid++) {
			float * ptr_c = pq.centroids.data() + (m * pq.ksub + cid) * pq.dsub;
			distance_table[m * pq.ksub + cid] = L2_sqr(ptr_q, ptr_c, pq.dsub);
		}
		ptr_q += pq.dsub;
	}
}

void AKNN_T::compute_dis_table2(std::vector<float> &distance_table, uint q)
{
	float * ptr_q = get_ptr(query.data, q, query.dim);
	for (uint i = 0; i < base.num; i++){
		float * xc = get_ptr(pq.clscentroids.data(), pq.idx[i], pq.d);
		float * cls = new float[pq.d];
		float * p = cls;
		memcpy(cls, xc, pq.d * sizeof(float));
		for (uint m = 0; m < pq.M; m++){
			uint cid = pq.codes[i * pq.M + m];
			float * ptr_c = pq.centroids.data() + (m * pq.ksub + cid) * pq.dsub;
			add_residual(p, ptr_c, pq.dsub);
			p += pq.dsub;
		}
		distance_table[i] = distance(ptr_q, cls, base.dim);
		delete[] cls;
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
		/* compute residual */
		//vector<float> distance_table(pq.M * pq.ksub);
		//compute_dis_table(distance_table, q);
		/* add residual */
		//vector<float> distance_table(base.num);
		//compute_dis_table2(distance_table, q);
		// *******

		vector<uint> neighbors;
		int maxc = -1;
		for (uint r = 0; r < params.R; r++) {
			vector<uint> tempn(params.K);
			uint initPoint = rand() % base.num;
			//get_neighbors(tempn, distance_table, q, initPoint, params.K, params.L, params.E);
			get_neighbors2(tempn, q, initPoint, params.K, params.L, params.E);
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
	qtime += chrono::duration<float>(end - start).count();

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

	return sqrt(dis);
}

void AKNN_T::get_neighbors(std::vector<uint> & res, std::vector<float> & distance_table,
	uint q, uint initPoint, uint K, uint L, uint E)
{
	// Mod *******
	//float initDis = compute_PQdistance(initPoint, distance_table);
	float initDis = distance_table[initPoint];
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
			//float dis = compute_PQdistance(neighbors[j], distance_table);
			float dis = distance_table[neighbors[j]];
			// *******
			insert_pool(S, Neighbor(neighbors[j], dis), end);
			end = min(end + 1, (size_t)L);
			in[neighbors[j]] = true;
		}
	}
	for (i = 0; i < K && i < S.size(); i++) res[i] = S[i].id;
}

void AKNN_T::mul_beta(float *vec1, float *vec2, uint dim)
{
	int nBlockWidth = 8;
	int cntBlock = dim / nBlockWidth;
	int cntRem = dim % nBlockWidth;

	__m256 mload1, mload2, mProd;
	float *p1 = vec1, *p2 = vec2;
	for (int i = 0; i < cntBlock; i++){
		mload1 = _mm256_loadu_ps(p1);
		mload2 = _mm256_loadu_ps(p2);
		mProd = _mm256_mul_ps(mload1, mload2);
		_mm256_storeu_ps(p1, mProd);
		p1 += nBlockWidth;
		p2 += nBlockWidth;
	}
	for (int i = 0; i < cntRem; i++) p1[i] *= p2[i];
}

float AKNN_T::compute_newdistance(uint pqid, uint q)
{
	float * ptr_c = get_ptr(pq.clscentroids.data(), pq.idx[pqid], pq.d);
	float * ptr_q = get_ptr(query.data, q, query.dim);
	float * cls = new float[pq.d];
	memcpy(cls, ptr_c, pq.d * sizeof(float));
	float * p = cls;
	for (uint m = 0; m < pq.M; m++){
		uint code = pq.codes[pqid * pq.M + m];
		float * ptr_b = pq.centroids.data() + (m * pq.ksub + code) * pq.dsub;
		mul_beta(p, ptr_b, pq.dsub);
		p += pq.dsub;
	}
	float dis = distance(cls, ptr_q, pq.d);
	delete[] cls;
	return dis;
}

void AKNN_T::get_neighbors2(std::vector<uint> &res, uint q, uint initPoint, uint K, uint L, uint E)
{
	float initDis = compute_newdistance(initPoint, q);
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
			float dis = compute_newdistance(neighbors[j], q);
			insert_pool(S, Neighbor(neighbors[j], dis), end);
			end = min(end + 1, (size_t)L);
			in[neighbors[j]] = true;
		}
	}
	for (i = 0; i < K && i < S.size(); i++) res[i] = S[i].id;
}
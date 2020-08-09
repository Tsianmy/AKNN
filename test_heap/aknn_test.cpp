#include "aknn_test.h"
#include "../include/aknn.cpp"
#include "../include/fixed_heap.hpp"

void AKNN_T::get_neighbors(vector<uint> & res, uint q, uint initPoint, uint K, uint L, uint E)
{
	float * ptr_q = get_ptr(query.data, q, query.dim);
	float initDis = distance(get_ptr(base.data, initPoint, base.dim), ptr_q, base.dim);

	fixedMinHeap<Neighbor> S(L);
	S.push(Neighbor(initPoint, initDis));

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
		for (j = 0; j < E; j++) {
			if (vis[neighbors[j]] || in[neighbors[j]]) continue;
			float * pj = get_ptr(base.data, neighbors[j], base.dim);
			float dis = distance(pj, ptr_q, base.dim);
			S.push(Neighbor(neighbors[j], dis));
			in[neighbors[j]] = true;
		}
	}
	for (i = 0; i < K && !S.empty(); i++) {
		res[i] = S.top().id;
		S.pop();
	}
}
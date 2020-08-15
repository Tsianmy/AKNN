#include "mIndexHeap.h"
#include "fixed_heap.hpp"
#include <vector>

namespace aknn {

void mIndexHeap::search_neighbors(idx_t qi, idx_t initPi, idx_t K, idx_t L, idx_t E, idx_t * res)
{
	float initDis = compute_distance(initPi, qi);
	fixedMinHeap<Neighbor> S(L);
	S.push(Neighbor(initPi, initDis));

	std::vector<bool> vis(nb), in(nb);

	idx_t i = 0;
	while (i < L) {
		// find the index of the first unchecked node in S
		idx_t j;
		for (j = 0; j < S.size(); j++) {
			if (!vis[S[j].id]) {
				i = j;
				break;
			}
		}
		if (j == S.size()) break;

		idx_t id = S[i].id;
		vis[id] = true;

		// add neighbors to the candidate pool and reorder by distance
		idx_t * neighbors = get_ptr(xg, id, k);
		for (j = 0; j < E; j++) {
			if (vis[neighbors[j]] || in[neighbors[j]]) continue;
			float dis = compute_distance(neighbors[j], qi);
			S.push(Neighbor(neighbors[j], dis));
			in[neighbors[j]] = true;
		}
	}
	for (i = 0; i < K && !S.empty(); i++) {
		res[i] = S.top().id;
		S.pop();
	}
}

}	// namespace aknn
#pragma once

#include "mIndexG.h"

namespace aknn {

class mIndexHeap : public mIndexG
{
public:
	mIndexHeap() {}
	void search_neighbors(idx_t qi, idx_t initPi, idx_t K, idx_t L, idx_t E, idx_t * res) override;
};

}	// namespace aknn
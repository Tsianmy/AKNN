#ifndef PQ_H
#define PQ_H

struct ProductQuantizer
{
	uint M, d, dsub, n, ksub;
	std::vector<float> R;			// d * d
	std::vector<float> centroids;	// (M * ksub) * dsub
	std::vector<uint8_t> codes;		// n * M
};

#endif // !PQ_H

#ifndef PQ_H
#define PQ_H

struct ProductQuantizer
{
	uint M, d, dsub, n, ksub;
	std::vector<float> R;			// d * d
	std::vector<float> centroids;	// (M * ksub) * dsub
	std::vector<uint8_t> codes;		// n * M
};

struct ProductQuantizerResi
{
	uint M, d, dsub, n, ksub, cn;
	std::vector<float> R;				// d * d
	std::vector<float> centroids;		// (M * ksub) * dsub
	std::vector<uint8_t> codes;			// n * M
	std::vector<float> clscentroids;	// cn * d
	std::vector<uint16_t> idx;			// n
};

#endif // !PQ_H

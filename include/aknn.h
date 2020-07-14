#ifndef AKNN_H
#define AKNN_H

#include <vector>
typedef unsigned uint;

class Param {
public:
	char *basename, *queryname, *graphname, *gtname, *outputname;
	uint K, L, E, I;

	Param(){}
	Param(char *_base, char *_query, char *_graph, char *_gt, char *_output, uint _K, uint _L, uint _E, uint _I):
		basename(_base), queryname(_query), graphname(_graph), gtname(_gt), outputname(_output),
		K(_K), L(_L), E(_E), I(_I)
	{
		if (L < K) L = K;
	}
};

struct Neighbor {
	uint id;
	float distance;
	Neighbor(uint _id = 0, float _d = 0) : id(_id), distance(_d) {}
	bool operator < (const Neighbor & n) {
		return distance < n.distance;
	}
};

template<typename T>
struct Data {
	uint dim, num;
	T * data = nullptr;
};

class AKNN {
public:
	explicit AKNN(Param _params) : params(_params) {}
	void load();
	void search();
	void save();

	static float distance(float * vec1, float * vec2, uint dim);
	static float * get_ptr(float * begin, const uint index, const uint dim);
	static int * get_ptr(int * begin, const uint index, const uint dim);

	~AKNN();

protected:
	Param params;
	Data<float> base, query;
	Data<int> graph, groundtruth;
	Data<int> searchRes;

	void load_data(char* filename, float*& data, uint& num, uint& dim);
	void load_data(char* filename, int*& data, uint& num, uint& dim);

	void get_neighbors(std::vector<uint> & res, uint q, uint initPoint, uint K, uint L, uint E);
};

#endif // !AKNN_H

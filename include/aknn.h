#ifndef AKNN_H
#define AKNN_H

#include <vector>
#include <unordered_set>
typedef unsigned uint;

struct Param {
	uint K, L, E, R;
	Param() {}
	Param(uint _K, uint _L, uint _E, uint _R = 1) : K(_K), L(_L), E(_K), R(_R) {}
};

struct Neighbor {
	uint id;
	float distance;
	Neighbor(uint _id = 0, float _d = 0) : id(_id), distance(_d) {}
	bool operator < (const Neighbor & n) const {
		return distance < n.distance;
	}
	bool operator > (const Neighbor & n) const {
		return distance > n.distance;
	}
};

template<typename T>
struct Data {
	uint dim, num;
	T * data = nullptr;
};

class AKNN {
public:
	AKNN() {}
	AKNN(const char *basename, const char *queryname, const char *graphname, const char *gtname);
	void load(const char *basename, const char *queryname, const char *graphname, const char *gtname);
	void search();
	void save(const char *outputname);
	void init_params(Param _params);
	void display();
	void set_E(uint E);
	void set_R(uint R);
	void set_L(uint L);
	void gen_lknn(uint K, char * lknnName);

	static void insert_pool(std::vector<Neighbor>& pool, Neighbor p, size_t end);
	static float distance(float * vec1, float * vec2, uint dim);
	static float * get_ptr(float * begin, const uint index, const uint dim);
	static int * get_ptr(int * begin, const uint index, const uint dim);

	~AKNN();

protected:
	Param params;
	bool ready = false;
	Data<float> base, query;
	Data<int> graph, groundtruth;
	Data<int> searchRes;
	std::vector<std::unordered_set<uint>> gtset;

	void load_data(const char* filename, float*& data, uint& num, uint& dim);
	void load_data(const char* filename, int*& data, uint& num, uint& dim);
	void display_data(float* data, uint num, uint dim);
	void display_data(int* data, uint num, uint dim);

	void get_neighbors(std::vector<uint> & res, uint q, uint initPoint, uint K, uint L, uint E);
};

#endif // !AKNN_H

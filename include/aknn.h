#ifndef AKNN_H
#define AKNN_H

class Param {
public:
	char *basename, *queryname, *graphname, *gtname;
	unsigned K, L;

	Param(){}
	Param(char *_base, char *_query, char *_graph, char *_gt, unsigned _K, unsigned _L):
		basename(_base), queryname(_query), graphname(_graph), gtname(_gt), K(_K), L(_L) {}
};

template<typename T>
struct Data {
	unsigned dim, num;
	T * data = nullptr;
};

class AKNN {
public:
	explicit AKNN(Param _params) : params(_params) {}
	void load();
	~AKNN();

protected:
	Param params;
	Data<float> base, query;
	Data<int> graph, groundtruth;

	void load_float(char* filename, float*& data, unsigned& num, unsigned& dim);
	void load_int(char* filename, int*& data, unsigned& num, unsigned& dim);

	float distance(float * vec1, float * vec2, unsigned dim);
	float * get_ptr(float * begin, const unsigned index, const unsigned dim);
};

#endif // !AKNN_H

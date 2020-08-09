#include "aknn_test.h"
#include "../include/aknn.cpp"
#include <set>

void AKNN_T::test_groundtruth()
{
	cout << "test_groundtruth\n";
	cout << "\ndistance(query0, point0..99):\n";
	unsigned v1 = 0;
	float * qv0 = query.data;
	for (unsigned v2 = 0; v2 < 100; v2++) {
		float * vec = get_ptr(base.data, v2, base.dim);
		cout << "d(q0, " << v2 << "):" << distance(qv0, vec, base.dim) << " ";
	}

	set<unsigned> s;

	cout << "\n\ndistance(query0, groundtruth0..9):\n";
	for (size_t i = 0; i < groundtruth.dim; i++) {
		unsigned v2 = groundtruth.data[v1 * groundtruth.dim + i];
		s.insert(v2);
		float * vec = get_ptr(base.data, v2, base.dim);
		cout << "d(q0, " << v2 << "):" << distance(qv0, vec, base.dim) << " ";
	}
	cout << endl;

	float * gt99 = get_ptr(base.data, groundtruth.data[99], base.dim);
	bool flag = true;
	for (unsigned v2 = 0; v2 < base.num; v2++) {
		if (s.count(v2)) continue;
		float * vec = get_ptr(base.data, v2, base.dim);
		if (distance(qv0, vec, base.dim) < distance(qv0, gt99, base.dim)) {
			flag = false;
			break;
		}
	}
	if (flag == true) cout << "truly nearest neighbors!\n";
	else cout << "not nearest neighbors!\n";
}

void AKNN_T::test_knng()
{
	cout << "test_knng\n";

	cout << "\ndistance(base0, point1..99):\n";
	unsigned v1 = 0;
	float * vec0 = get_ptr(base.data, 0, base.dim);
	for (unsigned v2 = 1; v2 < 100; v2++) {
		float * vec = get_ptr(base.data, v2, base.dim);
		cout << "d(0, " << v2 << "):" << distance(vec0, vec, base.dim) << " ";
	}
	cout << endl;

	cout << "\n\ndistance(base0, knn0..99):\n";
	set<unsigned> s;
	for (size_t i = 0; i < graph.dim; i++) {
		unsigned v2 = graph.data[v1 * graph.dim + i];
		s.insert(v2);
		float * vec = get_ptr(base.data, v2, base.dim);
		cout << "d(0, " << v2 << "):" << distance(vec0, vec, base.dim) << " ";
	}
	cout << endl;

	float * gt99 = get_ptr(base.data, graph.data[99], base.dim);
	bool flag = true;
	for (unsigned v2 = 1; v2 < base.num; v2++) {
		if (s.count(v2)) continue;
		float * vec = get_ptr(base.data, v2, base.dim);
		if (distance(vec0, vec, base.dim) < distance(vec0, gt99, base.dim)) {
			flag = false;
			break;
		}
	}
	if (flag == true) cout << "truly nearest neighbors!\n";
	else cout << "not nearest neighbors!\n";
}
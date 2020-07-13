#include <iostream>
#include <algorithm>
#include <assert.h>
#include <functional>

template<typename T>
class fixedMinHeap {
public:
	fixedMinHeap(size_t sz) : capacity(sz) { arr = new T[sz]; }
	void push(T x);
	T top();
	void pop();
	~fixedMinHeap();
	void display();
private:
	T * arr = nullptr;
	size_t size = 0;
	size_t capacity = 0;
	void upadjust(const int low, const int high);
};

template<typename T>
inline void fixedMinHeap<T>::upadjust(const int beg, const int end)
{
	int cur = end, p = (cur - 1) / 2;
	while (cur >= beg) {
		if (arr[cur] < arr[p]) {
			swap(arr[cur], arr[p]);
			cur = p;
			p = (cur - 1) / 2;
		}
		else break;
	}
}

template<typename T>
void fixedMinHeap<T>::push(T x)
{
	assert(size <= capacity);
	if (size < capacity) {
		arr[size++] = x;
		push_heap(arr, arr + size, greater<T>());
	}
	else {
		T * mn = max_element(arr + size / 2, arr + size);
		if (x < *mn) {
			*mn = x;
			upadjust(0, mn - arr);
		}
	}
}

template<typename T>
T fixedMinHeap<T>::top()
{
	assert(size > 0);
	return arr[0];
}

template<typename T>
inline void fixedMinHeap<T>::pop()
{
	assert(size > 0);
	pop_heap(arr, arr + size, greater<T>());
	size--;
}

template<typename T>
fixedMinHeap<T>::~fixedMinHeap()
{
	delete arr;
	arr = nullptr;
	size = capacity = 0;
}

template<typename T>
void fixedMinHeap<T>::display()
{
	for (auto i = 0; i < size; i++) printf(" %d", arr[i]);
	printf("\n");
}


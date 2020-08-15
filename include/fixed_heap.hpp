#ifndef FIXED_HEAP_H
#define FIXED_HEAP_H

#include <iostream>
#include <algorithm>
#include <assert.h>
#include <functional>

template<typename T>
class fixedHeap {
public:
	fixedHeap(size_t sz) : _capacity(sz) { arr = new T[sz]; }
	void push(T x);
	T top();
	void pop();
	size_t size();
	size_t capacity();
	bool empty();
	T * data();
	void display();
	T & operator [] (int i);
	~fixedHeap();
protected:
	T * arr = nullptr;
	size_t _size = 0;
	size_t _capacity = 0;
};

template<typename T>
void fixedHeap<T>::push(T x)
{
	assert(_size <= _capacity);
	if (_size < _capacity) {
		arr[_size++] = x;
		push_heap(arr, arr + _size);
	}
	else {
		if (x < arr[0]) {
			pop_heap(arr, arr + _size);
			arr[_size - 1] = x;
			push_heap(arr, arr + _size);
		}
	}
}

template<typename T>
T fixedHeap<T>::top()
{
	assert(_size > 0);
	return arr[0];
}

template<typename T>
inline void fixedHeap<T>::pop()
{
	assert(_size > 0);
	pop_heap(arr, arr + _size);
	_size--;
}

template<typename T>
inline size_t fixedHeap<T>::size()
{
	return _size;
}

template<typename T>
inline size_t fixedHeap<T>::capacity()
{
	return _capacity;
}

template<typename T>
inline bool fixedHeap<T>::empty()
{
	return _size == 0;
}

template<typename T>
inline T * fixedHeap<T>::data()
{
	return arr;
}

template<typename T>
void fixedHeap<T>::display()
{
	for (auto i = 0; i < _size; i++) printf(" %d", arr[i]);
	printf("\n");
}

template<typename T>
inline T & fixedHeap<T>::operator[](int i)
{
	return arr[i];
}

template<typename T>
fixedHeap<T>::~fixedHeap()
{
	delete arr;
	arr = nullptr;
	_size = _capacity = 0;
}

/************************
*   Min Heap
************************/

template<typename T>
class fixedMinHeap: public fixedHeap<T> {
public:
	fixedMinHeap(size_t sz) : fixedHeap<T>(sz) {}
	void push(T x);
	void pop();
};

template<typename T>
void fixedMinHeap<T>::push(T x)
{
	assert(this->_size <= this->_capacity);
	if (this->_size < this->_capacity) {
		this->arr[this->_size++] = x;
		push_heap(this->arr, this->arr + this->_size, std::greater<T>());
	}
	else {
		T * mn = std::max_element(this->arr + this->_size / 2, this->arr + this->_size);
		if (x < *mn) {
			*mn = x;
			push_heap(this->arr, mn + 1, std::greater<T>());
		}
	}
}

template<typename T>
inline void fixedMinHeap<T>::pop()
{
	assert(this->_size > 0);
	pop_heap(this->arr, this->arr + this->_size, std::greater<T>());
	this->_size--;
}

#endif

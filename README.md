# Project of AKNN

This is a project of AKNN.

### Requirements and Dependencies

- **Requirements:**
	
	- A C++ compiler
	
- **Dependencies:**
	- Faiss library (https://github.com/facebookresearch/faiss)
	- Intel MKL or Openblas

### DataSet

**SIFT1M** and **GIST1M** from http://corpus-texmex.irisa.fr/.

You need to put the dataset to the directory AKNN/data.

Additionally, **KNN-graphs** built on the dataset are needed. Their paths should be AKNN/data/sift_100NN_100.graph and AKNN/data/gist_100NN_100.graph.

### Modules

- **test_structure:** Check the structure of data.
- **base:** Base module for approximate nearest neighbor search with AVX intrinsics and candidate pool implemented by insertion sort.
- **test_heap:** Use a fixed-size heap to implement the candidate pool.
- **test_mmap:** Avoid loading base data by mmap.
- **test_opq:** Avoid loading base data by optimized product quantization.
- **test_new:** A new method to combine graph-based search and product quantization.
- **gen_knn:** Generate a k-NN graph (k <= 100).
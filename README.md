# AKNN

A project of AKNN.

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
- **gen_knn:** Generate a k-NN graph (k <= 100).
- **gen_mbase:** Generate data for mmap.
- **demos:** Different searching methods.
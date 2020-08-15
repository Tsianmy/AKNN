## Demos

Different searching methods.

##### Usage

```
mkdir build
cmake -B build/ [-Dmod=<mod>] [-Dplot=<ON/OFF>] [-Dgist=<ON/OFF>]
make -C build/
./AKNN.exe
```

##### Options

###### mod

- "base" (default): Base Graph-based search.
- "test_heap": Implement candidate pool $S$ with fixed-size heap.
- "test_mmap": Avoid loading base data by mmap.
- "test_opq": Use optimized product quantization to search with less memory.
- "test_mq1": Use two stage quantization, coarse quantization and product quantization for residual.
- "test_mq2": Use two stage quantization. Add residual to centroids to approximate a point instead of calculating a preliminary distance table.
- "test_mq3": Use two stage quantization. Firstly use coarse quantization. Secondly compute beta and use product quantization.

###### plot

Control queries per second (QPS). A QPS - recall figure can be ploted with the output. "OFF" for default.

###### gist

Search in GIST1M if "ON". Else search in SIFT1M (default).


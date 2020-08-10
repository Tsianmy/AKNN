## Test_opq

Avoid loading base data by optimized product quantization.

##### Usage

In CMakeLists.txt, `add_definitions(-DTRAIN)` is used to switch to training mode or search mode.

```bash
mkdir build
cmake -B build/
make -C build/
./AKNN.exe
```


# Lesson 7 benchmark. Needs the CUDA toolkit (nvcc) and cuBLAS, which ships with it.
# -arch=native compiles for the GPU in this machine.
NVCC ?= nvcc

bench: bench.cu
	$(NVCC) -O3 -std=c++17 -arch=native -o $@ $< -lcublas

clean:
	rm -f bench

.PHONY: clean

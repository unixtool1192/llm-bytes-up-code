// Lesson 7 - The hardware fit.
//
// Measures the numbers the lesson uses, on whatever NVIDIA GPU runs it:
//   1. what the GPU says about itself (SMs, caches, registers, memory)
//   2. how fast it can read memory, with a deliberately simple hand-written kernel
//   3. how long multiplying one LLM weight matrix by N token vectors takes, for N = 1 .. 4096
//      (the shape of blk.N.ffn_gate in the course's model: 11008 x 4096, BF16)
//   4. peak matrix-multiply throughput, BF16 on tensor cores versus FP32 without them
//
// Build and run:   make && ./bench
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__); exit(1); } } while (0)

// ---- 2. read bandwidth --------------------------------------------------------------
// Every thread reads 16-byte chunks in a grid-stride loop and adds them up. The sum is written
// out only so the compiler cannot skip the reads. This is a memory benchmark, not a reduction.
__global__ void read_kernel(const float4 *in, size_t n, float *out) {
    float acc = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float4 v = in[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 123.456f) out[0] = acc;   // practically never true; keeps the loads alive
}

__global__ void fill_bf16(__nv_bfloat16 *p, size_t n, float scale) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = __float2bfloat16(scale * (float)((i * 2654435761u) % 1000) / 1000.f - scale / 2);
}

__global__ void fill_f32(float *p, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = (float)((i * 2654435761u) % 1000) / 1000.f - 0.5f;
}

// Time `fn` with CUDA events: warm up, then average over `iters` runs, in milliseconds.
template <typename F> float time_ms(F fn, int warm, int iters) {
    for (int i = 0; i < warm; i++) fn();
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) fn();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return ms / iters;
}

int main() {
    // ---- 1. the device ----------------------------------------------------------------
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("== 1. the device ==\n");
    printf("  name:                          %s\n", p.name);
    printf("  compute capability:            %d.%d\n", p.major, p.minor);
    printf("  streaming multiprocessors:     %d\n", p.multiProcessorCount);
    printf("  max threads per SM:            %d\n", p.maxThreadsPerMultiProcessor);
    printf("  32-bit registers per SM:       %d  (%d KB)\n", p.regsPerMultiprocessor, p.regsPerMultiprocessor * 4 / 1024);
    printf("  shared memory per SM:          %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
    printf("  L2 cache:                      %d KB\n", p.l2CacheSize / 1024);
    printf("  global memory visible to GPU:  %.1f GB\n", p.totalGlobalMem / 1e9);
    printf("  memory bus width:              %d bits\n", p.memoryBusWidth);
    int integrated = 0;
    CK(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, 0));
    printf("  shares memory with the CPU:    %s\n", integrated ? "yes (integrated)" : "no (discrete)");
    size_t free_b, total_b;
    CK(cudaMemGetInfo(&free_b, &total_b));
    printf("  free right now:                %.1f GB\n", free_b / 1e9);

    // ---- 2. read bandwidth ------------------------------------------------------------
    printf("\n== 2. read bandwidth (hand-written kernel) ==\n");
    const size_t bytes = (size_t)2 << 30;                 // 2 GiB: far bigger than any cache
    float4 *buf; float *sink;
    CK(cudaMalloc(&buf, bytes)); CK(cudaMalloc(&sink, sizeof(float)));
    CK(cudaMemset(buf, 0, bytes));
    const size_t n4 = bytes / sizeof(float4);
    const int blocks = p.multiProcessorCount * 8, threads = 512;
    float ms = time_ms([&] { read_kernel<<<blocks, threads>>>(buf, n4, sink); }, 3, 20);
    CK(cudaGetLastError());
    printf("  read %.2f GB in %.2f ms  ->  %.0f GB/s\n", bytes / 1e9, ms, bytes / 1e9 / (ms / 1e3));
    CK(cudaFree(buf));

    // ---- 3. one weight matrix times N tokens ------------------------------------------
    // C (rows x N) = W (rows x cols) . X (cols x N), BF16 inputs, FP32 accumulate.
    // cuBLAS is column-major; for timing purposes the layout does not matter.
    printf("\n== 3. weight matrix (11008 x 4096, BF16) times N token vectors ==\n");
    const int rows = 11008, cols = 4096, maxN = 4096;
    __nv_bfloat16 *W, *X; float *C;
    CK(cudaMalloc(&W, (size_t)rows * cols * 2));
    CK(cudaMalloc(&X, (size_t)cols * maxN * 2));
    CK(cudaMalloc(&C, (size_t)rows * maxN * 4));
    fill_bf16<<<blocks, threads>>>(W, (size_t)rows * cols, 0.1f);
    fill_bf16<<<blocks, threads>>>(X, (size_t)cols * maxN, 1.0f);
    CK(cudaDeviceSynchronize());
    cublasHandle_t h; CB(cublasCreate(&h));
    const float one = 1.f, zero = 0.f;
    const double wbytes = (double)rows * cols * 2;
    printf("  %6s %10s %12s %14s %12s\n", "N", "time (ms)", "TFLOP/s", "weights GB/s", "ms per token");
    for (int N = 1; N <= maxN; N *= 2) {
        float t = time_ms([&] {
            CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, rows, N, cols, &one,
                            W, CUDA_R_16BF, rows, X, CUDA_R_16BF, cols, &zero,
                            C, CUDA_R_32F, rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }, 5, 50);
        double flops = 2.0 * rows * cols * N;
        printf("  %6d %10.3f %12.2f %14.0f %12.5f\n", N, t, flops / (t / 1e3) / 1e12, wbytes / (t / 1e3) / 1e9, t / N);
    }
    CK(cudaFree(W)); CK(cudaFree(X)); CK(cudaFree(C));

    // ---- 4. peak matrix multiply: tensor cores (BF16) vs plain FP32 -------------------
    printf("\n== 4. big square matrix multiply (8192 x 8192 x 8192) ==\n");
    const int S = 8192;
    __nv_bfloat16 *A16, *B16; float *A32, *B32, *C32;
    CK(cudaMalloc(&A16, (size_t)S * S * 2)); CK(cudaMalloc(&B16, (size_t)S * S * 2));
    CK(cudaMalloc(&A32, (size_t)S * S * 4)); CK(cudaMalloc(&B32, (size_t)S * S * 4));
    CK(cudaMalloc(&C32, (size_t)S * S * 4));
    fill_bf16<<<blocks, threads>>>(A16, (size_t)S * S, 1.0f);
    fill_bf16<<<blocks, threads>>>(B16, (size_t)S * S, 1.0f);
    fill_f32<<<blocks, threads>>>(A32, (size_t)S * S);
    fill_f32<<<blocks, threads>>>(B32, (size_t)S * S);
    CK(cudaDeviceSynchronize());
    const double big = 2.0 * S * S * S;
    float t16 = time_ms([&] {
        CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, S, S, S, &one, A16, CUDA_R_16BF, S, B16, CUDA_R_16BF, S,
                        &zero, C32, CUDA_R_32F, S, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }, 2, 5);
    CB(cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH));    // FP32 exactly, no TF32 tensor-core shortcut
    float t32 = time_ms([&] {
        CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, S, S, S, &one, A32, S, B32, S, &zero, C32, S));
    }, 1, 3);
    printf("  BF16 on tensor cores:  %8.1f ms  ->  %6.1f TFLOP/s\n", t16, big / (t16 / 1e3) / 1e12);
    printf("  FP32, no tensor cores: %8.1f ms  ->  %6.1f TFLOP/s\n", t32, big / (t32 / 1e3) / 1e12);
    printf("  ratio:                 %.1fx\n", t32 / t16);

    CB(cublasDestroy(h));
    CK(cudaFree(A16)); CK(cudaFree(B16)); CK(cudaFree(A32)); CK(cudaFree(B32)); CK(cudaFree(C32));
    CK(cudaFree(sink));
    return 0;
}

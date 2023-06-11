#include "cuda_fp16.h"

template<typename T>
__device__ void axpy(const size_t n, T* a, const T alpha, const T* b, const T beta) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        a[i] = a[i] * alpha + b[i] * beta;
    }
}

#if __CUDA_ARCH__ >= 530
extern "C" __global__ void axpy_f16(const size_t n, __half* a, const __half alpha, const __half* b, const __half beta) {
    axpy(n, a, alpha, b, beta);
}
#endif

extern "C" __global__ void axpy_f32(const size_t n, float* a, const float alpha, const float* b, const float beta) {
    axpy(n, a, alpha, b, beta);
}

extern "C" __global__ void axpy_f64(const size_t n, double* a, const double alpha, const double* b, const double beta) {
    axpy(n, a, alpha, b, beta);
}

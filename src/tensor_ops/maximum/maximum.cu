struct MaximumKernalOp {};

__device__ unsigned int get_strided_index(
    unsigned int idx,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    unsigned int strided_i = 0;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

extern "C" __global__ void maximum_forward(
    const MaximumKernalOp op,
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const float *lhs,
    const size_t *lhs_strides,
    const float *rhs,
    const size_t *rhs_strides,
    float *out,
    const size_t *out_strides
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    unsigned int lhs_i = get_strided_index(i, num_dims, dims, lhs_strides);
    unsigned int rhs_i = get_strided_index(i, num_dims, dims, rhs_strides);
    unsigned int out_i = get_strided_index(i, num_dims, dims, out_strides);

    out[out_i] = fmaxf(lhs[lhs_i], rhs[rhs_i]);
}

extern "C" __global__ void maximum_backward(
    const MaximumKernalOp op,
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const float *lhs,
    float *grad_lhs,
    const size_t *lhs_strides,
    const float *rhs,
    float *grad_rhs,
    const size_t *rhs_strides,
    const float *grad_out,
    const size_t *out_strides
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    unsigned int lhs_i = get_strided_index(i, num_dims, dims, lhs_strides);
    unsigned int rhs_i = get_strided_index(i, num_dims, dims, rhs_strides);
    unsigned int out_i = get_strided_index(i, num_dims, dims, out_strides);

    auto x = lhs[lhs_i];
    auto y = rhs[rhs_i];
    auto go = grad_out[out_i];

    float dfdx, dfdy;

    if (x > y) {
        dfdx = 1.0;
        dfdy = 0.0;
    } else if (x < y) {
        dfdx = 0.0;
        dfdy = 1.0;
    } else {
        dfdx = 0.5;
        dfdy = 0.5;
    }

    atomicAdd(grad_lhs + lhs_i, dfdx * go);
    atomicAdd(grad_rhs + rhs_i, dfdy * go);
}

#include "unary_op_macros.cuh"

struct NegateKernelOp {};

#if __CUDA_ARCH__ >= 530
UNARY_OP(__half, negate_fwd_f16, negate_bwd_f16, NegateKernelOp,
        -x,
        -1.0)
#endif

UNARY_OP(float, negate_fwd_f32, negate_bwd_f32, NegateKernelOp,
        -x,
        -1.0)

UNARY_OP(double, negate_fwd_f64, negate_bwd_f64, NegateKernelOp,
        -x,
        -1.0)

#include "unary_op_macros.cuh"

template<typename F>
struct ScalarAddKernelOp {
    F scalar;
};

#if __CUDA_ARCH__ >= 530
UNARY_OP(__half, sadd_fwd_f16, sadd_bwd_f16, ScalarAddKernelOp<__half>,
    x + op.scalar,
    1.0);
#endif

UNARY_OP(float, sadd_fwd_f32, sadd_bwd_f32, ScalarAddKernelOp<float>,
    x + op.scalar,
    1.0);

UNARY_OP(double, sadd_fwd_f64, sadd_bwd_f64, ScalarAddKernelOp<double>,
    x + op.scalar,
    1.0);
    

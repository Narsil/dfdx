#include "unary_op_macros.cuh"

struct ExpKernelOp {};

#if __CUDA_ARCH__ >= 530
UNARY_OP(__half, exp_fwd_f16, exp_bwd_f16, ExpKernelOp,
        expg(x),
        y)
#endif

UNARY_OP(float, exp_fwd_f32, exp_bwd_f32, ExpKernelOp,
        expg(x),
        y)

UNARY_OP(double, exp_fwd_f64, exp_bwd_f64, ExpKernelOp,
        expg(x),
        y)
        

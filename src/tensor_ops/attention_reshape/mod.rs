use crate::{shapes::*, tensor::*};

mod cpu_kernel;
#[cfg(feature = "cuda")]
mod cuda_kernel;

/// AttentionReshape an array or vec of tensors together along a new dimension.
pub trait TryAttentionReshape<E: Dtype>: DeviceStorage {
    fn attention_reshape<
        const S: char,
        const THREE_HIDDEN_DIM: usize,
        const NUM_HEADS: usize,
        const HEAD_DIM: usize,
    >(
        &self,
        qkv: &Tensor<(Dyn<S>, Const<THREE_HIDDEN_DIM>), E, Self>,
        past_key: &Tensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize), E, Self>,
        past_value: &Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), E, Self>,
    ) -> (
        Tensor<(Const<NUM_HEADS>, Dyn<S>, Const<HEAD_DIM>), E, Self>,
        Tensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize), E, Self>,
        Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), E, Self>,
    ) {
        self.try_attention_reshape(qkv, past_key, past_value)
            .unwrap()
    }

    /// Fallible version of [TryAttentionReshape::cat]
    fn try_attention_reshape<
        const S: char,
        const THREE_HIDDEN_DIM: usize,
        const NUM_HEADS: usize,
        const HEAD_DIM: usize,
    >(
        &self,
        qkv: &Tensor<(Dyn<S>, Const<THREE_HIDDEN_DIM>), E, Self>,
        past_key: &Tensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize), E, Self>,
        past_value: &Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), E, Self>,
    ) -> Result<
        (
            Tensor<(Const<NUM_HEADS>, Dyn<S>, Const<HEAD_DIM>), E, Self>,
            Tensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize), E, Self>,
            Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), E, Self>,
        ),
        Self::Err,
    >;
}

pub trait AttentionReshapeKernel<E: Dtype>: DeviceStorage {
    fn forward<
        const S: char,
        const THREE_HIDDEN_DIM: usize,
        const NUM_HEADS: usize,
        const HEAD_DIM: usize,
    >(
        &self,
        qkv: &Tensor<(Dyn<S>, Const<THREE_HIDDEN_DIM>), E, Self>,
        past_key: &Tensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize), E, Self>,
        past_value: &Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), E, Self>,
    ) -> Result<
        (
            Tensor<(Const<NUM_HEADS>, Dyn<S>, Const<HEAD_DIM>), E, Self>,
            Tensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize), E, Self>,
            Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), E, Self>,
        ),
        Self::Err,
    >;

    fn backward<
        const S: char,
        const THREE_HIDDEN_DIM: usize,
        const NUM_HEADS: usize,
        const HEAD_DIM: usize,
    >(
        &self,
        grad_inp: (
            &mut Self::Storage<(Const<NUM_HEADS>, Dyn<S>, Const<HEAD_DIM>), E>,
            &mut Self::Storage<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize), E>,
            &mut Self::Storage<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), E>,
        ),
        grad_out: (
            &Self::Storage<(Dyn<S>, Const<THREE_HIDDEN_DIM>), E>,
            &Self::Storage<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize), E>,
            &Self::Storage<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), E>,
        ),
    ) -> Result<(), Self::Err>;
}

impl<E: Dtype, D: AttentionReshapeKernel<E>> TryAttentionReshape<E> for D {
    /// Fallible version of [TryAttentionReshape::cat]
    fn try_attention_reshape<
        const S: char,
        const THREE_HIDDEN_DIM: usize,
        const NUM_HEADS: usize,
        const HEAD_DIM: usize,
    >(
        &self,
        qkv: &Tensor<(Dyn<S>, Const<THREE_HIDDEN_DIM>), E, Self>,
        past_key: &Tensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize), E, Self>,
        past_value: &Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), E, Self>,
    ) -> Result<
        (
            Tensor<(Const<NUM_HEADS>, Dyn<S>, Const<HEAD_DIM>), E, Self>,
            Tensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize), E, Self>,
            Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), E, Self>,
        ),
        Self::Err,
    > {
        let device = qkv.device.clone();
        device.forward(qkv, past_key, past_value)
    }
    // let mut tape: T = Default::default();
    // let (a, rhs): (Tensor<S1, E, Self>) = a.split_tape();
    // tape = tape.merge(rhs);

    // let (b, rhs): (Tensor<S2, E, Self>) = b.split_tape();
    // tape = tape.merge(rhs);

    // let device = a.device.clone();
    // let a_storage = &a.storage;
    // let b_storage = &b.storage;
    // let out = device.upgrade(device.forward(a_storage, b_storage)?);

    // let phantom_out = out.clone();

    // tape.try_alloc_grad(&a)?;
    // tape.try_alloc_grad(&b)?;
    // tape.try_alloc_grad(&out)?;
    // // tape.add_backward_op(move |grads| {
    // //     let (grad_inp, grad_out) = grads.many_and_ref(&tensors, &phantom_out);
    // //     device.backward(grad_inp, grad_out)?;
    // //     Ok(())
    // // });
    // Ok(out.put_tape(tape))
}

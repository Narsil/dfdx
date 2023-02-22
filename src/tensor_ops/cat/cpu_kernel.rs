use crate::{
    shapes::*,
    tensor::cpu::{Cpu, StridedArray},
};
use super::Catable;

impl<E: Dtype> super::CatKernel<E> for Cpu {
    fn forward<S1: Shape, S2: Shape + Catable<S1>>(
        &self,
        a: &Self::Storage<S1, E>,
        b: &Self::Storage<S2, E>,
    ) -> Result<Self::Storage<<S2 as Catable<S1>>::OutShape, E>, Self::Err>{
        // check that all the strides are the same
        // assert_eq!(a.strides, b.strides);

        let shape : <S2 as Catable<S1>>::OutShape = b.shape().cat_shape(a.shape()); 
        let numel = shape.num_elements();
        let mut data: std::vec::Vec<E> = std::vec::Vec::with_capacity(numel);
        data.extend_from_slice(a.data.as_ref());
        data.extend_from_slice(b.data.as_ref());
        let strides = shape.strides();
        Ok(StridedArray {
            data: std::sync::Arc::new(data),
            shape,
            strides,
        })
    }

    fn backward<S1: Shape, S2: Shape + Catable<S1>>(
        &self,
        grad_inp: (&mut Self::Storage<S1, E>, &mut Self::Storage<S2, E>),
        grad_out: &Self::Storage<<S2 as Catable<S1>>::OutShape, E>,
    ) -> Result<(), Self::Err>{
        todo!();
    }
}

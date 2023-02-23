use super::Catable;
use crate::{
    shapes::*,
    tensor::cuda::{Cuda, CudaArray},
};
use cudarc::driver::{LaunchAsync, LaunchConfig};
use std::{sync::Arc, vec::Vec};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/cat.ptx"));

trait HasCudaKernel<E> {
    const MOD: &'static str;
    const FNS: &'static [&'static str];
}

impl HasCudaKernel<f32> for Cuda {
    const MOD: &'static str = "cat_f32";
    const FNS: &'static [&'static str] = &["cat_f32"];
}

impl HasCudaKernel<f64> for Cuda {
    const MOD: &'static str = "cat_f64";
    const FNS: &'static [&'static str] = &["cat_f64"];
}

impl<E: Dtype> super::CatKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn forward<S1: Shape, S2: Shape + Catable<S1>>(
        &self,
        a: &Self::Storage<S1, E>,
        b: &Self::Storage<S2, E>,
    ) -> Result<Self::Storage<<S2 as Catable<S1>>::OutShape, E>, Self::Err> {
        let shape: <S2 as Catable<S1>>::OutShape = b.shape().cat_shape(a.shape());
        let numel = shape.num_elements();
        let mut data = unsafe { self.dev.alloc_async::<E>(numel) }?;

        assert_eq!(a.shape().strides(), a.strides);
        assert_eq!(b.shape().strides(), b.strides);

        let mut offset = 0;
        let item_numel = a.shape().num_elements();
        debug_assert_eq!(a.data.len(), item_numel);
        self.dev.device_copy_async(
            a.data.as_ref(),
            &mut data.try_slice_mut(offset..offset + item_numel).unwrap(),
        )?;
        offset += item_numel;

        let item_numel = b.shape().num_elements();
        debug_assert_eq!(b.data.len(), item_numel);
        self.dev.device_copy_async(
            b.data.as_ref(),
            &mut data.try_slice_mut(offset..offset + item_numel).unwrap(),
        )?;
        offset += item_numel;
        assert_eq!(offset, numel);

        let strides = shape.strides();
        Ok(CudaArray {
            data: std::sync::Arc::new(data),
            shape,
            strides,
        })
    }

    fn backward<S1: Shape, S2: Shape + Catable<S1>>(
        &self,
        grad_inp: (&mut Self::Storage<S1, E>, &mut Self::Storage<S2, E>),
        grad_out: &Self::Storage<<S2 as Catable<S1>>::OutShape, E>,
    ) -> Result<(), Self::Err> {
        todo!();
    }
}

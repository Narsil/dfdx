use crate::{
    gradients::{Merge, Tape},
    shapes::*,
    tensor::*,
};

mod cpu_kernel;
#[cfg(feature = "cuda")]
mod cuda_kernel;

/// Cat an array or vec of tensors together along a new dimension.
pub trait TryCat<E: Dtype>: DeviceStorage {
    // /// Cat an array or vec of tensors together along a new dimension.
    // ///
    // /// An array of tensors will be turned into a [Const] dim, and
    // /// a `Vec` of tensors will be turned into a [usize] dim.
    // ///
    // /// **Pytorch equivalent** `torch.cat`.
    // ///
    // /// Cating with an array:
    // /// ```rust
    // /// # use dfdx::prelude::*;
    // /// # let dev: Cpu = Default::default();
    // /// let a: Tensor<Rank2<3, 4>, f32, _> = dev.zeros();
    // /// let b: Tensor<Rank2<3, 4>, f32, _> = dev.zeros();
    // /// let _: Tensor<Rank3<2, 3, 4>, f32, _> = dev.cat([a, b]);
    // /// ```
    // ///
    // /// Cating with a vec:
    // /// ```rust
    // /// # use dfdx::prelude::*;
    // /// # let dev: Cpu = Default::default();
    // /// let a: Tensor<Rank2<3, 4>, f32, _> = dev.zeros();
    // /// let b: Tensor<Rank2<3, 4>, f32, _> = dev.zeros();
    // /// let _: Tensor<(usize, Const<3>, Const<4>), f32, _> = dev.cat(vec![a, b]);
    // /// ```
    fn cat<S1: Shape, S2: Shape, T>(
        &self,
        a: Tensor<S1, E, Self, T>,
        b: Tensor<S2, E, Self, T>,
    ) -> Tensor<<S2 as Catable<S1>>::OutShape, E, Self, T>
    where
        S2: Catable<S1>,
        T: Tape<Self> + Merge<T>{
            self.try_cat(a, b).unwrap()
        }

    /// Fallible version of [TryCat::cat]
    fn try_cat<S1: Shape, S2: Shape, T>(
        &self,
        a: Tensor<S1, E, Self, T>,
        b: Tensor<S2, E, Self, T>,
    ) -> Result<Tensor<<S2 as Catable<S1>>::OutShape, E, Self, T>, Self::Err>
    where
        S2: Catable<S1>,
        T: Tape<Self> + Merge<T>;
}

pub trait CatKernel<E: Dtype>: DeviceStorage {
    fn forward<S1: Shape, S2: Shape + Catable<S1>>(
        &self,
        a: &Self::Storage<S1, E>,
        b: &Self::Storage<S2, E>,
    ) -> Result<Self::Storage<<S2 as Catable<S1>>::OutShape, E>, Self::Err>;

    fn backward<S1: Shape, S2: Shape + Catable<S1>>(
        &self,
        grad_inp: (&mut Self::Storage<S1, E>, &mut Self::Storage<S2, E>),
        grad_out: &Self::Storage<<S2 as Catable<S1>>::OutShape, E>,
    ) -> Result<(), Self::Err>;
}

impl<E: Dtype, D: CatKernel<E>> TryCat<E> for D {
    fn try_cat<S1: Shape, S2: Shape + Catable<S1>, T>(
        &self,
        a: Tensor<S1, E, Self, T>,
        b: Tensor<S2, E, Self, T>,
    ) -> Result<Tensor<<S2 as Catable<S1>>::OutShape, E, Self, T>, Self::Err>
    where
        T: Tape<Self> + Merge<T>,
    {
        let mut tape: T = Default::default();
        let (a, rhs): (Tensor<S1, E, Self>, T) = a.split_tape();
        tape = tape.merge(rhs);

        let (b, rhs): (Tensor<S2, E, Self>, T) = b.split_tape();
        tape = tape.merge(rhs);


        let device = a.device.clone();
        let a_storage = &a.storage;
        let b_storage = &b.storage;
        let out = device.upgrade(device.forward(a_storage, b_storage)?);

        let phantom_out = out.clone();

        tape.try_alloc_grad(&a)?;
        tape.try_alloc_grad(&b)?;
        tape.try_alloc_grad(&out)?;
        // tape.add_backward_op(move |grads| {
        //     let (grad_inp, grad_out) = grads.many_and_ref(&tensors, &phantom_out);
        //     device.backward(grad_inp, grad_out)?;
        //     Ok(())
        // });
        Ok(out.put_tape(tape))
    }
}

pub trait Catable<S: Shape>: Shape {
    type OutShape: Shape; 

    fn cat_shape(&self, rhs: &S) -> Self::OutShape;
}

impl<A: Dim, B: Dim + core::ops::Add<A>> Catable<(A, )> for (B, ) where <B as core::ops::Add<A>>::Output: Dim{
    type OutShape = (<B as core::ops::Add<A>>::Output, );

    fn cat_shape(&self, rhs: &(A, )) -> Self::OutShape{
        (self.0 + rhs.0, )
    }
}

impl<D1: Dim, A: Dim, B: Dim + core::ops::Add<A>> Catable<(A, D1)> for (B, D1) where <B as core::ops::Add<A>>::Output: Dim{
    type OutShape = (<B as core::ops::Add<A>>::Output, D1);

    fn cat_shape(&self, rhs: &(A, D1)) -> Self::OutShape{
        (self.0 + rhs.0, rhs.1)
    }
}

impl<const A: char, const N: usize> core::ops::Add<Const<N>> for Dyn<A>{
    type Output = usize;
    fn add(self, rhs: Const<N>) -> Self::Output {
        self.size() + rhs.size()
    }
}

impl<const A: char, const N: usize> core::ops::Add<Dyn<A>> for Const<N> {
    type Output = usize;
    fn add(self, rhs: Dyn<A>) -> Self::Output {
        self.size() + rhs.size()
    }
}

impl<const A: char, const B: char> core::ops::Add<Dyn<A>> for Dyn<B> {
    type Output = usize;
    fn add(self, rhs: Dyn<A>) -> Self::Output {
        self.size() + rhs.size()
    }
}

impl<const M: usize> core::ops::Add<Const<M>> for usize {
    type Output = usize;
    fn add(self, rhs: Const<M>) -> Self::Output {
        self + rhs.size()
    }
}

impl<const M: usize> core::ops::Add<usize> for Const<M> {
    type Output = usize;
    fn add(self, rhs: usize) -> Self::Output {
        self.size() + rhs
    }
}

impl<const A: char> core::ops::Add<Dyn<A>> for usize {
    type Output = usize;
    fn add(self, rhs: Dyn<A>) -> Self::Output {
        self + rhs.size()
    }
}

impl<const A: char> core::ops::Add<usize> for Dyn<A> {
    type Output = usize;
    fn add(self, rhs: usize) -> Self::Output {
        self.size() + rhs
    }
}

#[cfg(feature="nightly")]
impl<const N: usize, const M: usize> core::ops::Add<Const<N>> for Const<M>
where
    Const<{ N + M }>: Sized,
{
    type Output = Const<{ N + M }>;
    fn add(self, rhs: Const<N>) -> Self::Output {
        Const
    }
}

#[cfg(not(feature="nightly"))]
impl<const N: usize, const M: usize> core::ops::Add<Const<N>> for Const<M>
{
    type Output = usize;
    fn add(self, rhs: Const<N>) -> Self::Output {
        self.size() + rhs.size()
    }
}


// pub trait CatKernel<E: Dtype>: DeviceStorage {
//     fn forward<S: Shape, Num: Dim>(
//         &self,
//         items: Vec<&Self::Storage<S, E>>,
//     ) -> Result<Self::Storage<S, E>, Self::Err>;
// 
//     fn backward<S: Shape, New: Dim>(
//         &self,
//         grad_inp: Vec<&mut Self::Storage<S, E>>,
//         grad_out: &Self::Storage<S, E>,
//     ) -> Result<(), Self::Err>;
// }
// 
// impl<E: Dtype, D: CatKernel<E>> TryCat<E> for D {
//     fn try_cat<S: Shape, T, Items>(
//         &self,
//         items: Items,
//     ) -> Result<Tensor<S, E, Self, T>, Self::Err>
//     where
//         Items: Catable<Tensor<S, E, Self, T>>,
//         T: Tape<Self> + Merge<T>,
//     {
//         let new_dim = items.dim();
//         assert!(new_dim.size() > 0);
// 
//         // need to split tape and transform into Vec for ease of implementation
//         let mut tensors = Vec::with_capacity(new_dim.size());
//         let mut tape: T = Default::default();
//         for item in items.into_iter() {
//             let (item, rhs): (Tensor<S, E, Self>, T) = item.split_tape();
//             tape = tape.merge(rhs);
//             tensors.push(item);
//         }
// 
//         // check that all the shapes are equal
//         let device = tensors[0].device.clone();
//         let shape = *tensors[0].shape();
//         for t in tensors.iter() {
//             assert_eq!(t.shape(), &shape);
//         }
// 
//         // we map to storage refs so kernels don't have to know about tensors
//         let storages: Vec<&D::Storage<S, E>> = tensors.iter().map(|t| &t.storage).collect();
//         let out = device.upgrade(device.forward(new_dim, storages)?);
// 
//         let phantom_out = out.clone();
//         for inp in tensors.iter() {
//             tape.try_alloc_grad(inp)?;
//         }
//         tape.try_alloc_grad(&out)?;
//         tape.add_backward_op(move |grads| {
//             let (grad_inp, grad_out) = grads.many_and_ref(&tensors, &phantom_out);
//             device.backward(grad_inp, grad_out)?;
//             Ok(())
//         });
//         Ok(out.put_tape(tape))
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gradients::NoneTape, tensor_ops::*, tests::*};
    use std::vec::Vec;

    #[test]
    fn test_shape_additions() {
        let a :Const<5> = Default::default();
        let b :Const<2> = Default::default();
        #[cfg(feature="nightly")]
        let _c: Const<7> = a + b;
        #[cfg(not(feature="nightly"))]
        let _c: Const<7> = (a + b).try_into().unwrap();
    }

    #[test]
    #[should_panic]
    #[cfg(not(feature="nightly"))]
    fn test_shape_additions_invalid() {
        let a :Const<5> = Default::default();
        let b :Const<2> = Default::default();
        let _c: Const<8> = (a + b).try_into().unwrap();
    }

    #[test]
    fn test_shape_dyn_additions() {
        let a: Dyn<'A'> = Dyn::<'A'>(3);
        let b :Const<2> = Default::default();

        let c = a + b;
        assert_eq!(c.size(), 5);

        let d = b + a;
        assert_eq!(d.size(), 5);

        let e: Dyn<'B'> = (a + b).into();
        assert_eq!(e.size(), 5);
    }

    #[test]
    fn test_shape_dyn_dyn_additions() {
        let a: Dyn<'A'> = Dyn::<'A'>(2);
        let b: Dyn<'B'> = Dyn::<'B'>(3);

        let c = a + b;
        assert_eq!(c.size(), 5);

        let d = b + a;
        assert_eq!(d.size(), 5);

        let e: Dyn<'C'> = (a + b).into();
        assert_eq!(e.size(), 5);
    }

    #[test]
    fn test_valid_cats() {
        let dev: TestDevice = Default::default();

        {
            let x: Tensor<Rank1<1>, TestDtype, _> = dev.sample_normal();
            let y: Tensor<Rank1<2>, TestDtype, _> = dev.sample_normal();
            #[cfg(feature="nightly")]
            {

                let mut c_vec: std::vec::Vec<f32> = x.array().to_vec();
                c_vec.extend(y.array());
                let c: Tensor<Rank1<3>, _, _> = dev.cat(x, y);

                assert_eq!(c.array().to_vec(), c_vec);
            }
            #[cfg(not(feature="nightly"))]
            {
                let mut c_vec = x.as_vec();
                c_vec.extend(y.as_vec());
                let c: Tensor<(usize, ), TestDtype, _> = dev.cat(x, y);
                assert_eq!(c.shape().0, 3);
                assert_eq!(c.as_vec(), c_vec);
            }

        }

        {
            let x: Tensor<Rank1<2>, TestDtype, _> = dev.sample_normal();
            let y: Tensor<Rank1<3>, TestDtype, _> = dev.sample_normal();
            let z: Tensor<Rank1<5>, TestDtype, _> = dev.sample_normal();
            #[cfg(feature="nightly")]
            {

                let mut c_vec: std::vec::Vec<f32> = x.array().to_vec();
                c_vec.extend(y.array());
                c_vec.extend(z.array());
                let c: Tensor<Rank1<10>, TestDtype, _> = dev.cat(dev.cat(x, y), z);
                assert_eq!(c.array().to_vec(), c_vec);
            }

            #[cfg(not(feature="nightly"))]
            {

                let mut c_vec: std::vec::Vec<f32> = x.array().to_vec();
                c_vec.extend(y.array());
                c_vec.extend(z.array());
                let c: Tensor<(usize,),  TestDtype, _> = dev.cat(dev.cat(x, y), z);
                assert_eq!(c.shape().0, 10);
                assert_eq!(c.as_vec(), c_vec);
            }
        }

    }

    #[test]
    fn test_valid_cats_2d() {
        let dev: TestDevice = Default::default();
        {
            let x: Tensor<Rank2<1, 3>, TestDtype, _> = dev.sample_normal();
            let y: Tensor<Rank2<2, 3>, TestDtype, _> = dev.sample_normal();
            let z: Tensor<Rank2<3, 3>, TestDtype, _> = dev.sample_normal();
            #[cfg(feature="nightly")]
            {

                let mut c_array: [[f32; 3]; 6] = [[0.0; 3]; 6];
                for (to, from) in c_array.iter_mut().zip(x.array().iter().chain(y.array().iter()).chain(z.array().iter())) { *to = *from }
                let c: Tensor<Rank2<6, 3>, TestDtype, _> = dev.cat(dev.cat(x, y), z);
                assert_eq!(c.array().to_vec(), c_array);
            }

            #[cfg(not(feature="nightly"))]
            {

                let mut c_vec: Vec<f32> = x.as_vec();
                c_vec.extend(y.as_vec());
                c_vec.extend(z.as_vec());
                let c: Tensor<(usize, Const<3>),  TestDtype, _> = dev.cat(dev.cat(x, y), z);
                assert_eq!(c.shape().0, 6);
                assert_eq!(c.as_vec(), c_vec);
            }
        }
    }

    // #[test]
    // #[should_panic]
    // fn test_cat_with_diff_sizes() {
    //     let dev: TestDevice = Default::default();
    //     let x: Tensor<_, TestDtype, _> = dev.sample_like(&(2, 3), rand_distr::StandardNormal);
    //     let y: Tensor<_, TestDtype, _> = dev.sample_like(&(3, 4), rand_distr::StandardNormal);
    //     let _ = dev.cat((x, y));
    // }

    // #[test]
    // #[should_panic]
    // fn test_cat_with_diff_strides() {
    //     let dev: TestDevice = Default::default();
    //     let x: Tensor<Rank2<2, 3>, TestDtype, _> = dev.sample_normal();
    //     let y: Tensor<Rank1<3>, TestDtype, _> = dev.sample_normal();
    //     let _ = dev.cat((x, y.broadcast()));
    // }

    // #[test]
    // fn test_cat_with_all_broadcasted() {
    //     let dev: TestDevice = Default::default();
    //     let x: Tensor<Rank1<2>, TestDtype, _> = dev.sample_normal();
    //     let y: Tensor<Rank1<3>, TestDtype, _> = dev.sample_normal();
    //     let r = dev.cat((
    //         x.trace(),
    //         y.trace(),
    //     ));
    //     assert_eq!(r.array(), x.array());
    //     let g = r.exp().mean().backward();
    //     let g1 = dev.cat((x.trace(), y.trace())).exp().mean().backward();
    //     assert_eq!(g.get(&x).array(), g1.get(&x).array());
    //     assert_eq!(g.get(&y).array(), g1.get(&y).array());
    // }

    // #[test]
    // fn test_cat_backwards() {
    //     let dev: TestDevice = Default::default();

    //     let x: Tensor<Rank2<1, 3>, TestDtype, _> = dev.sample_normal();
    //     let y: Tensor<Rank2<2, 3>, TestDtype, _> = dev.sample_normal();
    //     let z: Tensor<Rank2<3, 3>, TestDtype, _> = dev.sample_normal();
    //     let r = dev.cat((x.trace(), y.trace(), z.trace()));
    //     assert_eq!(r.array(), x.array());
    //     let r1 = r.retaped::<NoneTape>();
    //     let g1 = r1.trace().exp().mean().backward();
    //     let g = r.exp().mean().backward();
    //     let r_grad = g1.get(&r1).array();
    //     assert_eq!(r_grad[0], g.get(&x).array());
    //     assert_eq!(r_grad[1], g.get(&y).array());
    //     assert_eq!(r_grad[2], g.get(&z).array());
    // }
}

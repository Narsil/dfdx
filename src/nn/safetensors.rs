use crate::{
    shapes::{Dtype, Shape},
    tensor::{
        safetensors::{SafeDtype, SafeWriter},
        CopySlice, Tensor,
    },
};
use memmap2::MmapOptions;
use safetensors::tensor::{SafeTensorError, SafeTensors};

use super::tensor_collection::*;

use std::{path::Path, string::String};

/// Something that can be saved to a `.safetensors` (which is a `.zip`).
///
/// All [super::Module]s in nn implement SaveToSafeTensors, and the zips are formatted in a `.safetensors` fashion.
pub trait SaveToSafeTensors<E: Dtype + SafeDtype, D: CopySlice<E>>: TensorCollection<E, D> {
    fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> Result<(), SafeTensorError> {
        let mut writer = SafeWriter::new();
        self.write_safetensors(&mut writer)?;
        writer.save_safetensors(path.as_ref())?;
        Ok(())
    }

    fn write_safetensors(&self, w: &mut SafeWriter) -> Result<(), SafeTensorError> {
        Self::iter_tensors(&mut RecursiveWalker {
            m: self,
            f: w,
            path: &mut std::vec::Vec::new(),
        })
    }
}
impl<E: Dtype + SafeDtype, D: CopySlice<E>, T: TensorCollection<E, D>> SaveToSafeTensors<E, D>
    for T
{
}

/// Something that can be loaded from a `.safetensors` file.
///
/// All [super::Module]s in nn implement LoadFromSafeTensors, and the zips are formatted in a `.safetensors` fashion.
pub trait LoadFromSafeTensors<E: Dtype + SafeDtype, D: CopySlice<E>>:
    TensorCollection<E, D>
{
    /// Loads data from a `.safetensors` zip archive at the specified `path`.
    ///
    /// Example:
    /// ```ignore
    /// # use dfdx::prelude::*;
    /// let mut model: (Linear<5, 10>, Linear<10, 5>) = Default::default();
    /// model.load("tst.safetensors")?;
    /// ```
    fn load_safetensors<P: AsRef<Path>>(&mut self, path: P) -> Result<(), SafeTensorError> {
        let file = std::fs::File::open(path)?;
        let buffer = unsafe { MmapOptions::new().map(&file)? };
        let mut tensors = SafeTensors::deserialize(&buffer)?;
        self.read_safetensors(&mut tensors)?;
        Ok(())
    }

    fn read_safetensors<'data>(
        &mut self,
        tensors: &mut SafeTensors<'data>,
    ) -> Result<(), SafeTensorError> {
        Self::iter_tensors(&mut RecursiveWalker {
            m: self,
            f: tensors,
            path: &mut std::vec::Vec::new(),
        })
    }
}
impl<E: Dtype + SafeDtype, D: CopySlice<E>, T: TensorCollection<E, D>> LoadFromSafeTensors<E, D>
    for T
{
}

impl<E: Dtype + SafeDtype, D: CopySlice<E>> TensorVisitor<E, D> for SafeWriter {
    type Viewer = ViewTensorRef;
    type Err = SafeTensorError;

    fn visit<S: Shape>(
        &mut self,
        full_path: String,
        _: TensorOptions<S, E, D>,
        t: &Tensor<S, E, D>,
    ) -> Result<(), Self::Err> {
        self.add(full_path, t);
        Ok(())
    }
}

impl<'data, E: Dtype + SafeDtype, D: CopySlice<E>> TensorVisitor<E, D> for SafeTensors<'data> {
    type Viewer = ViewTensorMut;
    type Err = SafeTensorError;

    fn visit<S: Shape>(
        &mut self,
        full_path: String,
        _: TensorOptions<S, E, D>,
        t: &mut Tensor<S, E, D>,
    ) -> Result<(), Self::Err> {
        t.load_safetensors(self, &full_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        nn::{builders::*, *},
        shapes::*,
        tensor::{safetensors::SafeDtype, AsArray, SampleTensor, Tensor},
        tensor_ops::Device,
        tests::{TestDevice, TestDtype},
    };
    use rand_distr::{Distribution, Standard, StandardNormal};
    use tempfile::NamedTempFile;

    fn test_save_load<S: ConstShape, E: Dtype + SafeDtype, D: Device<E>, M: BuildOnDevice<D, E>>(
        dev: &D,
    ) where
        M::Built: Module<Tensor<S, E, D>> + SaveToSafeTensors<E, D> + LoadFromSafeTensors<E, D>,
        <M::Built as Module<Tensor<S, E, D>>>::Output: AsArray,
        StandardNormal: Distribution<E>,
    {
        let x = dev.sample_normal();
        let file = NamedTempFile::new().expect("failed to create tempfile");

        let saved: M::Built = M::build_on_device(dev);
        let mut loaded: M::Built = M::build_on_device(dev);

        let y = saved.forward(x.clone());

        assert_ne!(loaded.forward(x.clone()).array(), y.array());

        saved.save_safetensors(file.path()).expect("");
        loaded.load_safetensors(file.path()).expect("");

        assert_eq!(loaded.forward(x).array(), y.array());
    }

    #[test]
    fn test_batchnorm2d_save_load() {
        let dev: TestDevice = Default::default();
        type Model = BatchNorm2D<3>;

        let x: Tensor<Rank3<3, 4, 5>, TestDtype, _> = dev.sample_normal();
        let file = NamedTempFile::new().expect("failed to create tempfile");

        let mut saved = Model::build_on_device(&dev);
        let mut loaded = Model::build_on_device(&dev);

        saved.running_mean.fill_with_distr(Standard);
        saved.running_var.fill_with_distr(Standard);
        saved.scale.fill_with_distr(Standard);
        saved.bias.fill_with_distr(Standard);
        let y = saved.forward(x.clone());

        assert_ne!(loaded.forward(x.clone()).array(), y.array());

        saved.save(file.path()).expect("");
        loaded.load(file.path()).expect("");

        assert_eq!(loaded.forward(x).array(), y.array());
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_save_load_conv() {
        type T = Conv2D<2, 4, 3>;
        let dev: TestDevice = Default::default();
        test_save_load::<Rank3<2, 8, 8>, TestDtype, TestDevice, T>(&dev);
    }

    #[test]
    fn test_save_load_generalized_residual() {
        let dev: TestDevice = Default::default();
        type T = GeneralizedResidual<Linear<5, 5>, Linear<5, 5>>;
        test_save_load::<Rank1<5>, TestDtype, TestDevice, T>(&dev);
        test_save_load::<Rank1<5>, TestDtype, TestDevice, (T, T)>(&dev);
    }

    #[test]
    fn test_save_load_linear() {
        let dev: TestDevice = Default::default();
        type T = Linear<5, 5>;
        test_save_load::<Rank1<5>, TestDtype, TestDevice, T>(&dev);
        test_save_load::<Rank1<5>, TestDtype, TestDevice, (T, T)>(&dev);
    }

    #[test]
    fn test_save_load_tuple() {
        let dev: TestDevice = Default::default();
        type T = (
            (Linear<1, 2>, ReLU, Linear<2, 3>),
            (Dropout, Linear<3, 3>, Linear<3, 4>),
        );
        test_save_load::<Rank1<1>, TestDtype, TestDevice, T>(&dev);
    }

    #[test]
    fn test_save_load_layer_norm() {
        type M = LayerNorm1D<3>;
        let dev: TestDevice = Default::default();
        let x: Tensor<Rank1<3>, TestDtype, _> = dev.sample_normal();

        let file = NamedTempFile::new().expect("failed to create tempfile");

        let mut saved = M::build_on_device(&dev);
        let mut loaded = M::build_on_device(&dev);

        saved.gamma.fill_with_distr(Standard);
        saved.beta.fill_with_distr(Standard);
        let y = saved.forward(x.clone());

        assert_ne!(loaded.forward(x.clone()).array(), y.array());

        saved.save(file.path()).expect("");
        loaded.load(file.path()).expect("");

        assert_eq!(loaded.forward(x).array(), y.array());
    }

    #[test]
    fn test_save_load_repeated() {
        type T = Repeated<Linear<3, 3>, 4>;
        let dev: TestDevice = Default::default();
        test_save_load::<Rank1<3>, TestDtype, TestDevice, T>(&dev);
        test_save_load::<Rank1<3>, TestDtype, TestDevice, (T, T)>(&dev);
    }

    #[test]
    fn test_save_load_residual() {
        type T = Residual<Linear<5, 5>>;
        let dev: TestDevice = Default::default();
        test_save_load::<Rank1<5>, TestDtype, TestDevice, T>(&dev);
        test_save_load::<Rank1<5>, TestDtype, TestDevice, (T, T)>(&dev);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_save_load_mha() {
        let dev: TestDevice = Default::default();
        type Model = MultiHeadAttention<12, 4>;

        let saved = Model::build_on_device(&dev);

        let file = NamedTempFile::new().expect("failed to create tempfile");
        saved.save(file.path()).expect("");

        let mut loaded = Model::build_on_device(&dev);

        let q: Tensor<Rank3<2, 3, 12>, TestDtype, _> = dev.sample_normal();
        let k: Tensor<Rank3<2, 4, 12>, TestDtype, _> = dev.sample_normal();
        let v: Tensor<Rank3<2, 4, 12>, TestDtype, _> = dev.sample_normal();
        let y1 = saved.forward((q.clone(), k.clone(), v.clone()));

        let y2 = loaded.forward((q.clone(), k.clone(), v.clone()));
        assert_ne!(y1.array(), y2.array());

        loaded.load(file.path()).expect("");

        let y2 = loaded.forward((q.clone(), k.clone(), v.clone()));
        assert_eq!(y1.array(), y2.array());
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_save_load_transformer() {
        let dev: TestDevice = Default::default();
        type Model = Transformer<16, 4, 3, 4, 8>;

        let mut saved = Model::build_on_device(&dev);

        let file = NamedTempFile::new().expect("failed to create tempfile");
        saved.save(file.path()).expect("");

        let mut loaded = Model::build_on_device(&dev);

        let src: Tensor<Rank3<4, 12, 16>, TestDtype, _> = dev.sample_normal();
        let tgt: Tensor<Rank3<4, 6, 16>, TestDtype, _> = dev.sample_normal();
        let y1 = saved.forward_mut((src.clone(), tgt.clone()));

        let y2 = loaded.forward_mut((src.clone(), tgt.clone()));
        assert_ne!(y1.array(), y2.array());

        loaded.load(file.path()).expect("");

        let y2 = loaded.forward_mut((src.clone(), tgt.clone()));
        assert_eq!(y1.array(), y2.array());
    }
}

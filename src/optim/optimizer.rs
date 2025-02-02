use crate::{
    gradients::Gradients,
    shapes::{Dtype, Shape},
    tensor::{DeviceStorage, Tensor},
    unique_id::{HasUniqueId, UniqueId},
};

/// L2 and decoupled regularization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDecay<E> {
    /// Weight decay applied to the gradients before any momentum updates. Equivalent to L2 regularization.
    L2(E),

    /// Weight decay applied after any momentum updates, without modifying the gradients.
    /// See [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
    Decoupled(E),
}

/// Momentum used for [super::Sgd] and others
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Momentum<E> {
    /// Momentum that is applied to the velocity of a parameter directly.
    Classic(E),

    /// Momentum that is applied to both velocity and gradients. See [super::Sgd] nesterov paper for more.
    Nesterov(E),
}

/// All optimizers must implement the update function, which takes an object
/// that implements [GradientUpdate], and calls [GradientUpdate::update].
///
/// # Notes
///
/// 1. [GradientUpdate] requires an object that implements [crate::optim::ParamUpdater].
/// A common implementation involves implementing both [Optimizer] and [crate::optim::ParamUpdater]
/// on one struct, and passing self to [GradientUpdate::update]. See [super::Sgd] for an example
/// of implementing this trait.
///
/// 2. Update takes ownership of [Gradients], so update cannot be called
/// with the same gradients object.
///
/// 3. Optimizer itself is generic over M, not the update method. This means a single optimizer object
/// can only work on objects of type `M`. This also requires you to specify the model up front for the optimizer.
pub trait Optimizer<M: GradientUpdate<D, E>, D: DeviceStorage, E: Dtype> {
    /// Updates all of `module`'s parameters using `gradients`.
    ///
    /// Requires a `&mut self` because the optimizer may change some internally
    /// tracked values.
    fn update(
        &mut self,
        module: &mut M,
        gradients: Gradients<D>,
    ) -> Result<(), OptimizerUpdateError<D>>;
}

/// Represents something that can update a tensor.
///
/// See [crate::optim::Sgd] and [crate::optim::Adam] for examples on implementing this.
pub trait ParamUpdater<D: DeviceStorage, E: Dtype> {
    /// Retrieves the data associated with `p` if there is any.
    /// This can modify `self`, for instance if velocities are calculated
    /// based on the associated data!
    fn update_param<S: Shape>(
        &mut self,
        p: &mut Tensor<S, E, D>,
        unused: &mut UnusedTensors,
    ) -> Result<(), D::Err>;
}

/// Holds [UniqueId] of tensors that were missing gradients during
/// [GradientUpdate::update()], and therefore are unused
#[derive(Debug, Default)]
pub struct UnusedTensors {
    pub ids: std::vec::Vec<UniqueId>,
}

impl UnusedTensors {
    /// Adds a single unnammed parameter
    pub fn add<T: HasUniqueId>(&mut self, t: &T) {
        self.ids.push(*t.id());
    }

    /// Returns `true` if there are no missing gradients present
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// Represents something that can be updated with a [ParamUpdater].
pub trait GradientUpdate<D: DeviceStorage, E: Dtype>: Sized {
    /// Updates self given the [ParamUpdater].
    fn update<U>(&mut self, updater: &mut U, unused: &mut UnusedTensors) -> Result<(), D::Err>
    where
        U: ParamUpdater<D, E>;
}

impl<S: Shape, E: Dtype, D: DeviceStorage> GradientUpdate<D, E> for Tensor<S, E, D> {
    fn update<U: ParamUpdater<D, E>>(
        &mut self,
        opt: &mut U,
        unused: &mut UnusedTensors,
    ) -> Result<(), D::Err> {
        opt.update_param(self, unused)
    }
}

/// An error indicating that a parameter was not used in gradient
/// computation, and was therefore not present in [Gradients]
/// while a [GradientUpdate] was trying to update it.
#[derive(Debug)]
pub enum OptimizerUpdateError<D: DeviceStorage> {
    UnusedParams(UnusedTensors),
    DeviceError(D::Err),
}

impl<D: DeviceStorage> std::fmt::Display for OptimizerUpdateError<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnusedParams(unused) => write!(f, "Unused tensors: {:?}", unused),
            Self::DeviceError(err) => write!(f, "{err}"),
        }
    }
}

#[cfg(feature = "std")]
impl<D: DeviceStorage + std::fmt::Debug> std::error::Error for OptimizerUpdateError<D> {}

#[allow(clippy::from_over_into)]
impl<D: DeviceStorage> Into<Result<(), OptimizerUpdateError<D>>> for UnusedTensors {
    fn into(self) -> Result<(), OptimizerUpdateError<D>> {
        if self.is_empty() {
            Ok(())
        } else {
            Err(OptimizerUpdateError::UnusedParams(self))
        }
    }
}

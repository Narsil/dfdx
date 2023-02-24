//! Demonstrates how to save and load arrays with safetensors

#[cfg(feature = "safetensors")]
fn main() {
    use dfdx::{
        nn::{DeviceBuildExt, LoadFromSafeTensors, SaveToSafeTensors},
        prelude::Linear,
        shapes::{Rank0, Rank1, Rank2},
        tensor::safetensors::SafeWriter,
        tensor::{AsArray, Tensor, TensorFrom, ZerosTensor},
    };
    use safetensors::tensor::SafeTensors;
    #[cfg(not(feature = "cuda"))]
    type Device = dfdx::tensor::Cpu;

    #[cfg(feature = "cuda")]
    type Device = dfdx::tensor::Cuda;

    let dev: Device = Default::default();

    type Model = Linear<4, 2>;
    let m = dev.build_module::<Model, f32>();

    m.save_safetensors("linear.safetensors")
        .expect("Failed to write");

    let mut m2 = dev.build_module::<Model, f32>();
    assert_ne!(m.weight.array(), m2.weight.array());
    assert_ne!(m.bias.array(), m2.bias.array());
    m2.load_safetensors("linear.safetensors")
        .expect("Failed to load");
    assert_eq!(m.weight.array(), m2.weight.array());
    assert_eq!(m.bias.array(), m2.bias.array());

    let a = dev.tensor(1.234f32);
    let b = dev.tensor([1.0f32, 2.0, 3.0]);
    let c = dev.tensor([[1.0f32, 2.0, 3.0], [-1.0, -2.0, -3.0]]);

    let path = std::path::Path::new("out.safetensors");

    let mut w = SafeWriter::new();
    w.add("a".to_string(), &a);
    w.add("b".to_string(), &b);
    w.add("c".to_string(), &c);
    w.save_safetensors(path).unwrap();

    let mut a: Tensor<Rank0, f32, _> = dev.zeros();
    let mut b: Tensor<Rank1<3>, f32, _> = dev.zeros();
    let mut c: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();

    let filename = "out.safetensors";
    let buffer = std::fs::read(filename).expect("Couldn't read file");
    let tensors = SafeTensors::deserialize(&buffer).expect("Couldn't read safetensors file");
    a.load_safetensors(&tensors, "a").expect("Loading a failed");
    b.load_safetensors(&tensors, "b").expect("Loading b failed");
    c.load_safetensors(&tensors, "c").expect("Loading c failed");

    assert_eq!(a.array(), 1.234);
    assert_eq!(b.array(), [1.0, 2.0, 3.0]);
    assert_eq!(c.array(), [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
}

#[cfg(not(feature = "safetensors"))]
fn main() {
    panic!("Use the 'safetensors' feature to run this example");
}

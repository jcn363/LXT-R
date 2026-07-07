use ltx_types::*;
use tch::Kind;

#[test]
fn test_norm_layer_type_variants() {
    assert_eq!(NormLayerType::Group as i32, 0);
    assert_eq!(NormLayerType::Pixel as i32, 1);
    assert_eq!(NormLayerType::Group, NormLayerType::Group);
    assert_ne!(NormLayerType::Group, NormLayerType::Pixel);
}

#[test]
fn test_norm_layer_type_clone_copy() {
    let a = NormLayerType::Group;
    let b = a;
    let c = a;
    assert_eq!(a, b);
    assert_eq!(a, c);
}

#[test]
fn test_dtype_variants() {
    assert_eq!(DType::Float32 as i32, 0);
    assert_eq!(DType::Float16 as i32, 1);
    assert_eq!(DType::BFloat16 as i32, 2);
    assert_eq!(DType::Float8E4m3fn as i32, 3);
    assert_eq!(DType::Float32, DType::Float32);
    assert_ne!(DType::Float32, DType::Float16);
}

#[test]
fn test_dtype_clone_copy() {
    let a = DType::BFloat16;
    let b = a;
    let c = a;
    assert_eq!(a, b);
    assert_eq!(a, c);
}

#[test]
fn test_dtype_to_tch_kind() {
    assert_eq!(DType::Float32.to_tch_kind(), Kind::Float);
    assert_eq!(DType::Float16.to_tch_kind(), Kind::Half);
    assert_eq!(DType::BFloat16.to_tch_kind(), Kind::BFloat16);
    assert_eq!(DType::Float8E4m3fn.to_tch_kind(), Kind::BFloat16);
}

#[test]
fn test_dtype_parse_valid() {
    assert_eq!(DType::parse("float32"), Some(DType::Float32));
    assert_eq!(DType::parse("f32"), Some(DType::Float32));
    assert_eq!(DType::parse("float16"), Some(DType::Float16));
    assert_eq!(DType::parse("f16"), Some(DType::Float16));
    assert_eq!(DType::parse("bfloat16"), Some(DType::BFloat16));
    assert_eq!(DType::parse("bf16"), Some(DType::BFloat16));
    assert_eq!(DType::parse("float8_e4m3fn"), Some(DType::Float8E4m3fn));
    assert_eq!(DType::parse("fp8"), Some(DType::Float8E4m3fn));
}

#[test]
fn test_dtype_parse_invalid() {
    assert_eq!(DType::parse("invalid"), None);
    assert_eq!(DType::parse(""), None);
    assert_eq!(DType::parse("int8"), None);
    assert_eq!(DType::parse("float64"), None);
}

#[test]
fn test_dtype_debug_clone_copy() {
    let a = DType::Float32;
    let b = a;
    let c = a;
    assert_eq!(a, b);
    assert_eq!(a, c);
    let debug_str = format!("{:?}", a);
    assert!(debug_str.contains("Float32"));
}

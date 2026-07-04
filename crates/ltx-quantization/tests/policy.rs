use ltx_quantization::QuantizationPolicy;
use ltx_types::DType;

#[test]
fn test_default_policy() {
    let p = QuantizationPolicy::default();
    assert_eq!(p.weight_dtype, DType::BFloat16);
    assert!(!p.activate_fp8);
    assert!(p.group_size.is_none());
}

#[test]
fn test_fp8_per_tensor_policy() {
    let p = QuantizationPolicy::fp8_per_tensor();
    assert_eq!(p.weight_dtype, DType::Float8E4m3fn);
    assert!(!p.activate_fp8);
    assert!(p.group_size.is_none());
}

#[test]
fn test_is_fp8_true() {
    let p = QuantizationPolicy::fp8_per_tensor();
    assert!(p.is_fp8());
}

#[test]
fn test_is_fp8_false_default() {
    let p = QuantizationPolicy::default();
    assert!(!p.is_fp8());
}

#[test]
fn test_policy_clone() {
    let p = QuantizationPolicy::fp8_per_tensor();
    let p2 = p.clone();
    assert_eq!(p.weight_dtype, p2.weight_dtype);
    assert_eq!(p.activate_fp8, p2.activate_fp8);
}

#[test]
fn test_policy_debug() {
    let p = QuantizationPolicy::default();
    let debug_str = format!("{:?}", p);
    assert!(debug_str.contains("QuantizationPolicy"));
    assert!(debug_str.contains("BFloat16"));
}

#[test]
fn test_policy_deserialize() {
    let json = r#"{"weight_dtype": "Float8E4m3fn", "activate_fp8": true, "group_size": 128}"#;
    let p: QuantizationPolicy = serde_json::from_str(json).unwrap();
    assert_eq!(p.weight_dtype, DType::Float8E4m3fn);
    assert!(p.activate_fp8);
    assert_eq!(p.group_size, Some(128));
}

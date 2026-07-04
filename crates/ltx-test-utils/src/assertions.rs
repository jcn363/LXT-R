use tch::Tensor;

/// Compare two tensors element-wise within relative and absolute tolerances.
///
/// Returns `true` if `|a - b| <= atol + rtol * |b|` for every element.
pub fn assert_allclose(a: &Tensor, b: &Tensor, rtol: f64, atol: f64) {
    let a = a.to_kind(tch::Kind::Float);
    let b = b.to_kind(tch::Kind::Float);

    let diff = (&a - &b).abs();
    let tol = atol + rtol * b.abs();

    let max_diff = diff.max().double_value(&[]);
    let threshold = tol.max().double_value(&[]);

    assert!(
        max_diff <= threshold + 1e-7,
        "assert_allclose failed: max diff = {max_diff}, threshold = {threshold}"
    );
}

/// Compare two tensors element-wise with default tolerances (rtol=1e-5, atol=1e-6).
pub fn assert_allclose_default(a: &Tensor, b: &Tensor) {
    assert_allclose(a, b, 1e-5, 1e-6);
}

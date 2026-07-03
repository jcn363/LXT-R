use tch::Tensor;

/// Trait for items that can be added to a conditioning context.
pub trait ConditioningItem {
    /// The key name used to store this item in a batch dict.
    fn key(&self) -> &str;

    /// The tensor value of this conditioning item.
    fn tensor(&self) -> &Tensor;

    /// The batch index range this item applies to.
    fn batch_range(&self) -> (i64, i64);

    /// Whether this item should be scaled by a guidance factor.
    fn is_guided(&self) -> bool {
        true
    }
}

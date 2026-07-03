/// Preprocesses images for SigLIP: normalize to [-1, 1] range.
pub struct ImageProcessor {
    image_size: i64,
    mean: [f64; 3],
    std: [f64; 3],
}

impl ImageProcessor {
    pub fn new(image_size: i64) -> Self {
        Self {
            image_size,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        }
    }

    /// Normalize pixel values to [-1, 1] range expected by SigLIP.
    /// Input: (B, 3, H, W) with values in [0, 255].
    pub fn normalize(&self, pixel_values: &tch::Tensor) -> tch::Tensor {
        let device = pixel_values.device();
        let mean = tch::Tensor::from_slice(&self.mean)
            .to_device(device)
            .reshape([1, 3, 1, 1]);
        let std = tch::Tensor::from_slice(&self.std)
            .to_device(device)
            .reshape([1, 3, 1, 1]);

        let x = pixel_values.to_kind(tch::Kind::Float) / 255.0;
        (x - mean) / std
    }

    /// Preprocess a batch of images: normalize only.
    /// Resize should be handled by the image loading pipeline.
    pub fn preprocess(&self, pixel_values: &tch::Tensor) -> tch::Tensor {
        self.normalize(pixel_values)
    }

    pub fn image_size(&self) -> i64 {
        self.image_size
    }
}

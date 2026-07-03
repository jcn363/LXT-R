use tch::Tensor;

use crate::siglip::SigLIPVisionTower;

pub struct FeatureExtractor {
    vision_tower: SigLIPVisionTower,
}

impl FeatureExtractor {
    pub fn new(vision_tower: SigLIPVisionTower) -> Self {
        Self { vision_tower }
    }

    /// Extract features from preprocessed pixel values.
    /// Input: (B, 3, H, W) tensor normalized for SigLIP.
    /// Output: (B, num_patches + 1, hidden_size) tensor.
    pub fn forward(&self, pixel_values: &Tensor) -> Tensor {
        self.vision_tower.forward(pixel_values)
    }

    /// Extract spatial features only (excluding CLS token).
    /// Output: (B, num_patches, hidden_size).
    pub fn forward_spatial(&self, pixel_values: &Tensor) -> Tensor {
        let features = self.forward(pixel_values);
        features.narrow(1, 1, features.size()[1] - 1)
    }

    pub fn hidden_size(&self) -> i64 {
        self.vision_tower.hidden_size()
    }
}

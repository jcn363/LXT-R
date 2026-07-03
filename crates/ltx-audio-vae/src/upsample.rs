use std::borrow::Borrow;
use tch::nn::{ModuleT, Path};
use tch::Tensor;

use ltx_conv::AsymConvTranspose2d;
use ltx_resblock::ResnetBlock2D;
use ltx_types::NormLayerType;

/// A single upsampling stage: ConvTranspose2d → ResnetBlock2D.
pub struct UpsampleStage {
    pub(crate) conv: Box<dyn ModuleT>,
    pub(crate) resblock: ResnetBlock2D,
}

impl UpsampleStage {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.conv.forward_t(x, false);
        self.resblock.forward(&h)
    }
}

/// Build the decoder upsampling path.
///
/// Each stage consists of a strided `ConvTranspose2d` (via `make_conv_nd`)
/// to double the time dimension, followed by a `ResnetBlock2D`.
///
/// # Arguments
/// * `vs` — parameter namespace.
/// * `channels` — channel progression in reverse (from bottleneck to output),
///   e.g. `[1024, 512, 256, 128, 64]`.  Length determines the number of stages.
/// * `norm_type` — normalization type for ResnetBlock2D layers.
/// * `norm_groups` — number of groups for GroupNorm.
pub fn build_upsampling_path<'a>(
    vs: impl Borrow<Path<'a>>,
    channels: &[i64],
    norm_type: NormLayerType,
    norm_groups: i64,
) -> Vec<UpsampleStage> {
    let vs = vs.borrow();
    let num_stages = channels.len();
    let mut stages = Vec::with_capacity(num_stages);

    for i in 0..num_stages {
        let in_ch = channels[i];
        let out_ch = if i + 1 < num_stages { channels[i + 1] } else { channels[i] / 2 };

        // ConvTranspose2d: only upsample along Time, preserve Freq
        let conv: Box<dyn ModuleT> = Box::new(AsymConvTranspose2d::new(
            vs / format!("upsample_{i}"),
            in_ch,    // in_channels
            out_ch,   // out_channels
            4,        // kernel_time (upsamples time dim)
            1,        // kernel_freq (no change to freq dim)
            2,        // stride_time (doubles time dim)
            1,        // stride_freq (preserve freq dim)
        ));

        let resblock = ResnetBlock2D::new(
            vs / format!("resblock_{i}"),
            out_ch,
            out_ch,
            norm_type,
            norm_groups,
            true,
        );

        stages.push(UpsampleStage { conv, resblock });
    }

    stages
}

/// Run the full upsampling path on an input tensor.
///
/// # Arguments
/// * `x` — input tensor of shape `(B, C_in, T, F)`.
/// * `stages` — stages produced by `build_upsampling_path`.
///
/// # Returns
/// Tensor of shape `(B, C_out, T', F)` where `T'` is the upsampled
/// time dimension.
pub fn upsample_forward(x: &Tensor, stages: &[UpsampleStage]) -> Tensor {
    let mut h = x.shallow_clone();
    for stage in stages {
        h = stage.forward(&h);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upsampling_path_shapes() {
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let root = vs.root();
        let channels = [256, 128, 64];
        let stages =
            build_upsampling_path(root / "up", &channels, NormLayerType::Group, ltx_types::VAE_NORM_NUM_GROUPS);
        assert_eq!(stages.len(), 3);

        let x = Tensor::randn([1, 256, 16, 128], (tch::Kind::Float, tch::Device::Cpu));
        let y = upsample_forward(&x, &stages);
        // After 3 upsample stages (each doubles time dim):
        // 16 → 32 → 64 → 128
        assert_eq!(y.size(), vec![1, 32, 128, 128]);
    }
}

use std::borrow::Borrow;
use tch::nn::Path;
use tch::Tensor;

use ltx_conv::CausalConv2d;
use ltx_resblock::ResnetBlock2D;
use ltx_types::NormLayerType;

/// A single downsampling stage: ResnetBlock2D → optional CausalConv2d stride-2.
pub struct DownsampleStage {
    pub(crate) resblock: ResnetBlock2D,
    pub(crate) conv: Option<CausalConv2d>,
}

impl DownsampleStage {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.resblock.forward(x);
        match &self.conv {
            Some(c) => c.forward(&h, true),
            None => h,
        }
    }
}

/// Build the encoder downsampling path.
///
/// Each stage consists of a `ResnetBlock2D` followed by an optional
/// `CausalConv2d` with `stride=2` to halve the time dimension.
///
/// # Arguments
/// * `vs` — parameter namespace.
/// * `channels` — channel progression, e.g. `[64, 128, 256, 512, 1024]`.
///   Length determines the number of stages.  Downsampling happens at
///   every stage except the last.
/// * `norm_type` — normalization type for ResnetBlock2D layers.
/// * `norm_groups` — number of groups for GroupNorm.
pub fn build_downsampling_path<'a>(
    vs: impl Borrow<Path<'a>>,
    channels: &[i64],
    norm_type: NormLayerType,
    norm_groups: i64,
) -> Vec<DownsampleStage> {
    let vs = vs.borrow();
    let num_stages = channels.len();
    let mut stages = Vec::with_capacity(num_stages);

    for i in 0..num_stages {
        let in_ch = if i == 0 { channels[0] } else { channels[i - 1] };
        let out_ch = channels[i];

        let resblock =
            ResnetBlock2D::new(vs / format!("resblock_{i}"), in_ch, out_ch, norm_type, norm_groups, true);

        let conv = if i < num_stages - 1 {
            Some(CausalConv2d::new_with_axes(
                vs / format!("downsample_{i}"),
                out_ch,
                out_ch,
                4,  // kernel_time (strided)
                1,  // kernel_freq (no downsampling)
                2,  // stride_time (halves time dim)
                1,  // stride_freq (preserve freq dim)
                ltx_conv::CausalityAxis::Time,
            ))
        } else {
            None
        };

        stages.push(DownsampleStage { resblock, conv });
    }

    stages
}

/// Run the full downsampling path on an input tensor.
///
/// # Arguments
/// * `x` — input tensor of shape `(B, C_in, T, F)`.
/// * `stages` — stages produced by `build_downsampling_path`.
///
/// # Returns
/// Tensor of shape `(B, C_out, T', F)` where `T'` is the downsampled
/// time dimension.
pub fn downsample_forward(x: &Tensor, stages: &[DownsampleStage]) -> Tensor {
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
    fn test_downsampling_path_shapes() {
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let root = vs.root();
        let channels = [64, 128, 256];
        let stages = build_downsampling_path(
            root / "down",
            &channels,
            NormLayerType::Group,
            ltx_types::VAE_NORM_NUM_GROUPS,
        );
        assert_eq!(stages.len(), 3);

        let x = Tensor::randn([1, 64, 64, 128], (tch::Kind::Float, tch::Device::Cpu));
        let y = downsample_forward(&x, &stages);
        // After 2 downsample stages (index 0 and 1 have stride-2 conv):
        // 64 → 32 → 16
        assert_eq!(y.size(), vec![1, 256, 16, 128]);
    }
}

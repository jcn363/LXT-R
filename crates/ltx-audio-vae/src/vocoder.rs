use std::borrow::Borrow;
use tch::nn::{Conv1D, ConvConfig, Module, ModuleT, Path};
use tch::Tensor;

use ltx_resblock::ResBlock1;
use ltx_types::NormLayerType;

/// Wrap `tch::nn::ConvTransposeND` to implement `ModuleT`.
#[derive(Debug)]
struct ConvTranspose1dModule(tch::nn::ConvTransposeND<[i64; 1]>);

impl ModuleT for ConvTranspose1dModule {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.0.forward(xs)
    }
}

/// Wrap `tch::nn::Conv1D` to implement `ModuleT`.
#[derive(Debug)]
struct Conv1DModule(Conv1D);

impl ModuleT for Conv1DModule {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.0.forward(xs)
    }
}

/// THE ONLY Vocoder in the codebase.
///
/// Converts latent audio features to raw waveform samples using
/// `ConvTranspose1d` layers for upsampling and `ResBlock1` layers
/// for refinement.  This is the final stage of the audio synthesis
/// pipeline.
///
/// Architecture:
/// ```text
/// Input (B, C_in, T)
///   → ConvTranspose1d × N  (each doubles T)
///   → ResBlock1 × M        (refinement at final resolution)
///   → LeakyReLU
///   → Conv1d → (B, 1, T_out)
/// ```
pub struct Vocoder {
    upsample_convs: Vec<ConvTranspose1dModule>,
    res_blocks: Vec<ResBlock1>,
    final_conv: Conv1DModule,
    activation_slope: f64,
}

impl std::fmt::Debug for Vocoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vocoder")
            .field("upsample_count", &self.upsample_convs.len())
            .field("res_block_count", &self.res_blocks.len())
            .finish()
    }
}

impl Vocoder {
    /// Create a new Vocoder.
    ///
    /// # Arguments
    /// * `vs` — parameter namespace.
    /// * `in_channels` — number of input latent channels.
    /// * `upsample_channels` — channel progression for transposed convolution
    ///   upsampling layers.  Each entry creates one `ConvTranspose1d` layer
    ///   that doubles the time dimension.
    /// * `res_block_channels` — channel count for the refinement `ResBlock1`
    ///   layers applied after all upsampling.
    /// * `num_res_blocks` — number of `ResBlock1` refinement layers.
    /// * `negative_slope` — LeakyReLU slope; use 0.0 for SiLU in ResBlock1.
    pub fn new<'a>(
        vs: impl Borrow<Path<'a>>,
        in_channels: i64,
        upsample_channels: &[i64],
        res_block_channels: i64,
        num_res_blocks: i64,
        negative_slope: f64,
    ) -> Self {
        let vs = vs.borrow();

        // Build transposed convolution upsampling layers
        let mut upsample_convs = Vec::with_capacity(upsample_channels.len());
        let mut prev_ch = in_channels;
        for (i, &out_ch) in upsample_channels.iter().enumerate() {
            let cfg = tch::nn::ConvTransposeConfig {
                stride: 2,
                padding: 1,
                output_padding: 0,
                groups: 1,
                bias: true,
                dilation: 1,
                ..Default::default()
            };
            let conv =
                tch::nn::conv_transpose1d(vs / format!("upsample_{i}"), prev_ch, out_ch, 4, cfg);
            upsample_convs.push(ConvTranspose1dModule(conv));
            prev_ch = out_ch;
        }

        // Build refinement residual blocks
        let mut res_blocks = Vec::with_capacity(num_res_blocks as usize);
        for i in 0..num_res_blocks {
            let block = ResBlock1::new(
                vs / format!("resblock_{i}"),
                res_block_channels,
                3, // kernel_size
                NormLayerType::Group,
                ltx_types::VAE_NORM_NUM_GROUPS,
                negative_slope,
            );
            res_blocks.push(block);
        }

        // Final 1×1 convolution to single output channel
        let final_conv = tch::nn::conv1d(
            vs / "final_conv",
            res_block_channels,
            1,
            7,
            ConvConfig {
                padding: 3,
                ..Default::default()
            },
        );

        Self {
            upsample_convs,
            res_blocks,
            final_conv: Conv1DModule(final_conv),
            activation_slope: negative_slope,
        }
    }

    /// Forward pass: latent features → waveform.
    ///
    /// # Arguments
    /// * `x` — input tensor of shape `(B, C_in, T)`.
    ///
    /// # Returns
    /// Audio tensor of shape `(B, 1, T_out)` where
    /// `T_out = T × 2^num_upsample_layers`.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = x.shallow_clone();

        // Upsample through transposed convolutions
        for conv in &self.upsample_convs {
            h = conv.forward_t(&h, false);
        }

        // Refinement through residual blocks
        for block in &self.res_blocks {
            h = block.forward(&h);
        }

        // Activation
        let h = if self.activation_slope > 0.0 {
            let positive = h.clamp_min(0.0);
            let negative = h.clamp_max(0.0) * self.activation_slope;
            positive + negative
        } else {
            h.silu()
        };

        // Final projection to single channel
        self.final_conv.forward_t(&h, false)
    }
}

impl ModuleT for Vocoder {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocoder_forward() {
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let root = vs.root();
        let vocoder = Vocoder::new(
            root / "vocoder",
            64,         // in_channels
            &[128, 64], // upsample_channels (2 upsample layers)
            64,         // res_block_channels
            2,          // num_res_blocks
            ltx_types::LRELU_SLOPE,
        );

        let x = Tensor::randn([1, 64, 32], (tch::Kind::Float, tch::Device::Cpu));
        let y = vocoder.forward(&x);
        // 32 → 64 → 128 after 2 upsample layers, output has 1 channel
        assert_eq!(y.size(), vec![1, 1, 128]);
    }
}

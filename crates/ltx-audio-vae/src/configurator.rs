use serde::Deserialize;
use std::borrow::Borrow;
use tch::nn::{Conv2D, Module, ModuleT, Path};
use tch::Tensor;

use ltx_attention::SimpleAttnBlock;
use ltx_conv::{CausalConv2d, CausalityAxis};
use ltx_norm::GroupNorm;
use ltx_resblock::ResnetBlock2D;
use ltx_types::{NormLayerType, VAE_NORM_NUM_GROUPS};

use crate::downsample::DownsampleStage;
use crate::upsample::UpsampleStage;

/// Configuration for the Audio VAE (encoder + decoder).
#[derive(Debug, Clone, Deserialize)]
pub struct AudioVAEConfig {
    /// Channel counts for the encoder downsampling path.
    /// Length determines the number of encoder stages.
    pub encoder_channels: Vec<i64>,
    /// Channel counts for the decoder upsampling path (in reverse order).
    pub decoder_channels: Vec<i64>,
    /// Number of groups for GroupNorm in ResnetBlock2D layers.
    pub norm_groups: i64,
    /// Number of attention blocks in the encoder mid-section.
    pub num_mid_attention: i64,
    /// Feature dimension of the input audio (e.g. 128 for mel spectrograms).
    pub input_features: i64,
    /// Number of output latent channels.
    pub latent_channels: i64,
}

impl Default for AudioVAEConfig {
    fn default() -> Self {
        Self {
            encoder_channels: vec![64, 128, 256, 512, 1024],
            decoder_channels: vec![1024, 512, 256, 128, 64],
            norm_groups: VAE_NORM_NUM_GROUPS,
            num_mid_attention: 2,
            input_features: 128,
            latent_channels: 64,
        }
    }
}

/// Wrap `tch::nn::Conv2D` to implement `ModuleT`.
#[derive(Debug)]
struct Conv2DModule(Conv2D);

impl ModuleT for Conv2DModule {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.0.forward(xs)
    }
}

/// THE ONLY AudioEncoder in the codebase.
///
/// Compresses raw audio into a latent representation using convolutional
/// downsampling, ResnetBlock2D refinement, and optional attention blocks.
pub struct AudioEncoder {
    conv_in: Conv2DModule,
    downsample_stages: Vec<DownsampleStage>,
    mid_attention: Vec<SimpleAttnBlock>,
    norm_out: GroupNorm,
    conv_out: Conv2DModule,
}

impl std::fmt::Debug for AudioEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioEncoder").finish()
    }
}

impl AudioEncoder {
    /// Create a new AudioEncoder from a configuration.
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &AudioVAEConfig) -> Self {
        let vs = vs.borrow();
        let first_ch = config.encoder_channels[0];
        let last_ch = *config.encoder_channels.last().unwrap();

        let conv_in = Conv2DModule(tch::nn::conv2d(
            vs / "conv_in",
            config.input_features,
            first_ch,
            3,
            tch::nn::ConvConfig {
                padding: 1,
                ..Default::default()
            },
        ));

        let num_stages = config.encoder_channels.len();
        let mut downsample_stages = Vec::with_capacity(num_stages);
        for i in 0..num_stages {
            let in_ch = if i == 0 { first_ch } else { config.encoder_channels[i - 1] };
            let out_ch = config.encoder_channels[i];

            let resblock = ResnetBlock2D::new(
                vs / format!("down_{i}") / "resblock",
                in_ch,
                out_ch,
                NormLayerType::Group,
                config.norm_groups,
                true,
            );

            let conv = if i < num_stages - 1 {
                Some(CausalConv2d::new_with_axes(
                    vs / format!("down_{i}") / "downsample",
                    out_ch,
                    out_ch,
                    4, 1, 2, 1,
                    CausalityAxis::Time,
                ))
            } else {
                None
            };

            downsample_stages.push(DownsampleStage { resblock, conv });
        }

        let mut mid_attention = Vec::with_capacity(config.num_mid_attention as usize);
        for _ in 0..config.num_mid_attention {
            mid_attention.push(SimpleAttnBlock::new(last_ch));
        }

        let norm_out = GroupNorm::with_defaults(config.norm_groups, last_ch);

        let conv_out = Conv2DModule(tch::nn::conv2d(
            vs / "conv_out",
            last_ch,
            config.latent_channels,
            3,
            tch::nn::ConvConfig {
                padding: 1,
                ..Default::default()
            },
        ));

        Self { conv_in, downsample_stages, mid_attention, norm_out, conv_out }
    }

    /// Encode audio to latent representation.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.conv_in.forward_t(x, false);
        let mut h = h;
        for stage in &self.downsample_stages {
            h = stage.forward(&h);
        }
        for attn in &self.mid_attention {
            h = attn.forward(&h);
        }
        let h = self.norm_out.forward(&h).silu();
        self.conv_out.forward_t(&h, false)
    }
}

impl ModuleT for AudioEncoder {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs)
    }
}

/// THE ONLY AudioDecoder in the codebase.
///
/// Reconstructs audio from a latent representation using convolutional
/// upsampling, ResnetBlock2D refinement, and optional attention blocks.
pub struct AudioDecoder {
    conv_in: Conv2DModule,
    upsample_stages: Vec<UpsampleStage>,
    mid_attention: Vec<SimpleAttnBlock>,
    norm_out: GroupNorm,
    conv_out: Conv2DModule,
}

impl std::fmt::Debug for AudioDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioDecoder").finish()
    }
}

impl AudioDecoder {
    /// Create a new AudioDecoder from a configuration.
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &AudioVAEConfig) -> Self {
        let vs = vs.borrow();
        let first_ch = config.decoder_channels[0];
        let last_ch = *config.decoder_channels.last().unwrap();

        let conv_in = Conv2DModule(tch::nn::conv2d(
            vs / "conv_in",
            config.latent_channels,
            first_ch,
            3,
            tch::nn::ConvConfig {
                padding: 1,
                ..Default::default()
            },
        ));

        let num_stages = config.decoder_channels.len();
        let mut upsample_stages = Vec::with_capacity(num_stages);
        for i in 0..num_stages {
            let in_ch = config.decoder_channels[i];
            let out_ch = if i + 1 < num_stages {
                config.decoder_channels[i + 1]
            } else {
                last_ch
            };

            let conv: Box<dyn ModuleT> = Box::new(ltx_conv::AsymConvTranspose2d::new(
                vs / format!("up_{i}") / "upsample",
                in_ch,
                out_ch,
                4,  // kernel_time
                1,  // kernel_freq
                2,  // stride_time
                1,  // stride_freq
            ));

            let resblock = ResnetBlock2D::new(
                vs / format!("up_{i}") / "resblock",
                out_ch,
                out_ch,
                NormLayerType::Group,
                config.norm_groups,
                true,
            );

            upsample_stages.push(UpsampleStage { conv, resblock });
        }

        let mut mid_attention = Vec::with_capacity(config.num_mid_attention as usize);
        for _ in 0..config.num_mid_attention {
            mid_attention.push(SimpleAttnBlock::new(last_ch));
        }

        let norm_out = GroupNorm::with_defaults(config.norm_groups, last_ch);

        let conv_out = Conv2DModule(tch::nn::conv2d(
            vs / "conv_out",
            last_ch,
            config.input_features,
            3,
            tch::nn::ConvConfig {
                padding: 1,
                ..Default::default()
            },
        ));

        Self {
            conv_in,
            upsample_stages,
            mid_attention,
            norm_out,
            conv_out,
        }
    }

    /// Decode latent representation to audio.
    ///
    /// Input: `(B, latent_channels, T', F)`.
    /// Output: `(B, input_features, T, F)`.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.conv_in.forward_t(x, false);

        let mut h = h;
        for stage in &self.upsample_stages {
            h = stage.forward(&h);
        }

        for attn in &self.mid_attention {
            h = attn.forward(&h);
        }

        let h = self.norm_out.forward(&h);
        let h = h.silu();
        self.conv_out.forward_t(&h, false)
    }
}

impl ModuleT for AudioDecoder {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs)
    }
}

/// Construct an `AudioEncoder` and `AudioDecoder` from a configuration.
///
/// Both models share the same `VarStore` for parameter management.
pub fn from_config(config: &AudioVAEConfig) -> (AudioEncoder, AudioDecoder) {
    let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let encoder = AudioEncoder::new(vs.root() / "encoder", config);
    let decoder = AudioDecoder::new(vs.root() / "decoder", config);
    (encoder, decoder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_encoder_decoder_roundtrip_shapes() {
        let config = AudioVAEConfig::default();
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);

        let encoder = AudioEncoder::new(vs.root() / "encoder", &config);
        let decoder = AudioDecoder::new(vs.root() / "decoder", &config);

        let x = Tensor::randn([1, 128, 64, 128], (tch::Kind::Float, tch::Device::Cpu));

        // Debug: trace encoder shapes
        let h = encoder.conv_in.forward_t(&x, false);
        eprintln!("conv_in: {:?}", h.size());
        let mut h = h;
        for (i, stage) in encoder.downsample_stages.iter().enumerate() {
            h = stage.resblock.forward(&h);
            eprintln!("after resblock {i}: {:?}", h.size());
            if let Some(ref c) = stage.conv {
                h = c.forward(&h, true);
                eprintln!("after downsample {i}: {:?}", h.size());
            }
        }
        eprintln!("before attention: {:?}", h.size());
        for (i, attn) in encoder.mid_attention.iter().enumerate() {
            h = attn.forward(&h);
            eprintln!("after attention {i}: {:?}", h.size());
        }
        h = encoder.norm_out.forward(&h);
        h = h.silu();
        let latent = encoder.conv_out.forward_t(&h, false);
        eprintln!("latent: {:?}", latent.size());

        assert_eq!(latent.size()[0], 1);
        assert_eq!(latent.size()[1], config.latent_channels);
        assert_eq!(latent.size()[3], 128);

        let reconstructed = decoder.forward(&latent);
        assert_eq!(reconstructed.size()[0], 1);
        assert_eq!(reconstructed.size()[1], config.input_features);
        assert_eq!(reconstructed.size()[3], 128);
    }
}

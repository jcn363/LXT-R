use tch::nn::Path;

use ltx_types::NormLayerType;

use crate::{
    EncoderBlockDesc, EncoderBlockKind, VideoDecoder, VideoEncoder, VideoVAE,
};

/// space_to_depth ratio for RGB input: r=4 gives 3*4*4 = 48 input channels.
pub const SPACE_TO_DEPTH_R: i64 = 4;
pub const CONV_IN_CHANNELS: i64 = 3 * SPACE_TO_DEPTH_R * SPACE_TO_DEPTH_R; // 48

/// Encoder conv_out outputs 129 channels (128 sampled latent + 1 scale).
pub const ENCODER_CONV_OUT_CHANNELS: i64 = 129;
/// Latent channels after sampling (first 128 of 129).
pub const SAMPLED_LATENT_CHANNELS: i64 = 128;

/// Default encoder block descriptors matching the Python LTX-Video VAE.
///
/// 10 blocks:
///   0: ResBlocks x4, ch=128
///   1: DownsampleConv, in=128, out=128
///   2: ChannelChangeDownsample, in=128, out=256
///   3: ResBlocks x3, ch=256
///   4: DownsampleConv, in=256, out=256
///   5: ChannelChangeDownsample, in=256, out=512
///   6: ResBlocks x3, ch=512
///   7: DownsampleConv, in=512, out=512
///   8: ResBlocks x3, ch=512
///   9: ResBlocks x4, ch=512
pub fn default_encoder_block_descs() -> Vec<EncoderBlockDesc> {
    use EncoderBlockKind::*;
    vec![
        EncoderBlockDesc { kind: ResBlocks(4), in_ch: 128, out_ch: 128 },
        EncoderBlockDesc { kind: DownsampleConv, in_ch: 128, out_ch: 128 },
        EncoderBlockDesc { kind: ChannelChangeDownsample, in_ch: 128, out_ch: 256 },
        EncoderBlockDesc { kind: ResBlocks(3), in_ch: 256, out_ch: 256 },
        EncoderBlockDesc { kind: DownsampleConv, in_ch: 256, out_ch: 256 },
        EncoderBlockDesc { kind: ChannelChangeDownsample, in_ch: 256, out_ch: 512 },
        EncoderBlockDesc { kind: ResBlocks(3), in_ch: 512, out_ch: 512 },
        EncoderBlockDesc { kind: DownsampleConv, in_ch: 512, out_ch: 512 },
        EncoderBlockDesc { kind: ResBlocks(3), in_ch: 512, out_ch: 512 },
        EncoderBlockDesc { kind: ResBlocks(4), in_ch: 512, out_ch: 512 },
    ]
}

/// Build encoder from config — the only place encoder construction happens.
pub fn build_encoder(vs: &Path, norm_type: NormLayerType, norm_groups: i64, causal: bool) -> VideoEncoder {
    let block_descs = default_encoder_block_descs();
    VideoEncoder::new(
        vs,
        CONV_IN_CHANNELS,
        128, // base_channels
        &block_descs,
        ENCODER_CONV_OUT_CHANNELS,
        norm_type,
        norm_groups,
        causal,
    )
}

/// Build decoder from config — the only place decoder construction happens.
pub fn build_decoder(vs: &Path, norm_type: NormLayerType, norm_groups: i64, causal: bool) -> VideoDecoder {
    VideoDecoder::new(
        vs,
        SAMPLED_LATENT_CHANNELS, // 128 — decoder takes sampled latent
        1024,                     // base_channels (first resblock stage)
        norm_type,
        norm_groups,
        causal,
    )
}

/// Build full VideoVAE from VarStore root.
pub fn build_video_vae(vs: &Path, norm_type: NormLayerType, norm_groups: i64, causal: bool) -> VideoVAE {
    let encoder = build_encoder(&(vs / "encoder"), norm_type, norm_groups, causal);
    let decoder = build_decoder(&(vs / "decoder"), norm_type, norm_groups, causal);
    VideoVAE::new_encoder_decoder(encoder, decoder, 32)
}

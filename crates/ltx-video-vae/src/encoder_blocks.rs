use std::borrow::Borrow;

use tch::nn::{ModuleT, Path};

use ltx_conv::make_conv_nd;
use ltx_resblock::make_resblock;
use ltx_types::NormLayerType;

/// Components of one encoder stage — the `_make_encoder_block` dispatcher
/// assembles these from shared primitives.
pub struct EncoderBlock {
    pub conv: Box<dyn ModuleT>,
    pub resblock: Box<dyn ModuleT>,
}

/// THE ONLY encoder-block factory in the codebase.
///
/// Creates a convolution + residual block pair for one encoder stage.
/// Uses `ltx_conv::make_conv_nd` and `ltx_resblock::make_resblock` — never
/// reimplements convolution or residual logic.
pub fn make_encoder_block<'a>(
    vs: impl Borrow<Path<'a>>,
    in_channels: i64,
    out_channels: i64,
    stride: i64,
    norm_type: NormLayerType,
    norm_groups: i64,
    causal: bool,
) -> EncoderBlock {
    let vs = vs.borrow();
    let conv = make_conv_nd(
        vs / "conv",
        3,
        in_channels,
        out_channels,
        3,
        stride,
        1,
        causal,
        "zeros",
    );
    let resblock = make_resblock(
        3,
        out_channels,
        out_channels,
        norm_type,
        norm_groups,
        causal,
        vs / "resblock",
    );

    EncoderBlock { conv, resblock }
}

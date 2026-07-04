use tch::nn::{ModuleT, Path};
use tch::Tensor;

use ltx_conv::make_conv_nd;
use ltx_norm::build_norm_layer;
use ltx_types::NormLayerType;

/// THE ONLY ResnetBlock3D in the codebase.
///
/// Standard residual block for the video VAE encoder/decoder:
/// `x + conv2(silu(norm2(conv1(silu(norm1(x))))))`
///
/// When `in_channels != out_channels`, a 1x1x1 projection shortcut is used.
pub struct ResnetBlock3D {
    norm1: Box<dyn ModuleT>,
    conv1: Box<dyn ModuleT>,
    norm2: Box<dyn ModuleT>,
    conv2: Box<dyn ModuleT>,
    shortcut: Box<dyn ModuleT>,
}

impl std::fmt::Debug for ResnetBlock3D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResnetBlock3D").finish()
    }
}

impl ResnetBlock3D {
    pub fn new<'a>(
        vs: impl std::borrow::Borrow<Path<'a>>,
        in_channels: i64,
        out_channels: i64,
        norm_type: NormLayerType,
        norm_groups: i64,
        causal: bool,
    ) -> Self {
        let vs = vs.borrow();

        let norm1 = build_norm_layer(norm_type, in_channels, norm_groups);
        let conv1 = make_conv_nd(
            vs / "conv1",
            3,
            in_channels,
            out_channels,
            3,
            1,
            1,
            causal,
            "zeros",
        );
        let norm2 = build_norm_layer(norm_type, out_channels, norm_groups);
        let conv2 = make_conv_nd(
            vs / "conv2",
            3,
            out_channels,
            out_channels,
            3,
            1,
            1,
            causal,
            "zeros",
        );

        let shortcut: Box<dyn ModuleT> = if in_channels != out_channels {
            make_conv_nd(
                vs / "shortcut",
                3,
                in_channels,
                out_channels,
                1,
                1,
                0,
                causal,
                "zeros",
            )
        } else {
            Box::new(Identity)
        };

        Self {
            norm1,
            conv1,
            norm2,
            conv2,
            shortcut,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.norm1.forward_t(x, false).silu();
        let h = self.conv1.forward_t(&h, true);
        let h = self.norm2.forward_t(&h, false).silu();
        let h = self.conv2.forward_t(&h, true);
        self.shortcut.forward_t(x, true) + h
    }
}

impl ModuleT for ResnetBlock3D {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs)
    }
}

#[derive(Debug)]
struct Identity;

impl ModuleT for Identity {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        xs.shallow_clone()
    }
}

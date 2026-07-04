use ltx_types::NormLayerType;
use tch::nn::ModuleT;

use crate::group_norm::GroupNorm;
use crate::pixel_norm::PixelNorm;

pub fn build_norm_layer(
    norm_type: NormLayerType,
    channels: i64,
    num_groups: i64,
) -> Box<dyn ModuleT + Send> {
    match norm_type {
        NormLayerType::Group => Box::new(GroupNorm::with_defaults(num_groups, channels)),
        NormLayerType::Pixel => Box::new(PixelNorm::default()),
    }
}

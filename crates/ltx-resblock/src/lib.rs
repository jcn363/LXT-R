//! Residual blocks for the LTX-2.3 Rust rewrite.
//!
//! Provides ResnetBlock1D, ResnetBlock2D, ResnetBlock3D, UNetMidBlock3D,
//! and a factory function for creating residual blocks with various configurations.

pub mod factory;
pub mod resblock_1d;
pub mod resblock_2d;
pub mod resblock_3d;
pub mod unet_mid;

pub use factory::make_resblock;
pub use resblock_1d::ResBlock1;
pub use resblock_2d::ResnetBlock2D;
pub use resblock_3d::ResnetBlock3D;
pub use unet_mid::UNetMidBlock3D;

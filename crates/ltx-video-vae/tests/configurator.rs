use ltx_video_vae::configurator::{
    default_encoder_block_descs, CONV_IN_CHANNELS, ENCODER_CONV_OUT_CHANNELS,
    SAMPLED_LATENT_CHANNELS, SPACE_TO_DEPTH_R,
};
use ltx_video_vae::EncoderBlockKind;

#[test]
fn test_encoder_block_descs_count() {
    let descs = default_encoder_block_descs();
    assert_eq!(descs.len(), 10, "Python model has 10 encoder down_blocks");
}

#[test]
fn test_encoder_block_descs_channels() {
    let descs = default_encoder_block_descs();
    // Channel progression: 128->128->256->256->512->512->512->512
    assert_eq!(descs[0].out_ch, 128); // ResBlocks x4
    assert_eq!(descs[1].out_ch, 128); // DownsampleConv
    assert_eq!(descs[2].out_ch, 256); // ChannelChange
    assert_eq!(descs[3].out_ch, 256); // ResBlocks x3
    assert_eq!(descs[4].out_ch, 256); // DownsampleConv
    assert_eq!(descs[5].out_ch, 512); // ChannelChange
    assert_eq!(descs[6].out_ch, 512); // ResBlocks x3
    assert_eq!(descs[7].out_ch, 512); // DownsampleConv
    assert_eq!(descs[8].out_ch, 512); // ResBlocks x3
    assert_eq!(descs[9].out_ch, 512); // ResBlocks x4
}

#[test]
fn test_encoder_block_descs_types() {
    let descs = default_encoder_block_descs();
    assert!(matches!(descs[0].kind, EncoderBlockKind::ResBlocks(4)));
    assert!(matches!(descs[1].kind, EncoderBlockKind::DownsampleConv));
    assert!(matches!(
        descs[2].kind,
        EncoderBlockKind::ChannelChangeDownsample
    ));
    assert!(matches!(descs[3].kind, EncoderBlockKind::ResBlocks(3)));
    assert!(matches!(descs[4].kind, EncoderBlockKind::DownsampleConv));
    assert!(matches!(
        descs[5].kind,
        EncoderBlockKind::ChannelChangeDownsample
    ));
    assert!(matches!(descs[6].kind, EncoderBlockKind::ResBlocks(3)));
    assert!(matches!(descs[7].kind, EncoderBlockKind::DownsampleConv));
    assert!(matches!(descs[8].kind, EncoderBlockKind::ResBlocks(3)));
    assert!(matches!(descs[9].kind, EncoderBlockKind::ResBlocks(4)));
}

#[test]
fn test_channel_constants() {
    assert_eq!(SPACE_TO_DEPTH_R, 4);
    assert_eq!(CONV_IN_CHANNELS, 48); // 3 * 4 * 4
    assert_eq!(ENCODER_CONV_OUT_CHANNELS, 129);
    assert_eq!(SAMPLED_LATENT_CHANNELS, 128);
}

#[test]
fn test_encoder_architecture_totals() {
    let descs = default_encoder_block_descs();
    // Count total resblocks across all stages
    let total_resblocks: i64 = descs
        .iter()
        .map(|d| match d.kind {
            EncoderBlockKind::ResBlocks(n) => n,
            _ => 0,
        })
        .sum();
    // 4 + 3 + 3 + 3 + 4 = 17 resblocks across 5 ResBlock stages
    assert_eq!(total_resblocks, 17);

    // Count stride-2 downsamples: 3 DownsampleConv + 2 ChannelChangeDownsample = 5
    let downsamples = descs
        .iter()
        .filter(|d| !matches!(d.kind, EncoderBlockKind::ResBlocks(_)))
        .count();
    assert_eq!(downsamples, 5, "5 spatial downsampling stages = 32x total");
}

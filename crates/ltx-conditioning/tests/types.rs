use ltx_conditioning::{Keyframe, LatentCond, ReferenceVideo};
use tch::{Device, Kind, Tensor};

#[test]
fn test_latent_cond_new_with_mask() {
    let latent = Tensor::randn([1, 4, 8, 16, 16], (Kind::Float, Device::Cpu));
    let mask = Tensor::ones([1, 1, 8, 16, 16], (Kind::Bool, Device::Cpu));
    let cond = LatentCond::new(latent, Some(mask));
    assert_eq!(cond.latent.size(), vec![1, 4, 8, 16, 16]);
    assert!(cond.mask.is_some());
}

#[test]
fn test_latent_cond_new_no_mask() {
    let latent = Tensor::randn([1, 4, 2, 8, 8], (Kind::Float, Device::Cpu));
    let cond = LatentCond::new(latent, None);
    assert!(cond.mask.is_none());
}

#[test]
fn test_reference_video_slice_frames() {
    let frames = Tensor::randn([1, 3, 10, 8, 8], (Kind::Float, Device::Cpu));
    let rv = ReferenceVideo::new(frames, 2, 7);
    let slice = rv.slice_frames();
    assert_eq!(slice.size(), vec![1, 3, 5, 8, 8]);
}

#[test]
fn test_reference_video_full_range() {
    let frames = Tensor::randn([1, 3, 10, 8, 8], (Kind::Float, Device::Cpu));
    let rv = ReferenceVideo::new(frames, 0, 10);
    assert_eq!(rv.num_frames(), 10);
    let slice = rv.slice_frames();
    assert_eq!(slice.size()[2], 10);
}

#[test]
fn test_keyframe_soft() {
    let latent = Tensor::zeros([1, 4, 1, 8, 8], (Kind::Float, Device::Cpu));
    let kf = Keyframe::soft(3, latent, 0.5);
    assert_eq!(kf.frame_index, 3);
    assert!((kf.strength - 0.5).abs() < 1e-6);
}

#[test]
fn test_keyframe_new() {
    let latent = Tensor::zeros([1, 4, 1, 8, 8], (Kind::Float, Device::Cpu));
    let kf = Keyframe::new(5, latent, 0.75);
    assert_eq!(kf.frame_index, 5);
    assert!((kf.strength - 0.75).abs() < 1e-6);
}

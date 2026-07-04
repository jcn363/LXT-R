use tch::Tensor;

use crate::shapes::AudioLatentShape;

pub struct VideoLatentTools<'a> {
    tensor: &'a Tensor,
}

impl<'a> VideoLatentTools<'a> {
    pub fn new(tensor: &'a Tensor) -> Self {
        Self { tensor }
    }

    pub fn to_pixel_coords(
        &self,
        time_scale: i64,
        height_scale: i64,
        width_scale: i64,
    ) -> (i64, i64, i64) {
        let sizes = self.tensor.size();
        let frames = sizes[2] * time_scale;
        let height = sizes[3] * height_scale;
        let width = sizes[4] * width_scale;
        (frames, height, width)
    }

    pub fn per_channel_statistics(&self) -> (Tensor, Tensor) {
        let dims: &[i64] = &[0, 2, 3, 4];
        let means = self.tensor.mean_dim(dims, true, tch::Kind::Float);
        let stds = self.tensor.std_dim(dims, true, true);
        (means, stds)
    }
}

pub struct AudioLatentTools<'a> {
    tensor: &'a Tensor,
}

impl<'a> AudioLatentTools<'a> {
    pub fn new(tensor: &'a Tensor) -> Self {
        Self { tensor }
    }

    pub fn to_audio_shape(&self) -> AudioLatentShape {
        let sizes = self.tensor.size();
        AudioLatentShape::new(sizes[0], sizes[1], sizes[2], sizes[3])
    }
}

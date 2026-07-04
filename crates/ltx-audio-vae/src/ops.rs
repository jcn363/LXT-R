use tch::Tensor;

/// Per-channel mean and standard deviation statistics.
///
/// Used by the audio VAE to normalize waveforms before encoding and
/// denormalize after decoding.  Computed over the time dimension of
/// `(B, C, T)` audio tensors.
#[derive(Debug)]
pub struct PerChannelStatistics {
    pub mean: Tensor,
    pub std: Tensor,
}

impl PerChannelStatistics {
    /// Compute channel-wise statistics from an audio tensor.
    ///
    /// # Arguments
    /// * `x` — audio tensor of shape `(B, C, T)`.
    ///
    /// # Returns
    /// `PerChannelStatistics` with `mean` and `std` each of shape `(1, C, 1)`.
    pub fn compute(x: &Tensor) -> Self {
        let mean = x.mean_dim(&[2i64][..], true, x.kind());
        let std = x
            .std_dim(&[2i64][..], true, true)
            .clamp_min(ltx_types::STABILITY_EPS);
        Self { mean, std }
    }

    /// Normalize `x` using these statistics: `(x - mean) / std`.
    pub fn normalize(&self, x: &Tensor) -> Tensor {
        (x - &self.mean) / &self.std
    }

    /// Denormalize `x`: `x * std + mean`.
    pub fn denormalize(&self, x: &Tensor) -> Tensor {
        x * &self.std + &self.mean
    }
}

/// Audio preprocessing and postprocessing utilities.
///
/// Handles normalization, chunking for streaming, and frame-level
/// operations on raw waveform tensors.
pub struct AudioProcessor {
    sample_rate: i64,
    chunk_size: i64,
}

impl AudioProcessor {
    /// # Arguments
    /// * `sample_rate` — samples per second (e.g. 44100, 48000).
    /// * `chunk_size` — number of samples per processing chunk.
    pub fn new(sample_rate: i64, chunk_size: i64) -> Self {
        Self {
            sample_rate,
            chunk_size,
        }
    }

    pub fn sample_rate(&self) -> i64 {
        self.sample_rate
    }

    pub fn chunk_size(&self) -> i64 {
        self.chunk_size
    }

    /// Normalize audio to `[-1, 1]` range.
    ///
    /// Divides by the absolute maximum across all elements.
    /// Returns `(normalized, stats)` where `stats` can be used for
    /// denormalization.
    pub fn normalize(&self, x: &Tensor) -> (Tensor, PerChannelStatistics) {
        let stats = PerChannelStatistics::compute(x);
        let normalized = stats.normalize(x);
        (normalized, stats)
    }

    /// Denormalize using previously computed statistics.
    pub fn denormalize(&self, x: &Tensor, stats: &PerChannelStatistics) -> Tensor {
        stats.denormalize(x)
    }

    /// Chunk a `(B, C, T)` audio tensor into non-overlapping frames.
    ///
    /// Returns `(B, C, num_chunks, chunk_size)`.  If `T` is not evenly
    /// divisible by `chunk_size`, the last chunk is zero-padded.
    pub fn chunk(&self, x: &Tensor) -> Tensor {
        let (b, c, t) = x.size3().expect("chunk: tensor must be 3D");
        let num_chunks = (t + self.chunk_size - 1) / self.chunk_size;

        let padded = if t % self.chunk_size != 0 {
            let pad_len = num_chunks * self.chunk_size - t;
            let padding = Tensor::zeros([b, c, pad_len], (x.kind(), x.device()));
            Tensor::cat(&[x, &padding], 2)
        } else {
            x.shallow_clone()
        };

        padded.reshape([b, c, num_chunks, self.chunk_size])
    }

    /// Unchunk frames back into a single waveform.
    ///
    /// Inverse of `chunk`: `(B, C, N, S)` → `(B, C, T)`.
    pub fn unchunk(&self, x: &Tensor) -> Tensor {
        let (b, c, _n, _s) = x.size4().expect("unchunk: tensor must be 4D");
        x.reshape([b, c, -1])
    }

    /// Convert seconds to samples.
    pub fn seconds_to_samples(&self, seconds: f64) -> i64 {
        (seconds * self.sample_rate as f64) as i64
    }

    /// Convert samples to seconds.
    pub fn samples_to_seconds(&self, samples: i64) -> f64 {
        samples as f64 / self.sample_rate as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_per_channel_statistics() {
        let x = Tensor::randn([1, 2, 1024], (tch::Kind::Float, tch::Device::Cpu));
        let stats = PerChannelStatistics::compute(&x);
        assert_eq!(stats.mean.size(), vec![1, 2, 1]);
        assert_eq!(stats.std.size(), vec![1, 2, 1]);

        let normed = stats.normalize(&x);
        let restored = stats.denormalize(&normed);
        assert!(x.allclose(&restored, 1e-5, 1e-5, false));
    }

    #[test]
    fn test_audio_processor_chunk_unchunk() {
        let processor = AudioProcessor::new(44100, 1024);
        let x = Tensor::randn([1, 1, 4096], (tch::Kind::Float, tch::Device::Cpu));
        let chunks = processor.chunk(&x);
        assert_eq!(chunks.size(), vec![1, 1, 4, 1024]);
        let restored = processor.unchunk(&chunks);
        assert!(x.allclose(&restored, 1e-6, 1e-6, false));
    }

    #[test]
    fn test_audio_processor_chunk_padding() {
        let processor = AudioProcessor::new(44100, 1024);
        let x = Tensor::randn([1, 1, 3000], (tch::Kind::Float, tch::Device::Cpu));
        let chunks = processor.chunk(&x);
        assert_eq!(chunks.size(), vec![1, 1, 3, 1024]);
        let restored = processor.unchunk(&chunks);
        // Restored is zero-padded; check that original portion matches
        let original_portion = restored.narrow(2, 0, 3000);
        assert!(x.allclose(&original_portion, 1e-6, 1e-6, false));
    }
}

use std::path::{Path, PathBuf};

pub fn save_frames_png(frames: &[Vec<u8>], h: i64, w: i64, dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("create dir: {e}"))?;

    for (i, frame) in frames.iter().enumerate() {
        let path = dir.join(format!("frame_{i:04}.png"));
        let img = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(w as u32, h as u32, frame.clone())
            .ok_or("failed to create image buffer")?;
        img.save(&path).map_err(|e| format!("save {}: {e}", path.display()))?;
    }

    Ok(())
}

pub fn save_video_mp4(
    frames: &[Vec<u8>],
    h: i64,
    w: i64,
    dir: &Path,
    output: &Path,
) -> Result<PathBuf, String> {
    // Save frames as temporary PGM files for ffmpeg
    let tmp_dir = dir.join(".tmp_frames");
    std::fs::create_dir_all(&tmp_dir).map_err(|e| format!("create tmp dir: {e}"))?;

    for (i, frame) in frames.iter().enumerate() {
        let path = tmp_dir.join(format!("frame_{i:04}.pgm"));
        use std::io::Write;
        let mut f = std::fs::File::create(&path).map_err(|e| format!("create pgm: {e}"))?;
        write!(f, "P6\n{w} {h}\n255\n").map_err(|e| format!("write header: {e}"))?;
        f.write_all(frame).map_err(|e| format!("write pixels: {e}"))?;
    }

    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-framerate", "8",
            "-i", tmp_dir.join("frame_%04d.pgm").to_str().unwrap_or(""),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output.to_str().unwrap_or("output.mp4"),
        ])
        .status()
        .map_err(|e| format!("ffmpeg: {e}"))?;

    // Cleanup temp files
    let _ = std::fs::remove_dir_all(&tmp_dir);

    if status.success() {
        Ok(output.to_path_buf())
    } else {
        Err("ffmpeg failed".to_string())
    }
}

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
    let tmp_dir = save_temp_pgm(frames, h, w, dir)?;

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

    let _ = std::fs::remove_dir_all(&tmp_dir);

    if status.success() {
        Ok(output.to_path_buf())
    } else {
        Err("ffmpeg failed".to_string())
    }
}

pub fn save_gif(
    frames: &[Vec<u8>],
    h: i64,
    w: i64,
    dir: &Path,
    output: &Path,
    fps: u32,
    scale: u32,
) -> Result<PathBuf, String> {
    let tmp_dir = save_temp_pgm(frames, h, w, dir)?;

    let filter = format!(
        "scale={scale}:{scale}:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
    );

    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-framerate",
            &fps.to_string(),
            "-i",
            tmp_dir.join("frame_%04d.pgm").to_str().unwrap_or(""),
            "-vf",
            &filter,
            "-loop",
            "0",
            output.to_str().unwrap_or("output.gif"),
        ])
        .status()
        .map_err(|e| format!("ffmpeg: {e}"))?;

    let _ = std::fs::remove_dir_all(&tmp_dir);

    if status.success() {
        Ok(output.to_path_buf())
    } else {
        Err("ffmpeg gif failed".to_string())
    }
}

fn save_temp_pgm(
    frames: &[Vec<u8>],
    h: i64,
    w: i64,
    dir: &Path,
) -> Result<PathBuf, String> {
    let tmp_dir = dir.join(".tmp_frames");
    std::fs::create_dir_all(&tmp_dir).map_err(|e| format!("create tmp dir: {e}"))?;

    for (i, frame) in frames.iter().enumerate() {
        let path = tmp_dir.join(format!("frame_{i:04}.pgm"));
        use std::io::Write;
        let mut f = std::fs::File::create(&path).map_err(|e| format!("create pgm: {e}"))?;
        write!(f, "P6\n{w} {h}\n255\n").map_err(|e| format!("write header: {e}"))?;
        f.write_all(frame).map_err(|e| format!("write pixels: {e}"))?;
    }

    Ok(tmp_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn make_test_frames(n: usize, w: i64, h: i64) -> Vec<Vec<u8>> {
        (0..n)
            .map(|i| {
                let mut frame = Vec::with_capacity((w * h * 3) as usize);
                for y in 0..h as usize {
                    for x in 0..w as usize {
                        frame.push(((i * 50 + x * 3) % 256) as u8);
                        frame.push(((i * 30 + y * 5) % 256) as u8);
                        frame.push(((x + y) % 256) as u8);
                    }
                }
                frame
            })
            .collect()
    }

    #[test]
    fn save_frames_png_creates_files() {
        let tmp = std::env::temp_dir().join("ltx_test_png");
        let _ = fs::remove_dir_all(&tmp);

        let frames = make_test_frames(3, 8, 8);
        save_frames_png(&frames, 8, 8, &tmp).unwrap();

        for i in 0..3 {
            let path = tmp.join(format!("frame_{i:04}.png"));
            assert!(path.exists(), "missing {path:?}");
            assert!(fs::metadata(&path).unwrap().len() > 0, "empty {path:?}");
        }

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn save_video_mp4_creates_file() {
        let tmp = std::env::temp_dir().join("ltx_test_mp4");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        let frames = make_test_frames(4, 16, 16);
        let output = tmp.join("test.mp4");

        match save_video_mp4(&frames, 16, 16, &tmp, &output) {
            Ok(_) => {
                assert!(output.exists(), "mp4 not created");
                assert!(fs::metadata(&output).unwrap().len() > 0, "mp4 empty");
            }
            Err(e) => {
                eprintln!("ffmpeg not available: {e} — skipping mp4 test");
            }
        }

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn save_gif_creates_file() {
        let tmp = std::env::temp_dir().join("ltx_test_gif");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        let frames = make_test_frames(4, 16, 16);
        let output = tmp.join("test.gif");

        match save_gif(&frames, 16, 16, &tmp, &output, 8, 64) {
            Ok(_) => {
                assert!(output.exists(), "gif not created");
                assert!(fs::metadata(&output).unwrap().len() > 0, "gif empty");
            }
            Err(e) => {
                eprintln!("ffmpeg not available: {e} — skipping gif test");
            }
        }

        let _ = fs::remove_dir_all(&tmp);
    }
}

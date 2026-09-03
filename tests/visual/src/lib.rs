//! Visual compatibility suite (spec §3, §40): the branding overlay layer
//! must leave every original pixel untouched outside the explicit mask.
//! Screenshot fixtures are PNGs; the diff is zero-pixel outside masked
//! branding regions.

use image::{GenericImageView, Rgba, RgbaImage};

pub mod runner;

/// The masked branding region where changed pixels are permitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrandingMask {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl BrandingMask {
    pub fn contains(&self, x: u32, y: u32) -> bool {
        x >= self.x && y >= self.y && x < self.x + self.width && y < self.y + self.height
    }
}

/// A pixel-level diff that permits changes only inside the mask. Outside the
/// mask the requirement is zero changed pixels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PixelDiff {
    pub changed_outside_mask: Vec<(u32, u32)>,
    pub changed_inside_mask: Vec<(u32, u32)>,
    pub total_changed: usize,
}

impl PixelDiff {
    pub fn zero_pixel_difference_outside_mask(&self) -> bool {
        self.changed_outside_mask.is_empty()
    }
}

/// Compare two images pixel by pixel; report every changed pixel, classified
/// by whether it falls inside the branding mask. Images must be the same
/// dimensions (a size mismatch is a hard failure, not a diff).
pub fn diff_images(
    before: &RgbaImage,
    after: &RgbaImage,
    mask: Option<BrandingMask>,
) -> Result<PixelDiff, String> {
    if before.dimensions() != after.dimensions() {
        return Err(format!(
            "image size mismatch: {:?} vs {:?}",
            before.dimensions(),
            after.dimensions()
        ));
    }
    let (w, h) = before.dimensions();
    let mut outside = Vec::new();
    let mut inside = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if before.get_pixel(x, y) != after.get_pixel(x, y) {
                let in_mask = mask.map(|m| m.contains(x, y)).unwrap_or(false);
                if in_mask {
                    inside.push((x, y));
                } else {
                    outside.push((x, y));
                }
            }
        }
    }
    let total = outside.len() + inside.len();
    Ok(PixelDiff {
        changed_outside_mask: outside,
        changed_inside_mask: inside,
        total_changed: total,
    })
}

/// The Faktor branding overlay: a transparent `+` positioned independently so
/// adjacent UI never shifts. Implemented as an overlay layer composited onto
/// the original — the original's pixels outside the mask are untouched by
/// construction.
pub fn compose_plus_overlay(
    original: &RgbaImage,
    mask: BrandingMask,
    plus_color: Rgba<u8>,
) -> RgbaImage {
    let mut out = original.clone();
    let cx = mask.x + mask.width / 2;
    let cy = mask.y + mask.height / 2;
    let arm = (mask.width.min(mask.height) / 3).max(1);
    // Vertical arm.
    for dy in 0..mask.height {
        let y = mask.y + dy;
        let x = cx;
        if mask.contains(x, y) && out.in_bounds(x, y) {
            out.put_pixel(x, y, plus_color);
        }
    }
    // Horizontal arm.
    for dx in 0..mask.width {
        let x = mask.x + dx;
        let y = cy;
        if mask.contains(x, y) && out.in_bounds(x, y) {
            out.put_pixel(x, y, plus_color);
        }
    }
    let _ = arm;
    out
}

/// Generate a deterministic fixture screenshot (used by the fixtures dir and
/// by tests that need a stable before-image).
pub fn fixture_screenshot(width: u32, height: u32, seed: u8) -> RgbaImage {
    let mut img = RgbaImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let v = (x as u8)
                .wrapping_mul(31)
                .wrapping_add(y as u8)
                .wrapping_add(seed);
            img.put_pixel(x, y, Rgba([v, v.wrapping_mul(2), 255 - v, 255]));
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask() -> BrandingMask {
        BrandingMask {
            x: 10,
            y: 10,
            width: 12,
            height: 12,
        }
    }

    #[test]
    fn identical_images_zero_diff() {
        let img = fixture_screenshot(64, 64, 7);
        let diff = diff_images(&img, &img, Some(mask())).unwrap();
        assert_eq!(diff.total_changed, 0);
        assert!(diff.zero_pixel_difference_outside_mask());
    }

    #[test]
    fn mask_contains_checks_bounds() {
        let m = mask();
        assert!(m.contains(10, 10));
        assert!(m.contains(21, 21));
        assert!(!m.contains(9, 9));
        assert!(!m.contains(22, 10));
        assert!(!m.contains(10, 22));
    }

    #[test]
    fn changes_outside_mask_are_reported() {
        let before = fixture_screenshot(64, 64, 1);
        let mut after = before.clone();
        after.put_pixel(0, 0, Rgba([255, 0, 0, 255])); // outside the mask
        after.put_pixel(15, 15, Rgba([0, 255, 0, 255])); // inside the mask
        let diff = diff_images(&before, &after, Some(mask())).unwrap();
        assert_eq!(diff.changed_outside_mask, vec![(0, 0)]);
        assert_eq!(diff.changed_inside_mask, vec![(15, 15)]);
        assert!(!diff.zero_pixel_difference_outside_mask());
    }

    #[test]
    fn changes_only_inside_mask_pass() {
        let before = fixture_screenshot(64, 64, 2);
        let mut after = before.clone();
        for y in 10..22 {
            for x in 10..22 {
                after.put_pixel(x, y, Rgba([9, 9, 9, 255]));
            }
        }
        let diff = diff_images(&before, &after, Some(mask())).unwrap();
        assert!(
            diff.zero_pixel_difference_outside_mask(),
            "all changes inside the branding mask"
        );
        assert_eq!(diff.changed_inside_mask.len(), 12 * 12);
    }

    #[test]
    fn size_mismatch_is_hard_failure() {
        let a = fixture_screenshot(64, 64, 0);
        let b = fixture_screenshot(63, 64, 0);
        assert!(diff_images(&a, &b, Some(mask())).is_err());
    }

    #[test]
    fn plus_overlay_touches_only_mask_pixels() {
        let original = fixture_screenshot(64, 64, 3);
        let m = mask();
        let overlaid = compose_plus_overlay(&original, m, Rgba([255, 255, 255, 255]));
        let diff = diff_images(&original, &overlaid, Some(m)).unwrap();
        assert!(
            diff.zero_pixel_difference_outside_mask(),
            "the + overlay must never touch pixels outside the mask"
        );
        assert!(
            !diff.changed_inside_mask.is_empty(),
            "the + is visible inside the mask"
        );
    }

    #[test]
    fn plus_overlay_is_independent_of_adjacent_layout() {
        // The overlay touches ONLY the mask, regardless of the base image:
        // the changed-pixel set inside the mask is identical for both bases.
        let a = fixture_screenshot(64, 64, 4);
        let b = fixture_screenshot(64, 64, 5);
        let m = mask();
        let oa = compose_plus_overlay(&a, m, Rgba([255, 255, 255, 255]));
        let ob = compose_plus_overlay(&b, m, Rgba([255, 255, 255, 255]));
        // Each overlay leaves its own base untouched outside the mask...
        let da = diff_images(&a, &oa, Some(m)).unwrap();
        assert!(da.zero_pixel_difference_outside_mask());
        let db = diff_images(&b, &ob, Some(m)).unwrap();
        assert!(db.zero_pixel_difference_outside_mask());
        // ...and the overlay draws the same pixels in both cases.
        assert_eq!(da.changed_inside_mask, db.changed_inside_mask);
    }

    #[test]
    fn png_fixture_roundtrip() {
        // Fixtures must roundtrip through PNG encoding losslessly.
        let img = fixture_screenshot(32, 32, 9);
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        let decoded = image::load_from_memory(&buf.into_inner())
            .unwrap()
            .to_rgba8();
        let diff = diff_images(&img, &decoded, None).unwrap();
        assert_eq!(diff.total_changed, 0, "PNG roundtrip must be lossless");
    }

    #[test]
    fn alpha_channel_changes_are_detected() {
        let before = fixture_screenshot(16, 16, 0);
        let mut after = before.clone();
        // Change only the alpha channel outside the mask.
        after.put_pixel(
            3,
            3,
            Rgba([
                before.get_pixel(3, 3).0[0],
                before.get_pixel(3, 3).0[1],
                before.get_pixel(3, 3).0[2],
                0,
            ]),
        );
        let diff = diff_images(&before, &after, None).unwrap();
        assert_eq!(diff.total_changed, 1);
    }

    #[test]
    fn fixture_screenshots_are_deterministic() {
        assert_eq!(fixture_screenshot(16, 16, 5), fixture_screenshot(16, 16, 5));
    }
}

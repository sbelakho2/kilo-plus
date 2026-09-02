//! Pixel-regression runner over the frozen screenshot corpus
//! (`fixtures/screenshots`). The audit contract: the branded `+` overlay
//! must change pixels ONLY inside its explicit mask; every pixel outside
//! the mask and the OS font-rendering tolerance zone must be byte-identical
//! between baseline and overlay. Inside the mask, changes are required
//! (the overlay must actually be visible).

use std::path::Path;

use image::RgbaImage;

use crate::{diff_images, BrandingMask};

/// A rectangular region in screenshot coordinates (a branding mask or the
/// OS font-rendering tolerance zone).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PixelRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl PixelRegion {
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// One manifest entry: baseline screenshot, branded overlay, the mask where
/// the overlay may change pixels, and the tolerance zone that absorbs
/// OS font-rendering differences (may be zero-sized).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ManifestEntry {
    pub name: String,
    pub baseline: String,
    pub overlay: String,
    pub mask: PixelRegion,
    pub tolerance: PixelRegion,
}

/// Offending-pixel reporting is bounded (bounded-everything commandment).
const MAX_REPORTED_PIXELS: usize = 20;

/// Run the pixel regression over every entry of `fixtures_dir/manifest.json`.
///
/// Returns the number of entries that passed — equal to the manifest length
/// on success. Errors on the first failing entry, naming the entry and the
/// offending pixels (bounded to the first 20).
pub fn run_pixel_regression(fixtures_dir: &Path) -> Result<usize, String> {
    let manifest_path = fixtures_dir.join("manifest.json");
    let raw = std::fs::read(&manifest_path)
        .map_err(|e| format!("cannot read manifest {}: {e}", manifest_path.display()))?;
    let entries: Vec<ManifestEntry> = serde_json::from_slice(&raw).map_err(|e| {
        format!(
            "manifest {} is not valid JSON: {e}",
            manifest_path.display()
        )
    })?;
    for entry in &entries {
        check_entry(fixtures_dir, entry)?;
    }
    Ok(entries.len())
}

fn check_entry(fixtures_dir: &Path, entry: &ManifestEntry) -> Result<(), String> {
    let baseline = load_png(&fixtures_dir.join(&entry.baseline), &entry.name, "baseline")?;
    let overlay = load_png(&fixtures_dir.join(&entry.overlay), &entry.name, "overlay")?;
    if baseline.dimensions() != overlay.dimensions() {
        return Err(format!(
            "{}: baseline/overlay size mismatch: {:?} vs {:?}",
            entry.name,
            baseline.dimensions(),
            overlay.dimensions()
        ));
    }
    let (w, h) = baseline.dimensions();
    let mask = BrandingMask {
        x: entry.mask.x,
        y: entry.mask.y,
        width: entry.mask.width,
        height: entry.mask.height,
    };
    let tolerance = BrandingMask {
        x: entry.tolerance.x,
        y: entry.tolerance.y,
        width: entry.tolerance.width,
        height: entry.tolerance.height,
    };
    check_bounds(&entry.name, "mask", &mask, w, h)?;
    check_bounds(&entry.name, "tolerance", &tolerance, w, h)?;

    let diff =
        diff_images(&baseline, &overlay, Some(mask)).map_err(|e| format!("{}: {e}", entry.name))?;
    // The tolerance zone absorbs outside-mask changes (font rendering).
    let outside: Vec<(u32, u32)> = diff
        .changed_outside_mask
        .iter()
        .copied()
        .filter(|(x, y)| !tolerance.contains(*x, *y))
        .collect();
    if !outside.is_empty() {
        let shown: Vec<(u32, u32)> = outside.iter().take(MAX_REPORTED_PIXELS).copied().collect();
        return Err(format!(
            "{}: {} pixels changed outside mask+tolerance; first {}: {shown:?}",
            entry.name,
            outside.len(),
            MAX_REPORTED_PIXELS
        ));
    }
    // The overlay must be visible inside a non-empty mask. A zero-size mask
    // is accepted: there is nothing to prove.
    if !entry.mask.is_empty() && diff.changed_inside_mask.is_empty() {
        return Err(format!(
            "{}: overlay invisible inside a non-empty mask (zero changed pixels inside)",
            entry.name
        ));
    }
    Ok(())
}

fn load_png(path: &Path, entry: &str, kind: &str) -> Result<RgbaImage, String> {
    image::open(path)
        .map(|img| img.to_rgba8())
        .map_err(|e| format!("{entry}: cannot decode {kind} {}: {e}", path.display()))
}

fn check_bounds(
    entry: &str,
    kind: &str,
    region: &BrandingMask,
    w: u32,
    h: u32,
) -> Result<(), String> {
    // checked_add: hostile manifests must not overflow into a panic.
    let in_bounds = region
        .x
        .checked_add(region.width)
        .is_some_and(|end| end <= w)
        && region
            .y
            .checked_add(region.height)
            .is_some_and(|end| end <= h);
    if !in_bounds {
        return Err(format!(
            "{}: {kind} region ({},{},{},{}) out of bounds for {w}x{h} image",
            entry, region.x, region.y, region.width, region.height
        ));
    }
    Ok(())
}

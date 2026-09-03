// Generates the screenshot fixtures for the visual pixel-regression suite:
// for each audit case a pure baseline, the branded `+` overlay, and the
// manifest entry that locks the mask + OS font-rendering tolerance zone.

use faktor_tests_visual::runner::{ManifestEntry, PixelRegion};
use faktor_tests_visual::{compose_plus_overlay, fixture_screenshot, BrandingMask};
use image::Rgba;

struct Case {
    name: &'static str,
    seed: u8,
    mask: (u32, u32, u32, u32),
    tolerance: (u32, u32, u32, u32),
}

/// The full audit matrix (16 cases): VS Code dark carries every UI state;
/// VS Code light, JetBrains dark/light keep their frozen pairs. Every case
/// gets a distinct seed (distinct screenshot) and a distinct mask position.
const CASES: [Case; 16] = [
    Case {
        name: "vscode-dark-empty-chat",
        seed: 10,
        mask: (760, 8, 24, 24),
        tolerance: (0, 0, 0, 0),
    },
    Case {
        name: "vscode-dark-streaming",
        seed: 11,
        mask: (760, 36, 24, 24),
        tolerance: (0, 576, 800, 24),
    },
    Case {
        name: "vscode-dark-tool-call",
        seed: 12,
        mask: (640, 200, 24, 24),
        tolerance: (0, 0, 0, 0),
    },
    Case {
        name: "vscode-dark-diff",
        seed: 13,
        mask: (400, 420, 24, 24),
        tolerance: (0, 0, 0, 0),
    },
    Case {
        name: "vscode-dark-permission",
        seed: 14,
        mask: (400, 260, 24, 24),
        tolerance: (0, 0, 0, 0),
    },
    Case {
        name: "vscode-dark-settings",
        seed: 15,
        mask: (16, 560, 24, 24),
        tolerance: (0, 0, 0, 0),
    },
    Case {
        name: "vscode-dark-model-selector",
        seed: 16,
        mask: (390, 180, 24, 24),
        tolerance: (0, 0, 0, 0),
    },
    Case {
        name: "vscode-dark-agent-manager",
        seed: 17,
        mask: (760, 300, 24, 24),
        tolerance: (0, 572, 800, 28),
    },
    Case {
        name: "vscode-dark-background-agent",
        seed: 18,
        mask: (560, 520, 24, 24),
        tolerance: (0, 0, 0, 0),
    },
    Case {
        name: "vscode-dark-history",
        seed: 19,
        mask: (700, 540, 24, 24),
        tolerance: (0, 0, 0, 0),
    },
    Case {
        name: "vscode-dark-error",
        seed: 20,
        mask: (400, 300, 24, 24),
        tolerance: (0, 0, 0, 0),
    },
    Case {
        name: "vscode-dark-compaction",
        seed: 21,
        mask: (600, 120, 24, 24),
        tolerance: (0, 0, 800, 2),
    },
    Case {
        name: "vscode-light-empty-chat",
        seed: 22,
        mask: (760, 8, 24, 24),
        tolerance: (0, 0, 0, 0),
    },
    Case {
        name: "vscode-light-settings",
        seed: 23,
        mask: (16, 560, 24, 24),
        tolerance: (0, 0, 0, 0),
    },
    Case {
        name: "jetbrains-dark-empty-chat",
        seed: 24,
        mask: (750, 12, 24, 24),
        tolerance: (0, 0, 0, 0),
    },
    Case {
        name: "jetbrains-light-empty-chat",
        seed: 25,
        mask: (750, 12, 24, 24),
        tolerance: (0, 0, 0, 0),
    },
];

#[test]
#[ignore = "[visual] fixture generation — run explicitly to regenerate"]
fn gen_fixtures() {
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/screenshots");
    std::fs::create_dir_all(&base).unwrap();
    let mut manifest = Vec::new();
    for case in &CASES {
        let baseline = fixture_screenshot(800, 600, case.seed);
        let mask = BrandingMask {
            x: case.mask.0,
            y: case.mask.1,
            width: case.mask.2,
            height: case.mask.3,
        };
        let overlay = compose_plus_overlay(&baseline, mask, Rgba([255, 255, 255, 255]));
        baseline
            .save(base.join(format!("{}.png", case.name)))
            .unwrap();
        overlay
            .save(base.join(format!("{}-overlay.png", case.name)))
            .unwrap();
        manifest.push(ManifestEntry {
            name: case.name.to_string(),
            baseline: format!("{}.png", case.name),
            overlay: format!("{}-overlay.png", case.name),
            mask: PixelRegion {
                x: case.mask.0,
                y: case.mask.1,
                width: case.mask.2,
                height: case.mask.3,
            },
            tolerance: PixelRegion {
                x: case.tolerance.0,
                y: case.tolerance.1,
                width: case.tolerance.2,
                height: case.tolerance.3,
            },
        });
    }
    let json = serde_json::to_string_pretty(&manifest).unwrap();
    std::fs::write(base.join("manifest.json"), json).unwrap();
    println!(
        "wrote {} screenshot fixtures (baseline + overlay + manifest) to {}",
        manifest.len(),
        base.display()
    );
}

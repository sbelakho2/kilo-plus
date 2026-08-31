// Generates the screenshot fixtures for the visual compatibility suite.
#[test]
#[ignore = "[visual] fixture generation — run explicitly to regenerate"]
fn gen_fixtures() {
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/screenshots");
    let _ = std::fs::create_dir_all(&base);
    let cases = [
        ("vscode-dark-empty-chat", 800u32, 600u32, 10u8),
        ("vscode-light-empty-chat", 800u32, 600u32, 20u8),
        ("vscode-dark-streaming", 800u32, 600u32, 30u8),
        ("vscode-dark-tool-call", 800u32, 600u32, 40u8),
        ("vscode-dark-permission", 800u32, 600u32, 50u8),
        ("vscode-light-settings", 800u32, 600u32, 60u8),
        ("vscode-dark-model-selector", 800u32, 600u32, 70u8),
        ("vscode-dark-agent-manager", 800u32, 600u32, 80u8),
        ("jetbrains-dark-empty-chat", 800u32, 600u32, 90u8),
        ("jetbrains-light-empty-chat", 800u32, 600u32, 100u8),
    ];
    let mut n = 0u32;
    for (name, w, h, seed) in cases {
        let img = kilop_tests_visual::fixture_screenshot(w, h, seed);
        let mask = kilop_tests_visual::BrandingMask { x: w - 40, y: 8, width: 24, height: 24 };
        let img = kilop_tests_visual::compose_plus_overlay(&img, mask, image::Rgba([255, 255, 255, 255]));
        let path = base.join(format!("{name}.png"));
        img.save(&path).unwrap();
        n += 1;
    }
    println!("wrote {n} screenshot fixtures to {}", base.display());
}

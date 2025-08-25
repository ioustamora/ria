mod ai;
mod config;
mod ui;
mod utils;

use eframe::egui;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), eframe::Error> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "RIA AI Chat",
        options,
        Box::new(|cc| {
            // Don't install image loaders since we're not using them yet
            // egui_extras::install_image_loaders(&cc.egui_ctx);
            Ok(Box::new(ui::RiaApp::new(cc)))
        }),
    )
}

use crate::config::AppConfig;
use crate::ui::components::SystemStatusComponent;
use eframe::egui;

pub fn render_settings(ui: &mut egui::Ui, config: &mut AppConfig, system_status: &mut SystemStatusComponent) {
    ui.heading("Application Settings");
    ui.separator();
    ui.add_space(10.0);

    // Theme selection
    ui.horizontal(|ui| {
        ui.label("Theme:");
        egui::ComboBox::from_label("")
            .selected_text(format!("{:?}", config.theme))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut config.theme, crate::ui::app::Theme::Dark, "Dark");
                ui.selectable_value(&mut config.theme, crate::ui::app::Theme::Light, "Light");
                ui.selectable_value(&mut config.theme, crate::ui::app::Theme::System, "System");
            });
    });

    ui.add_space(10.0);

    // AI Settings
    ui.heading("AI Settings");
    ui.separator();
    ui.add_space(10.0);

    ui.horizontal(|ui| {
        ui.label("Model Path:");
        ui.text_edit_singleline(&mut config.ai_config.model_path);
        if ui.button("Browse").clicked() {
            // Would open file dialog in a real implementation
        }
    });

    ui.add_space(10.0);

    ui.horizontal(|ui| {
        ui.label("Max Tokens:");
        ui.add(egui::Slider::new(&mut config.ai_config.max_tokens, 1..=4096));
    });

    ui.horizontal(|ui| {
        ui.label("Temperature:");
        ui.add(egui::Slider::new(&mut config.ai_config.temperature, 0.0..=2.0).step_by(0.1));
    });

    ui.horizontal(|ui| {
        ui.label("Top-p:");
        ui.add(egui::Slider::new(&mut config.ai_config.top_p, 0.0..=1.0).step_by(0.05));
    });

    ui.add_space(10.0);

    // Execution Provider
    ui.horizontal(|ui| {
        ui.label("Execution Provider:");
        egui::ComboBox::from_label("")
            .selected_text(format!("{:?}", config.ai_config.execution_provider))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut config.ai_config.execution_provider, crate::ai::ExecutionProvider::Cpu, "CPU");
                ui.selectable_value(&mut config.ai_config.execution_provider, crate::ai::ExecutionProvider::Cuda, "CUDA");
                ui.selectable_value(&mut config.ai_config.execution_provider, crate::ai::ExecutionProvider::DirectML, "DirectML");
                ui.selectable_value(&mut config.ai_config.execution_provider, crate::ai::ExecutionProvider::CoreML, "CoreML");
                ui.selectable_value(&mut config.ai_config.execution_provider, crate::ai::ExecutionProvider::OpenVINO, "OpenVINO");
                ui.selectable_value(&mut config.ai_config.execution_provider, crate::ai::ExecutionProvider::QNN, "QNN (NPU)");
            });
    });

    ui.add_space(10.0);

    ui.checkbox(&mut config.ai_config.use_gpu, "Use GPU acceleration");
    ui.checkbox(&mut config.ai_config.use_npu, "Use NPU acceleration");

    ui.add_space(6.0);
    ui.checkbox(&mut config.ai_config.prefer_npu, "Prefer Intel NPU (OpenVINO) if available");

    ui.add_space(20.0);

    // Performance Settings
    ui.heading("Performance");
    ui.separator();
    ui.add_space(10.0);

    ui.horizontal(|ui| {
        ui.label("Animation Quality:");
        egui::ComboBox::from_label("")
            .selected_text(match config.animation_quality {
                0 => "Low",
                1 => "Medium", 
                _ => "High",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut config.animation_quality, 0, "Low");
                ui.selectable_value(&mut config.animation_quality, 1, "Medium");
                ui.selectable_value(&mut config.animation_quality, 2, "High");
            });
    });

    ui.checkbox(&mut config.enable_animations, "Enable animations");
    ui.checkbox(&mut config.enable_sound, "Enable sound effects");

    ui.add_space(20.0);

    // System Status and Memory Monitoring
    system_status.render(ui);

    ui.add_space(20.0);

    ui.heading("Automation");
    ui.separator();
    ui.add_space(10.0);
    ui.checkbox(&mut config.auto_load_last_model, "Auto-load last used model on startup");
    ui.checkbox(&mut config.auto_select_latest_model, "If none, auto-select most recent model");
    ui.checkbox(&mut config.auto_load_new_download, "Auto-load model immediately after download");
    ui.checkbox(&mut config.auto_fix_onnx_runtime, "Attempt ONNX Runtime auto-fix on version mismatch");
    ui.checkbox(&mut config.enable_ep_fallback, "Enable execution provider fallback attempts");

    ui.add_space(20.0);

    if ui.button("Save Settings").clicked() {
        if let Err(e) = config.save() {
            tracing::error!("Failed to save settings: {}", e);
        }
    }
}
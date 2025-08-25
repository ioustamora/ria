use eframe::egui;
use crate::utils::system::SystemInfo;
use std::time::{Instant, Duration};

// Download state tracking
#[derive(Debug, Clone)]
pub struct DownloadInfo {
    pub name: String,
    pub progress: f32,
    pub total_bytes: u64,
    pub downloaded_bytes: u64,
    pub speed_bps: f64,
    pub eta_seconds: f64,
    pub status: DownloadStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DownloadStatus {
    Starting,
    Downloading,
    Paused,
    Completed,
    Failed(String),
    Cancelled,
}

// Enhanced download progress component
pub struct DownloadProgressCard {
    pub info: DownloadInfo,
    progress_bar: ProgressBar,
    last_update: Instant,
}

impl DownloadProgressCard {
    pub fn new(info: DownloadInfo) -> Self {
        Self {
            progress_bar: ProgressBar::new(info.progress)
                .with_color_scheme(ProgressColorScheme::Download)
                .with_label(info.name.clone()),
            info,
            last_update: Instant::now(),
        }
    }
    
    pub fn update(&mut self, info: DownloadInfo) {
        self.info = info;
        self.progress_bar.set_progress(self.info.progress);
        self.last_update = Instant::now();
    }
    
    pub fn show(&mut self, ui: &mut egui::Ui) {
        egui::Frame::none()
            .fill(egui::Color32::from_rgb(25, 35, 45))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(60, 70, 80)))
            .rounding(8.0)
            .inner_margin(12.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    // Status icon
                    let (icon, color) = match &self.info.status {
                        DownloadStatus::Starting => ("⏳", egui::Color32::YELLOW),
                        DownloadStatus::Downloading => ("📥", egui::Color32::from_rgb(70, 130, 220)),
                        DownloadStatus::Paused => ("⏸️", egui::Color32::GRAY),
                        DownloadStatus::Completed => ("✅", egui::Color32::GREEN),
                        DownloadStatus::Failed(_) => ("❌", egui::Color32::RED),
                        DownloadStatus::Cancelled => ("🚫", egui::Color32::GRAY),
                    };
                    
                    ui.colored_label(color, icon);
                    ui.vertical(|ui| {
                        ui.strong(&self.info.name);
                        
                        // Progress bar
                        self.progress_bar.show(ui, [300.0, 20.0]);
                        
                        // Status details
                        ui.horizontal(|ui| {
                            if self.info.total_bytes > 0 {
                                ui.label(format!("{} / {}", 
                                    format_bytes(self.info.downloaded_bytes),
                                    format_bytes(self.info.total_bytes)
                                ));
                            }
                            
                            if self.info.speed_bps > 0.0 {
                                ui.label(format!("{}/s", format_bytes(self.info.speed_bps as u64)));
                            }
                            
                            if self.info.eta_seconds > 0.0 && self.info.eta_seconds < 3600.0 {
                                ui.label(format!("ETA: {}s", self.info.eta_seconds as u32));
                            }
                            
                            // Add cancel button for active downloads
                            if matches!(self.info.status, DownloadStatus::Downloading | DownloadStatus::Starting) {
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if ui.small_button("❌")
                                        .on_hover_text("Cancel download")
                                        .clicked() {
                                        // TODO: Implement cancellation logic
                                        // For now, just mark as cancelled in UI
                                    }
                                });
                            }
                        });
                        
                        // Error message for failed downloads
                        if let DownloadStatus::Failed(error) = &self.info.status {
                            ui.colored_label(egui::Color32::RED, format!("Error: {}", error));
                        }
                    });
                });
            });
    }
}

// System loading component
pub struct SystemLoadingIndicator {
    spinner: LoadingSpinner,
    stage: String,
    progress: f32,
}

impl SystemLoadingIndicator {
    pub fn new() -> Self {
        Self {
            spinner: LoadingSpinner::default(),
            stage: "Initializing...".to_string(),
            progress: 0.0,
        }
    }
    
    pub fn set_stage(&mut self, stage: String, progress: f32) {
        self.stage = stage;
        self.progress = progress.clamp(0.0, 1.0);
    }
    
    pub fn show(&mut self, ui: &mut egui::Ui) {
        ui.vertical_centered(|ui| {
            self.spinner.show(ui, 20.0);
            ui.add_space(10.0);
            ui.label(&self.stage);
            
            let mut progress_bar = ProgressBar::new(self.progress)
                .with_color_scheme(ProgressColorScheme::Processing)
                .without_percentage();
            progress_bar.show(ui, [200.0, 8.0]);
        });
    }
}

// Utility function for formatting bytes
fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    if unit_index == 0 {
        format!("{} {}", size as u64, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

pub struct AnimatedButton {
    hover_progress: f32,
    click_progress: f32,
}

impl Default for AnimatedButton {
    fn default() -> Self {
        Self {
            hover_progress: 0.0,
            click_progress: 0.0,
        }
    }
}

impl AnimatedButton {
    pub fn show(&mut self, ui: &mut egui::Ui, text: &str, size: [f32; 2]) -> egui::Response {
        let (rect, response) = ui.allocate_exact_size(size.into(), egui::Sense::click());
        
        // Update animations
        let dt = ui.input(|i| i.stable_dt);
        
        if response.hovered() {
            self.hover_progress = (self.hover_progress + dt * 8.0).min(1.0);
        } else {
            self.hover_progress = (self.hover_progress - dt * 8.0).max(0.0);
        }
        
        if response.clicked() {
            self.click_progress = 1.0;
        }
        self.click_progress = (self.click_progress - dt * 12.0).max(0.0);

        // Colors based on animation state
        let base_color = egui::Color32::from_rgb(70, 130, 180);
        let hover_color = egui::Color32::from_rgb(90, 150, 200);
        let click_color = egui::Color32::from_rgb(50, 110, 160);
        
        let current_color = if self.click_progress > 0.0 {
            lerp_color32(base_color, click_color, self.click_progress)
        } else {
            lerp_color32(base_color, hover_color, self.hover_progress)
        };

        // Draw button with rounded corners and shadow
        let shadow_rect = rect.translate([2.0, 2.0].into());
        ui.painter().rect_filled(
            shadow_rect,
            8.0,
            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 30),
        );

        ui.painter().rect_filled(rect, 8.0, current_color);

        // Text
        let text_color = egui::Color32::WHITE;
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            text,
            egui::FontId::proportional(14.0),
            text_color,
        );

        response
    }
}

pub struct LoadingSpinner {
    rotation: f32,
}

impl Default for LoadingSpinner {
    fn default() -> Self {
        Self { rotation: 0.0 }
    }
}

impl LoadingSpinner {
    pub fn show(&mut self, ui: &mut egui::Ui, radius: f32) {
        let dt = ui.input(|i| i.stable_dt);
        self.rotation += dt * 4.0; // 4 radians per second

        let (rect, _) = ui.allocate_exact_size([radius * 2.0, radius * 2.0].into(), egui::Sense::hover());
        let center = rect.center();

        // Draw spinning arc
        let _stroke = egui::Stroke::new(3.0, egui::Color32::from_rgb(100, 200, 255));
        
        for i in 0..8 {
            let angle = self.rotation + i as f32 * std::f32::consts::PI / 4.0;
            let alpha = ((i as f32 / 8.0) * 255.0) as u8;
            let color = egui::Color32::from_rgba_unmultiplied(100, 200, 255, alpha);
            
            let start = center + [angle.cos() * radius * 0.6, angle.sin() * radius * 0.6].into();
            let end = center + [angle.cos() * radius * 0.9, angle.sin() * radius * 0.9].into();
            
            ui.painter().line_segment([start, end], egui::Stroke::new(2.0, color));
        }
    }
}

pub struct ProgressBar {
    progress: f32,
    animated_progress: f32,
    label: Option<String>,
    show_percentage: bool,
    color_scheme: ProgressColorScheme,
}

pub enum ProgressColorScheme {
    Default,
    Download,
    Processing,
    Success,
    Warning,
}

impl ProgressBar {
    pub fn new(progress: f32) -> Self {
        Self {
            progress: progress.clamp(0.0, 1.0),
            animated_progress: 0.0,
            label: None,
            show_percentage: true,
            color_scheme: ProgressColorScheme::Default,
        }
    }
    
    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }
    
    pub fn with_color_scheme(mut self, scheme: ProgressColorScheme) -> Self {
        self.color_scheme = scheme;
        self
    }
    
    pub fn without_percentage(mut self) -> Self {
        self.show_percentage = false;
        self
    }
    
    pub fn set_progress(&mut self, progress: f32) {
        self.progress = progress.clamp(0.0, 1.0);
    }

    pub fn show(&mut self, ui: &mut egui::Ui, size: [f32; 2]) {
        let dt = ui.input(|i| i.stable_dt);
        
        // Smooth animation towards target progress
        let diff = self.progress - self.animated_progress;
        self.animated_progress += diff * dt * 5.0;

        let (rect, _) = ui.allocate_exact_size(size.into(), egui::Sense::hover());
        
        let (bg_color, fill_color) = match self.color_scheme {
            ProgressColorScheme::Default => (egui::Color32::from_rgb(40, 40, 50), egui::Color32::from_rgb(100, 200, 100)),
            ProgressColorScheme::Download => (egui::Color32::from_rgb(30, 40, 60), egui::Color32::from_rgb(70, 130, 220)),
            ProgressColorScheme::Processing => (egui::Color32::from_rgb(50, 40, 30), egui::Color32::from_rgb(255, 165, 0)),
            ProgressColorScheme::Success => (egui::Color32::from_rgb(30, 50, 30), egui::Color32::from_rgb(50, 200, 50)),
            ProgressColorScheme::Warning => (egui::Color32::from_rgb(60, 50, 30), egui::Color32::from_rgb(255, 200, 0)),
        };
        
        // Background with subtle border
        ui.painter().rect_filled(rect, 6.0, bg_color);
        ui.painter().rect_stroke(rect, 6.0, egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 80, 90)));

        // Progress fill with gradient effect
        let fill_width = rect.width() * self.animated_progress;
        if fill_width > 0.0 {
            let fill_rect = egui::Rect::from_min_size(rect.min, [fill_width, rect.height()].into());
            ui.painter().rect_filled(fill_rect, 6.0, fill_color);
            
            // Add shine effect
            let shine_rect = egui::Rect::from_min_size(
                rect.min + egui::Vec2::new(0.0, 1.0), 
                [fill_width, rect.height() / 3.0].into()
            );
            let shine_color = egui::Color32::from_rgba_unmultiplied(
                255, 255, 255, (40.0 * self.animated_progress) as u8
            );
            ui.painter().rect_filled(shine_rect, 6.0, shine_color);
        }

        // Text overlay
        let text = if let Some(label) = &self.label {
            if self.show_percentage {
                format!("{} ({:.0}%)", label, self.animated_progress * 100.0)
            } else {
                label.clone()
            }
        } else if self.show_percentage {
            format!("{:.0}%", self.animated_progress * 100.0)
        } else {
            String::new()
        };
        
        if !text.is_empty() {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                text,
                egui::FontId::proportional(11.0),
                egui::Color32::WHITE,
            );
        }
    }
}

pub struct PulsatingDot {
    phase: f32,
}

impl Default for PulsatingDot {
    fn default() -> Self {
        Self { phase: 0.0 }
    }
}

impl PulsatingDot {
    pub fn show(&mut self, ui: &mut egui::Ui, pos: egui::Pos2, base_radius: f32, color: egui::Color32) {
        let dt = ui.input(|i| i.stable_dt);
        self.phase += dt * 3.0;

        let pulse = self.phase.sin() * 0.3 + 0.7;
        let radius = base_radius * pulse;
        let alpha = (pulse * 255.0) as u8;
        
        let pulsed_color = egui::Color32::from_rgba_unmultiplied(
            color.r(),
            color.g(), 
            color.b(),
            alpha
        );

        ui.painter().circle_filled(pos, radius, pulsed_color);
    }
}

pub struct SystemStatusComponent {
    system_info: SystemInfo,
    last_update: Instant,
    update_interval: Duration,
    show_details: bool,
}

impl Default for SystemStatusComponent {
    fn default() -> Self {
        Self {
            system_info: SystemInfo::new(),
            last_update: Instant::now(),
            update_interval: Duration::from_secs(2), // Update every 2 seconds
            show_details: false,
        }
    }
}

impl SystemStatusComponent {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn render(&mut self, ui: &mut egui::Ui) {
        // Update system info periodically
        if self.last_update.elapsed() > self.update_interval {
            self.system_info.refresh();
            self.last_update = Instant::now();
        }

        ui.collapsing("📊 System Status", |ui| {
            let mem_info = self.system_info.get_memory_info();
            let cpu_info = self.system_info.get_cpu_info();
            
            ui.horizontal(|ui| {
                // Memory usage bar
                if let (Some(used_str), Some(total_str), Some(usage_percent)) = (
                    mem_info.get("used"),
                    mem_info.get("total"),
                    mem_info.get("usage_percent")
                ) {
                    let usage_val = usage_percent.replace('%', "").parse::<f32>().unwrap_or(0.0) / 100.0;
                    ui.label("RAM:");
                    let color = if usage_val > 0.85 { egui::Color32::RED } 
                              else if usage_val > 0.70 { egui::Color32::YELLOW } 
                              else { egui::Color32::GREEN };
                    ui.add(egui::ProgressBar::new(usage_val)
                        .text(format!("{}/{}", used_str, total_str))
                        .fill(color)
                        .desired_width(150.0));
                }
                
                ui.separator();
                
                // CPU usage
                if let Some(cpu_usage) = cpu_info.get("usage") {
                    ui.label(format!("CPU: {}", cpu_usage));
                }
            });

            if ui.button(if self.show_details { "Hide Details" } else { "Show Details" }).clicked() {
                self.show_details = !self.show_details;
            }

            if self.show_details {
                ui.separator();
                
                ui.group(|ui| {
                    ui.label(egui::RichText::new("Memory Details").strong());
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            for (key, value) in &mem_info {
                                ui.label(format!("{}: {}", key.replace('_', " "), value));
                            }
                        });
                        
                        ui.separator();
                        
                        ui.vertical(|ui| {
                            ui.label(egui::RichText::new("CPU Info").strong());
                            for (key, value) in &cpu_info {
                                if key != "usage" { // Already shown above
                                    ui.label(format!("{}: {}", key.replace('_', " "), value));
                                }
                            }
                        });
                    });
                });

                // NPU/Compute devices
                let devices = self.system_info.get_available_compute_devices();
                if devices.len() > 1 {
                    ui.group(|ui| {
                        ui.label(egui::RichText::new("Compute Devices").strong());
                        for device in &devices {
                            let icon = if device.contains("NPU") { "🧠" } else if device.contains("GPU") { "🎮" } else { "🖥️" };
                            ui.label(format!("{} {}", icon, device));
                        }
                    });
                }
            }
        });
    }

    pub fn get_memory_usage_percent(&self) -> f32 {
        let mem_info = self.system_info.get_memory_info();
        if let Some(usage_percent) = mem_info.get("usage_percent") {
            usage_percent.replace('%', "").parse::<f32>().unwrap_or(0.0)
        } else {
            0.0
        }
    }

    pub fn is_memory_high(&self) -> bool {
        self.get_memory_usage_percent() > 85.0 // Warning threshold
    }

    pub fn get_cpu_usage_percent(&self) -> f32 {
        let cpu_info = self.system_info.get_cpu_info();
        if let Some(cpu_usage) = cpu_info.get("usage") {
            cpu_usage.replace('%', "").parse::<f32>().unwrap_or(0.0)
        } else {
            0.0
        }
    }

    pub fn get_disk_usage_percent(&self) -> f32 {
        // Simplified disk usage - could be enhanced to show specific drive
        // For now, we'll estimate based on available info
        50.0 // Placeholder - would need proper disk monitoring
    }

    /// Render compact status bar for top of application
    pub fn render_status_bar(&mut self, ui: &mut egui::Ui) {
        // Update system info periodically (more frequently for status bar)
        if self.last_update.elapsed() > Duration::from_millis(1500) {
            self.system_info.refresh();
            self.last_update = Instant::now();
        }

        ui.horizontal(|ui| {
            ui.add_space(8.0);
            
            // Memory indicator
            let mem_percent = self.get_memory_usage_percent();
            let mem_color = if mem_percent > 85.0 { 
                egui::Color32::from_rgb(255, 107, 107) 
            } else if mem_percent > 70.0 { 
                egui::Color32::from_rgb(255, 193, 7) 
            } else { 
                egui::Color32::from_rgb(34, 197, 94) 
            };
            
            ui.colored_label(mem_color, "💾");
            ui.add(egui::ProgressBar::new(mem_percent / 100.0)
                .fill(mem_color)
                .desired_width(40.0)
                .show_percentage());
            
            ui.add_space(8.0);
            
            // CPU indicator  
            let cpu_percent = self.get_cpu_usage_percent();
            let cpu_color = if cpu_percent > 85.0 { 
                egui::Color32::from_rgb(255, 107, 107) 
            } else if cpu_percent > 70.0 { 
                egui::Color32::from_rgb(255, 193, 7) 
            } else { 
                egui::Color32::from_rgb(34, 197, 94) 
            };
            
            ui.colored_label(cpu_color, "🖥️");
            ui.add(egui::ProgressBar::new(cpu_percent / 100.0)
                .fill(cpu_color)
                .desired_width(40.0)
                .show_percentage());
            
            ui.add_space(8.0);
            
            // Disk indicator
            let disk_percent = self.get_disk_usage_percent();
            let disk_color = if disk_percent > 90.0 { 
                egui::Color32::from_rgb(255, 107, 107) 
            } else if disk_percent > 75.0 { 
                egui::Color32::from_rgb(255, 193, 7) 
            } else { 
                egui::Color32::from_rgb(34, 197, 94) 
            };
            
            ui.colored_label(disk_color, "💿");
            ui.add(egui::ProgressBar::new(disk_percent / 100.0)
                .fill(disk_color)
                .desired_width(40.0)
                .show_percentage());
            
            ui.add_space(12.0);
            ui.separator();
            ui.add_space(8.0);
            
            // GPU/NPU indicators
            let devices = self.system_info.get_available_compute_devices();
            let mut has_gpu = false;
            let mut has_npu = false;
            
            for device in &devices {
                if device.to_lowercase().contains("gpu") || device.to_lowercase().contains("nvidia") || device.to_lowercase().contains("amd") {
                    has_gpu = true;
                } else if device.to_lowercase().contains("npu") || device.to_lowercase().contains("neural") {
                    has_npu = true;
                }
            }
            
            // GPU indicator (placeholder usage)
            if has_gpu {
                let gpu_percent = 25.0; // Placeholder - would need proper GPU monitoring
                let gpu_color = if gpu_percent > 85.0 { 
                    egui::Color32::from_rgb(255, 107, 107) 
                } else if gpu_percent > 70.0 { 
                    egui::Color32::from_rgb(255, 193, 7) 
                } else { 
                    egui::Color32::from_rgb(34, 197, 94) 
                };
                
                ui.colored_label(gpu_color, "🎮");
                ui.add(egui::ProgressBar::new(gpu_percent / 100.0)
                    .fill(gpu_color)
                    .desired_width(40.0)
                    .show_percentage());
                ui.add_space(8.0);
            }
            
            // NPU indicator
            if has_npu {
                let npu_percent = 0.0; // Would show actual NPU usage when model is loaded
                let npu_color = if npu_percent > 85.0 { 
                    egui::Color32::from_rgb(255, 107, 107) 
                } else if npu_percent > 70.0 { 
                    egui::Color32::from_rgb(255, 193, 7) 
                } else if npu_percent > 0.0 {
                    egui::Color32::from_rgb(34, 197, 94) 
                } else {
                    egui::Color32::GRAY
                };
                
                ui.colored_label(npu_color, "🧠");
                ui.add(egui::ProgressBar::new(npu_percent / 100.0)
                    .fill(npu_color)
                    .desired_width(40.0)
                    .show_percentage());
            } else {
                // Show NPU as unavailable
                ui.colored_label(egui::Color32::GRAY, "🧠");
                ui.label(
                    egui::RichText::new("N/A")
                        .size(10.0)
                        .color(egui::Color32::GRAY)
                );
            }
            
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(8.0);
                
                // System time
                let now = chrono::Local::now();
                ui.label(
                    egui::RichText::new(now.format("%H:%M:%S").to_string())
                        .size(11.0)
                        .color(egui::Color32::GRAY)
                );
                
                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);
                
                // Compact system info
                let mem_info = self.system_info.get_memory_info();
                if let (Some(used_str), Some(total_str)) = (mem_info.get("used"), mem_info.get("total")) {
                    ui.label(
                        egui::RichText::new(format!("📊 {}/{}", used_str, total_str))
                            .size(10.0)
                            .color(egui::Color32::GRAY)
                    );
                }
            });
        });
    }
}

// Helper function to interpolate between colors
fn lerp_color32(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    egui::Color32::from_rgba_unmultiplied(
        (a.r() as f32 * (1.0 - t) + b.r() as f32 * t) as u8,
        (a.g() as f32 * (1.0 - t) + b.g() as f32 * t) as u8,
        (a.b() as f32 * (1.0 - t) + b.b() as f32 * t) as u8,
        (a.a() as f32 * (1.0 - t) + b.a() as f32 * t) as u8,
    )
}
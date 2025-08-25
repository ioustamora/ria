use crate::ai::models::{ModelInfo, ModelManager, ModelType, QuantizationType};
use crate::ai::ExecutionProvider;
use crate::ui::components::{DownloadProgressCard, DownloadInfo, DownloadStatus, SystemLoadingIndicator};
use eframe::egui;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use serde::{Deserialize, Serialize};

use std::time::{Instant, Duration};

pub struct ModelManagerUI {
    manager: Arc<RwLock<ModelManager>>,
    available_models: Vec<ModelInfo>,
    selected_model: Option<String>,
    download_url: String,
    download_name: String,
    downloading: HashMap<String, DownloadProgressCard>, // model_name -> download info
    progress_rx: mpsc::UnboundedReceiver<ProgressUpdate>, // Progress updates from download tasks
    progress_tx: mpsc::UnboundedSender<ProgressUpdate>, // Send progress updates
    scanning: bool,
    error_message: Option<String>,
    success_message: Option<String>,
    show_remote_models: bool,
    remote_models: Vec<RemoteModelInfo>,
    current_tab: ModelTab,
    system_models: Vec<ModelInfo>,
    system_models_loaded: bool,
    system_loading: Option<SystemLoadingIndicator>,
    tab_loading_states: HashMap<ModelTab, bool>,
    show_help: bool, // Show help overlay
    last_model_update: Option<Instant>, // Track when we last updated models
}

#[derive(Debug, Clone)]
struct ProgressUpdate {
    model_name: String,
    downloaded_bytes: u64,
    total_bytes: u64,
    speed_bps: f64,
    status: DownloadStatus,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ModelTab {
    Local,
    System,
    Remote,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RemoteModelInfo {
    pub name: String,
    pub description: String,
    pub url: String,
    pub size_mb: f64,
    pub model_type: ModelType,
    pub quantization: QuantizationType,
    pub requirements: String,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub tokenizer_url: Option<String>,
}

impl ModelManagerUI {
    pub fn new() -> Self {
        let models_dir = std::env::current_dir()
            .unwrap_or_default()
            .join("models");
        
        let manager = Arc::new(RwLock::new(
            ModelManager::new(&models_dir).unwrap_or_else(|_| {
                ModelManager::new(".").expect("Failed to create model manager")
            })
        ));

        // Create progress update channel
        let (progress_tx, progress_rx) = mpsc::unbounded_channel();

        let mut ui = Self {
            manager,
            available_models: Vec::new(),
            selected_model: None,
            download_url: String::new(),
            download_name: String::new(),
            downloading: HashMap::new(),
            progress_rx,
            progress_tx,
            scanning: false,
            error_message: None,
            success_message: None,
            show_remote_models: false,
            remote_models: Vec::new(),
            current_tab: ModelTab::Local,
            system_models: Vec::new(),
            system_models_loaded: false,
            system_loading: None,
            tab_loading_states: HashMap::new(),
            show_help: false,
            last_model_update: None,
        };

        ui.load_remote_models();
        ui
    }
    
    fn switch_to_tab(&mut self, tab: ModelTab) {
        // Clear previous errors/messages when switching tabs
        self.error_message = None;
        self.success_message = None;
        
        // Set loading state for tab transition
        self.tab_loading_states.insert(tab.clone(), true);
        
        // Update current tab and related state
        self.current_tab = tab.clone();
        match tab {
            ModelTab::Local => {
                self.show_remote_models = false;
                // Clear loading state immediately for local models (no async loading)
                self.tab_loading_states.insert(ModelTab::Local, false);
            },
            ModelTab::System => {
                self.show_remote_models = false;
                // Loading state will be cleared by system model loading process
            },
            ModelTab::Remote => {
                self.show_remote_models = true;
                // Clear loading state immediately (remote models are pre-loaded)
                self.tab_loading_states.insert(ModelTab::Remote, false);
            },
        }
    }

    pub fn refresh_models(&mut self) {
        self.scanning = true;
        self.system_models_loaded = false; // Force re-scan of system models
        
        let manager = self.manager.clone();
        
        // Spawn background task to scan both local and system models
        tokio::spawn(async move {
            let mut guard = manager.write().await;
            tracing::info!("Starting background model scan...");
            match guard.scan_models() {
                Ok(()) => {
                    let model_count = guard.get_available_models().len();
                    tracing::info!("Background model scan completed - found {} models", model_count);
                },
                Err(e) => {
                    tracing::error!("Failed to scan local models: {}", e);
                }
            }
        });
        
        // Load system models in background
        self.load_system_models();
        
        // Update local models immediately (sync scan)
        self.update_available_models();
        
        // Reset scanning state
        self.scanning = false;
    }
    
    fn update_available_models(&mut self) {
        // Get the current available models from the manager synchronously
        if let Ok(guard) = self.manager.try_read() {
            let models_before = self.available_models.len();
            self.available_models = guard.get_available_models().to_vec();
            self.last_model_update = Some(Instant::now());
            
            let models_after = self.available_models.len();
            if models_after != models_before {
                tracing::info!("Model list updated: {} -> {} models", models_before, models_after);
                // Show models found in debug
                for model in &self.available_models {
                    tracing::debug!("Found model: {} ({})", model.name, model.path.display());
                }
            }
        } else {
            tracing::debug!("Could not acquire read lock for model update, scheduling retry");
            // If we can't get a read lock, try again after a short delay
            let manager = self.manager.clone();
            tokio::spawn(async move {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                let guard = manager.read().await;
                let models = guard.get_available_models().to_vec();
                tracing::info!("Delayed scan found {} local models", models.len());
                // Note: We can't update the UI from here due to async context
                // The UI will need to poll for updates or we need a different approach
            });
        }
    }
    
    fn update_available_models_if_needed(&mut self) {
        // Update models every 2 seconds or if never updated
        let should_update = match self.last_model_update {
            None => true,
            Some(last_update) => last_update.elapsed() > Duration::from_secs(2),
        };
        
        if should_update {
            self.update_available_models();
        }
    }
    
    fn load_system_models(&mut self) {
        if self.system_models_loaded {
            return; // Already loaded
        }
        
        let manager = self.manager.clone();
        
        // Spawn background task to detect system models
        tokio::spawn(async move {
            let guard = manager.read().await;
            let detected_models = guard.detect_system_models();
            tracing::info!("System model detection completed: {} models found", detected_models.len());
            // Note: In a real app, you'd need to communicate back to the UI thread
            // For now, the detection happens but we can't update the UI from here
        });
    }

    fn handle_keyboard_shortcuts(&mut self, ui: &mut egui::Ui) {
        // Handle keyboard shortcuts for better UX
        ui.input(|i| {
            // Ctrl+R or F5 to refresh
            if i.key_pressed(egui::Key::F5) || (i.modifiers.ctrl && i.key_pressed(egui::Key::R)) {
                self.refresh_models();
            }
            
            // Tab navigation: Ctrl+1, Ctrl+2, Ctrl+3 for tabs
            if i.modifiers.ctrl && i.key_pressed(egui::Key::Num1) {
                self.switch_to_tab(ModelTab::Local);
            }
            if i.modifiers.ctrl && i.key_pressed(egui::Key::Num2) {
                self.switch_to_tab(ModelTab::System);
            }
            if i.modifiers.ctrl && i.key_pressed(egui::Key::Num3) {
                self.switch_to_tab(ModelTab::Remote);
            }
            
            // Escape to clear messages
            if i.key_pressed(egui::Key::Escape) {
                if self.show_help {
                    self.show_help = false;
                } else {
                    self.error_message = None;
                    self.success_message = None;
                }
            }
            
            // F1 or Ctrl+H to show help
            if i.key_pressed(egui::Key::F1) || (i.modifiers.ctrl && i.key_pressed(egui::Key::H)) {
                self.show_help = !self.show_help;
            }
        });
    }

    fn handle_progress_updates(&mut self) {
        // Process all pending progress updates
        while let Ok(update) = self.progress_rx.try_recv() {
            if let Some(download_card) = self.downloading.get_mut(&update.model_name) {
                let progress = if update.total_bytes > 0 { 
                    update.downloaded_bytes as f32 / update.total_bytes as f32 
                } else { 
                    0.0 
                };
                
                let eta_seconds = if update.speed_bps > 0.0 && update.total_bytes > update.downloaded_bytes {
                    (update.total_bytes - update.downloaded_bytes) as f64 / update.speed_bps
                } else {
                    0.0
                };

                let updated_info = DownloadInfo {
                    name: update.model_name.clone(),
                    progress,
                    total_bytes: update.total_bytes,
                    downloaded_bytes: update.downloaded_bytes,
                    speed_bps: update.speed_bps,
                    eta_seconds,
                    status: update.status.clone(),
                };
                
                download_card.update(updated_info);
                
                // Remove completed downloads after showing success
                if matches!(update.status, DownloadStatus::Completed) {
                    self.success_message = Some(format!("Successfully downloaded {}", update.model_name));
                    // Remove the download card after completion
                    // Force immediate model update since we just downloaded a model
                    self.last_model_update = None; // Force update on next render
                }
                
                if let DownloadStatus::Failed(error) = &update.status {
                    self.error_message = Some(format!("Failed to download {}: {}", update.model_name, error));
                    // Keep failed download visible for user to see
                }
            }
        }
    }

    pub fn render(&mut self, ui: &mut egui::Ui) {
        // Handle any pending download progress updates
        self.handle_progress_updates();
        
        // Handle keyboard shortcuts
        self.handle_keyboard_shortcuts(ui);
        
        // Periodically update available models to catch newly downloaded files
        self.update_available_models_if_needed();
        
        ui.heading("🧠 AI Model Management");
        ui.separator();
        ui.add_space(10.0);

        // Enhanced tabs with loading indicators
        ui.horizontal(|ui| {
            // Local Models tab
            let local_label = if *self.tab_loading_states.get(&ModelTab::Local).unwrap_or(&false) {
                "📁 Local Models ⏳"
            } else {
                "📁 Local Models"
            };
            if ui.selectable_label(self.current_tab == ModelTab::Local, local_label).clicked() {
                self.switch_to_tab(ModelTab::Local);
            }
            
            // System Models tab
            let system_label = if self.system_loading.is_some() {
                "🔍 System Models ⏳"
            } else if *self.tab_loading_states.get(&ModelTab::System).unwrap_or(&false) {
                "🔍 System Models ⏳"
            } else {
                "🔍 System Models"
            };
            if ui.selectable_label(self.current_tab == ModelTab::System, system_label).clicked() {
                self.switch_to_tab(ModelTab::System);
            }
            
            // Remote Models tab
            let remote_label = if *self.tab_loading_states.get(&ModelTab::Remote).unwrap_or(&false) {
                "🌐 Remote Models ⏳"
            } else {
                "🌐 Remote Models"
            };
            if ui.selectable_label(self.current_tab == ModelTab::Remote, remote_label).clicked() {
                self.switch_to_tab(ModelTab::Remote);
            }
            
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // Help button
                if ui.button("❓")
                    .on_hover_text("Show keyboard shortcuts (F1)")
                    .clicked() {
                    self.show_help = !self.show_help;
                }
                
                if ui.button("🔄 Refresh")
                    .on_hover_text("Refresh and rescan all model directories (F5 or Ctrl+R)")
                    .clicked() {
                    self.refresh_models();
                }
            });
        });

        ui.add_space(10.0);

        match self.current_tab {
            ModelTab::Local => self.render_local_models(ui),
            ModelTab::System => self.render_system_models(ui),
            ModelTab::Remote => self.render_remote_models(ui),
        }

        ui.add_space(20.0);
        self.render_status_messages(ui);
        
        // Show help overlay if requested
        if self.show_help {
            self.render_help_overlay(ui);
        }
    }

    fn render_local_models(&mut self, ui: &mut egui::Ui) {
        // Model directory info
        ui.horizontal(|ui| {
            ui.label("Models Directory:");
            ui.code("./models/");
            if ui.button("📂 Open Folder")
                .on_hover_text("Open the models directory in your file explorer")
                .clicked() {
                let models_dir = std::env::current_dir().unwrap_or_default().join("models");
                if let Err(e) = std::fs::create_dir_all(&models_dir) {
                    self.error_message = Some(format!("Failed to create models directory: {}", e));
                } else {
                    // Try to open the folder in file manager
                    let _ = std::process::Command::new("explorer")
                        .arg(models_dir)
                        .spawn();
                }
            }
        });

        ui.add_space(10.0);

        // Local models list
        egui::ScrollArea::vertical()
            .max_height(400.0)
            .show(ui, |ui| {
                if self.available_models.is_empty() {
                    ui.centered_and_justified(|ui| {
                        ui.vertical_centered(|ui| {
                            ui.add_space(50.0);
                            ui.label("No ONNX models found");
                            ui.add_space(10.0);
                            ui.label("Add .onnx files to the models/ directory");
                            ui.add_space(10.0);
                            if ui.button("Download Popular Models")
                                .on_hover_text("Browse and download pre-configured ONNX models")
                                .clicked() {
                                self.show_remote_models = true;
                            }
                        });
                    });
                } else {
                    let models_clone = self.available_models.clone();
                    for model in &models_clone {
                        self.render_local_model_card(ui, model);
                        ui.add_space(5.0);
                    }
                }
            });

        // Manual model addition with enhanced UX
        ui.separator();
        ui.add_space(10.0);
        ui.horizontal(|ui| {
            ui.strong("Add Model Manually");
            ui.label("ℹ️").on_hover_text("You can download ONNX models directly from URLs like Hugging Face or other repositories");
        });
        ui.add_space(5.0);
        
        ui.horizontal(|ui| {
            ui.label("URL:");
            ui.text_edit_singleline(&mut self.download_url)
                .on_hover_text("Enter direct URL to .onnx model file (e.g., from Hugging Face)");
        });
        ui.horizontal(|ui| {
            ui.label("Name:");
            ui.text_edit_singleline(&mut self.download_name)
                .on_hover_text("Choose a friendly name for this model");
            if ui.button("📥 Download")
                .on_hover_text("Download model from the URL above")
                .clicked() && !self.download_url.is_empty() && !self.download_name.is_empty() {
                self.start_download(self.download_url.clone(), self.download_name.clone());
            }
        });
    }

    fn render_remote_models(&mut self, ui: &mut egui::Ui) {
        ui.label("Popular ONNX Models:");
        ui.add_space(10.0);

        egui::ScrollArea::vertical()
            .max_height(500.0)
            .show(ui, |ui| {
                for model in &self.remote_models.clone() {
                    self.render_remote_model_card(ui, model);
                    ui.add_space(10.0);
                }
            });
    }

    fn render_local_model_card(&mut self, ui: &mut egui::Ui, model: &ModelInfo) {
        egui::Frame::none()
            .fill(egui::Color32::from_rgb(40, 40, 50))
            .rounding(8.0)
            .inner_margin(15.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    // Model icon based on type
                    let icon = match model.model_type {
                        ModelType::LanguageModel => "🗣️",
                        ModelType::ChatModel => "💬",
                        ModelType::CodeModel => "💻",
                        ModelType::MultiModal => "🎭",
                    };
                    
                    ui.label(egui::RichText::new(icon).size(24.0));
                    
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(&model.name).size(16.0).strong());
                            
                            // Selection radio button
                            let selected = self.selected_model.as_ref() == Some(&model.name);
                            if ui.radio(selected, "Use")
                                .on_hover_text("Select this model for AI chat")
                                .clicked() {
                                self.selected_model = Some(model.name.clone());
                            }
                        });
                        
                        ui.label(format!("Size: {}", crate::ai::models::ModelManager::format_file_size(model.size)));
                        ui.label(format!("Type: {:?}", model.model_type));
                        if let Some(quant) = &model.quantization {
                            ui.label(format!("Quantization: {:?}", quant));
                        }
                        ui.label(format!("Path: {}", model.path.display()));
                    });
                    
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("🗑️ Delete")
                            .on_hover_text("Permanently delete this model from your computer")
                            .clicked() {
                            if let Err(e) = std::fs::remove_file(&model.path) {
                                self.error_message = Some(format!("Failed to delete model: {}", e));
                            } else {
                                self.success_message = Some("Model deleted successfully".to_string());
                            }
                        }
                    });
                });
                
                // Supported providers
                ui.add_space(5.0);
                ui.horizontal(|ui| {
                    ui.label("Supports:");
                    for provider in &model.supported_providers {
                        let color = match provider {
                            ExecutionProvider::Cpu => egui::Color32::GRAY,
                            ExecutionProvider::Cuda => egui::Color32::GREEN,
                            ExecutionProvider::DirectML => egui::Color32::BLUE,
                            ExecutionProvider::CoreML => egui::Color32::from_rgb(255, 165, 0),
                            _ => egui::Color32::LIGHT_GRAY,
                        };
                        ui.colored_label(color, format!("{:?}", provider));
                    }
                });
            });
    }

    fn render_system_models(&mut self, ui: &mut egui::Ui) {
        // Information header
        ui.horizontal(|ui| {
            ui.label("🔍 System Model Detection");
            ui.separator();
            ui.label("Scanning Windows Copilot+ PC, Intel NPU, and common AI model locations...");
        });
        ui.add_space(10.0);

        // Load system models on first access to this tab
        if !self.system_models_loaded && self.current_tab == ModelTab::System {
            self.start_system_model_loading();
        }

        // Show loading indicator if system models are being loaded
        if let Some(ref mut loading) = self.system_loading {
            ui.add_space(20.0);
            loading.show(ui);
            ui.add_space(20.0);
        } else if self.system_models.is_empty() {
            ui.horizontal(|ui| {
                ui.label("ℹ️");
                ui.vertical(|ui| {
                    if !self.system_models_loaded {
                        ui.label("Preparing to scan for system models...");
                        ui.add_space(5.0);
                    } else {
                        ui.label("No system models detected on this machine.");
                    }
                    ui.label("System models are typically found on:");
                    ui.label("• Windows Copilot+ PCs (Phi-3 Silica)");
                    ui.label("• Intel NPU optimized systems");
                    ui.label("• Systems with pre-installed AI frameworks");
                });
            });
        } else {
            ui.label(format!("Found {} system model(s):", self.system_models.len()));
            ui.add_space(10.0);
            
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    for model in &self.system_models.clone() {
                        self.render_system_model_card(ui, model);
                        ui.add_space(10.0);
                    }
                });
        }
    }
    
    fn start_system_model_loading(&mut self) {
        if self.system_models_loaded {
            return;
        }
        
        // Initialize loading indicator
        let mut loading = SystemLoadingIndicator::new();
        loading.set_stage("Scanning system directories...".to_string(), 0.1);
        self.system_loading = Some(loading);
        
        // Simulate progressive loading stages
        self.load_system_models_with_progress();
    }
    
    fn load_system_models_with_progress(&mut self) {
        if self.system_models_loaded {
            return;
        }
        
        // Update progress stages
        if let Some(ref mut loading) = self.system_loading {
            loading.set_stage("Checking Windows system paths...".to_string(), 0.3);
        }
        
        // Since we can't use async in the UI thread, we'll create a temporary ModelManager
        // just for system model detection. This is safe since detect_system_models() 
        // only reads from the filesystem and doesn't modify state.
        let temp_manager = crate::ai::models::ModelManager::new("./models").ok();
        
        if let Some(manager) = temp_manager {
            if let Some(ref mut loading) = self.system_loading {
                loading.set_stage("Analyzing discovered models...".to_string(), 0.7);
            }
            
            self.system_models = manager.detect_system_models();
            
            if let Some(ref mut loading) = self.system_loading {
                loading.set_stage("Finalizing detection...".to_string(), 1.0);
            }
            
            self.system_models_loaded = true;
            self.system_loading = None; // Hide loading indicator
            tracing::info!("System model detection completed: {} models found", self.system_models.len());
        } else {
            self.system_loading = None;
            tracing::warn!("Failed to create temporary ModelManager for system model detection");
        }
    }

    fn render_system_model_card(&mut self, ui: &mut egui::Ui, model: &ModelInfo) {
        egui::Frame::none()
            .fill(egui::Color32::from_rgb(40, 50, 40)) // Slightly green tint for system models
            .rounding(8.0)
            .inner_margin(15.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    // Special icon for Phi-3 Silica
                    let icon = if model.name.contains("Phi-3 Silica") {
                        "🧠"
                    } else {
                        match model.model_type {
                            ModelType::LanguageModel => "🗣️",
                            ModelType::ChatModel => "💬", 
                            ModelType::CodeModel => "💻",
                            ModelType::MultiModal => "🎭",
                        }
                    };
                    
                    ui.label(egui::RichText::new(icon).size(24.0));
                    
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(&model.name).size(16.0).strong());
                            
                            // Special badge for Phi-3 Silica
                            if model.name.contains("Phi-3 Silica") {
                                ui.label(egui::RichText::new("COPILOT+ PC")
                                    .size(10.0)
                                    .color(egui::Color32::from_rgb(100, 255, 100))
                                    .background_color(egui::Color32::from_rgb(20, 60, 20)));
                            }
                            
                            // Selection radio button
                            let selected = self.selected_model.as_ref() == Some(&model.name);
                            if ui.radio(selected, "Use")
                                .on_hover_text("Select this model for AI chat")
                                .clicked() {
                                self.selected_model = Some(model.name.clone());
                            }
                        });
                        
                        ui.label(format!("Size: {}", crate::ai::models::ModelManager::format_file_size(model.size)));
                        ui.label(format!("Type: {:?}", model.model_type));
                        if let Some(quant) = &model.quantization {
                            ui.label(format!("Quantization: {:?}", quant));
                        }
                        ui.label(format!("Path: {}", model.path.display()));
                        
                        // Description with truncation for long paths
                        let desc = if model.description.len() > 80 {
                            format!("{}...", &model.description[..77])
                        } else {
                            model.description.clone()
                        };
                        ui.label(egui::RichText::new(desc)
                            .size(11.0)
                            .color(egui::Color32::GRAY));
                    });
                    
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if model.name.contains("Phi-3 Silica") {
                            ui.label(egui::RichText::new("✅ Ready")
                                .color(egui::Color32::GREEN));
                        } else {
                            ui.label(egui::RichText::new("📍 Detected")
                                .color(egui::Color32::YELLOW));
                        }
                    });
                });
                
                // Supported providers
                ui.add_space(5.0);
                ui.horizontal(|ui| {
                    ui.label("Supports:");
                    for provider in &model.supported_providers {
                        let color = match provider {
                            ExecutionProvider::Cpu => egui::Color32::GRAY,
                            ExecutionProvider::Cuda => egui::Color32::GREEN,
                            ExecutionProvider::DirectML => egui::Color32::BLUE,
                            ExecutionProvider::CoreML => egui::Color32::from_rgb(255, 165, 0),
                            ExecutionProvider::OpenVINO => egui::Color32::from_rgb(0, 150, 255),
                            _ => egui::Color32::LIGHT_GRAY,
                        };
                        ui.colored_label(color, format!("{:?}", provider));
                    }
                });
            });
    }

    fn render_remote_model_card(&mut self, ui: &mut egui::Ui, model: &RemoteModelInfo) {
        egui::Frame::none()
            .fill(egui::Color32::from_rgb(30, 40, 50))
            .rounding(8.0)
            .inner_margin(15.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let icon = match model.model_type {
                        ModelType::LanguageModel => "🗣️",
                        ModelType::ChatModel => "💬", 
                        ModelType::CodeModel => "💻",
                        ModelType::MultiModal => "🎭",
                    };
                    
                    ui.label(egui::RichText::new(icon).size(24.0));
                    
                    ui.vertical(|ui| {
                        ui.label(egui::RichText::new(&model.name).size(16.0).strong());
                        ui.label(&model.description);
                        ui.label(format!("Size: {:.1} MB", model.size_mb));
                        ui.label(format!("Type: {:?} ({:?})", model.model_type, model.quantization));
                        ui.label(format!("Requirements: {}", model.requirements));
                        if model.sha256.is_some() { ui.label("Checksum: SHA256 available"); }
                        if model.tokenizer_url.is_some() { ui.label("Tokenizer: available"); }
                    });
                    
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if let Some(download_card) = self.downloading.get_mut(&model.name) {
                            download_card.show(ui);
                        } else if ui.button("📥 Download")
                            .on_hover_text(format!("Download {} ({:.1} MB)", model.name, model.size_mb))
                            .clicked() {
                            self.start_download(model.url.clone(), model.name.clone());
                        }
                    });
                });
            });
    }

    fn render_status_messages(&mut self, ui: &mut egui::Ui) {
        // Enhanced error messages with more context
        if let Some(error) = &self.error_message.clone() {
            egui::Frame::none()
                .fill(egui::Color32::from_rgba_unmultiplied(60, 20, 20, 200))
                .stroke(egui::Stroke::new(1.5, egui::Color32::from_rgb(180, 60, 60)))
                .rounding(6.0)
                .inner_margin(12.0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("⚠️").size(16.0).color(egui::Color32::from_rgb(255, 100, 100)));
                        ui.vertical(|ui| {
                            ui.strong(egui::RichText::new("Error").color(egui::Color32::from_rgb(255, 150, 150)));
                            ui.label(egui::RichText::new(error).color(egui::Color32::WHITE));
                        });
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("✕").on_hover_text("Dismiss error").clicked() {
                                self.error_message = None;
                            }
                        });
                    });
                });
            ui.add_space(8.0);
        }
        
        // Enhanced success messages with better styling
        if let Some(success) = &self.success_message.clone() {
            egui::Frame::none()
                .fill(egui::Color32::from_rgba_unmultiplied(20, 60, 20, 200))
                .stroke(egui::Stroke::new(1.5, egui::Color32::from_rgb(60, 180, 60)))
                .rounding(6.0)
                .inner_margin(12.0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("✅").size(16.0).color(egui::Color32::from_rgb(100, 255, 100)));
                        ui.vertical(|ui| {
                            ui.strong(egui::RichText::new("Success").color(egui::Color32::from_rgb(150, 255, 150)));
                            ui.label(egui::RichText::new(success).color(egui::Color32::WHITE));
                        });
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("✕").on_hover_text("Dismiss notification").clicked() {
                                self.success_message = None;
                            }
                        });
                    });
                });
            ui.add_space(8.0);
        }
        
        // Enhanced scanning indicator
        if self.scanning {
            egui::Frame::none()
                .fill(egui::Color32::from_rgba_unmultiplied(40, 40, 60, 200))
                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(100, 100, 150)))
                .rounding(6.0)
                .inner_margin(10.0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Scanning for models...");
                    });
                });
            ui.add_space(8.0);
        }
        
        // Show active downloads
        if !self.downloading.is_empty() {
            ui.strong("Active Downloads:");
            ui.add_space(5.0);
            
            // Create a separate list to avoid borrowing issues
            let download_names: Vec<String> = self.downloading.keys().cloned().collect();
            for name in download_names {
                if let Some(download_card) = self.downloading.get_mut(&name) {
                    download_card.show(ui);
                    ui.add_space(8.0);
                }
            }
        }
    }

    fn start_download(&mut self, url: String, name: String) {
        tracing::info!("Download requested for: {}", name);
        let manager = self.manager.clone();
        let maybe_entry = self.remote_models.iter().find(|m| m.name == name).cloned();
        
        // Create download progress card
        let download_info = DownloadInfo {
            name: name.clone(),
            progress: 0.0,
            total_bytes: maybe_entry.as_ref().map(|e| (e.size_mb * 1024.0 * 1024.0) as u64).unwrap_or(0),
            downloaded_bytes: 0,
            speed_bps: 0.0,
            eta_seconds: 0.0,
            status: DownloadStatus::Starting,
        };
        
        let download_card = DownloadProgressCard::new(download_info);
        self.downloading.insert(name.clone(), download_card);
        self.success_message = Some(format!("Starting download of {}...", name));

        // Clone progress sender for the async task
        let progress_tx = self.progress_tx.clone();
        let download_name = name.clone();

        tokio::spawn(async move {
            let sha = maybe_entry.as_ref().and_then(|m| m.sha256.as_ref()).map(|s| s.clone());
            let tok_url = maybe_entry.as_ref().and_then(|m| m.tokenizer_url.as_ref()).map(|s| s.clone());

            // Create progress callback that sends updates through the channel
            let progress_callback = {
                let tx = progress_tx.clone();
                let name = download_name.clone();
                move |downloaded: u64, total: u64, speed: f64| {
                    let _ = tx.send(ProgressUpdate {
                        model_name: name.clone(),
                        downloaded_bytes: downloaded,
                        total_bytes: total,
                        speed_bps: speed,
                        status: DownloadStatus::Downloading,
                    });
                }
            };

            let mut guard = manager.write().await;
            match guard.download_model_with_verify_and_progress(&url, &name, sha.as_deref(), Some(progress_callback)).await {
                Ok(model_path) => {
                    tracing::info!("Model downloaded: {}", model_path.display());
                    
                    // Send completion status
                    let _ = progress_tx.send(ProgressUpdate {
                        model_name: download_name.clone(),
                        downloaded_bytes: 0, // Will be updated by progress callback
                        total_bytes: 0,
                        speed_bps: 0.0,
                        status: DownloadStatus::Completed,
                    });
                    
                    // Download tokenizer if provided
                    if let Some(tu) = tok_url {
                        let tok_name = format!("{}.tokenizer.json", crate::utils::sanitize_filename(&name));
                        let tok_path = guard.get_models_directory().join(tok_name);
                        if let Err(e) = guard.download_aux_file(&tu, &tok_path).await {
                            tracing::warn!("Failed to download tokenizer for {}: {}", name, e);
                        } else {
                            tracing::info!("Tokenizer downloaded for {}", name);
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Download failed for {}: {}", name, e);
                    
                    // Send failure status
                    let _ = progress_tx.send(ProgressUpdate {
                        model_name: download_name.clone(),
                        downloaded_bytes: 0,
                        total_bytes: 0,
                        speed_bps: 0.0,
                        status: DownloadStatus::Failed(e.to_string()),
                    });
                }
            }
        });
    }

    fn load_remote_models(&mut self) {
        // Try to load curated Intel NPU-friendly catalog first
        let catalog_path = std::path::Path::new("assets").join("model_catalog").join("intel_npu_onnx.json");
        if let Ok(contents) = std::fs::read_to_string(&catalog_path) {
            match serde_json::from_str::<Vec<RemoteModelInfo>>(&contents) {
                Ok(list) => {
                    tracing::info!("Loaded Intel NPU model catalog: {} entries", list.len());
                    self.remote_models = list;
                    return;
                }
                Err(e) => {
                    tracing::warn!("Failed to parse model catalog {}: {}", catalog_path.display(), e);
                }
            }
        } else {
            tracing::info!("Model catalog not found at {} - using built-in list", catalog_path.display());
        }

        // Popular ONNX models that work well for chat (fallback)
        self.remote_models = vec![
            RemoteModelInfo {
                name: "Phi-3-mini-4k-instruct".to_string(),
                description: "Microsoft's 3.8B parameter model optimized for chat and reasoning".to_string(),
                url: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx".to_string(),
                size_mb: 2400.0,
                model_type: ModelType::ChatModel,
                quantization: QuantizationType::INT4,
                requirements: "4GB RAM".to_string(),
                sha256: None,
                tokenizer_url: None,
            },
            RemoteModelInfo {
                name: "TinyLlama-1.1B-Chat".to_string(),
                description: "Compact 1.1B parameter chat model, great for testing".to_string(),
                url: "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0-ONNX/resolve/main/model.onnx".to_string(),
                size_mb: 2200.0,
                model_type: ModelType::ChatModel,
                quantization: QuantizationType::FP32,
                requirements: "2GB RAM".to_string(),
                sha256: None,
                tokenizer_url: None,
            },
            RemoteModelInfo {
                name: "CodeQwen1.5-7B-Chat".to_string(),
                description: "Code-specialized 7B model for programming assistance".to_string(),
                url: "https://example.com/codeqwen-7b.onnx".to_string(),
                size_mb: 14000.0,
                model_type: ModelType::CodeModel,
                quantization: QuantizationType::INT8,
                requirements: "16GB RAM".to_string(),
                sha256: None,
                tokenizer_url: None,
            },
            RemoteModelInfo {
                name: "Qwen2-0.5B-Instruct".to_string(),
                description: "Ultra-lightweight 0.5B model for basic chat".to_string(),
                url: "https://example.com/qwen2-0.5b.onnx".to_string(),
                size_mb: 1000.0,
                model_type: ModelType::ChatModel,
                quantization: QuantizationType::INT8,
                requirements: "1GB RAM".to_string(),
                sha256: None,
                tokenizer_url: None,
            },
        ];
    }

    pub fn get_selected_model(&self) -> Option<String> {
        self.selected_model.clone()
    }

    pub fn get_selected_model_info(&self) -> Option<ModelInfo> {
        if let Some(selected_name) = &self.selected_model {
            // For now, simulate model info since we need this to be sync
            // In a real implementation, you'd use channels or store model info locally
            Some(ModelInfo {
                name: selected_name.clone(),
                path: std::path::PathBuf::from(format!("./models/{}.onnx", selected_name)),
                size: 1000000, // 1MB placeholder
                model_type: crate::ai::models::ModelType::ChatModel,
                supported_providers: vec![crate::ai::ExecutionProvider::Cpu],
                description: format!("Simulated model info for {}", selected_name),
                quantization: Some(crate::ai::models::QuantizationType::FP32),
            })
        } else {
            None
        }
    }
    
    fn render_help_overlay(&mut self, ui: &mut egui::Ui) {
        // Show help overlay as a popup window
        egui::Window::new("🔧 Keyboard Shortcuts")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ui.ctx(), |ui| {
                ui.vertical(|ui| {
                    ui.heading("Navigation");
                    ui.label("Ctrl+1, Ctrl+2, Ctrl+3 - Switch between tabs");
                    ui.label("F5 or Ctrl+R - Refresh models");
                    ui.label("Escape - Clear messages/close help");
                    
                    ui.add_space(10.0);
                    ui.heading("Help & Actions");
                    ui.label("F1 or Ctrl+H - Show/hide this help");
                    ui.label("📂 Button - Open models folder");
                    ui.label("❓ Button - Toggle help overlay");
                    
                    ui.add_space(10.0);
                    ui.heading("Download Features");
                    ui.label("📥 Download button - Download selected model");
                    ui.label("❌ Cancel button - Cancel active download");
                    ui.label("Progress bars show real-time download status");
                    
                    ui.add_space(15.0);
                    ui.horizontal(|ui| {
                        if ui.button("Close").clicked() {
                            self.show_help = false;
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.small("Press Escape or F1 to close");
                        });
                    });
                });
            });
    }
}
use crate::ai::*;
use crate::ai::inference::InferenceEngine;
use crate::ai::providers::OnnxProvider;
use crate::config::AppConfig;
use crate::ui::models::ModelManagerUI;
use crate::ui::components::SystemStatusComponent;
use eframe::egui;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TryRecvError;
use std::time::Instant;
use crate::ai::inference::BasicDemoProvider;

pub struct RiaApp {
    chat_sessions: Vec<ChatSession>,
    current_session: Option<usize>,
    input_text: String,
    inference_engine: Arc<RwLock<InferenceEngine>>,
    config: AppConfig,
    show_settings: bool,
    show_models: bool,
    animation_time: f32,
    theme: Theme,
    model_manager: ModelManagerUI,
    model_loaded: bool,
    generating_response: bool,
    // Streaming state
    streaming_rx: Option<mpsc::Receiver<String>>,
    streaming_buffer: String,
    streaming_start: Option<Instant>,
    system_status: SystemStatusComponent,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Theme {
    Dark,
    Light,
    System,
}

impl Default for Theme {
    fn default() -> Self {
        Theme::Dark
    }
}

impl RiaApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Configure fonts
        let fonts = egui::FontDefinitions::default();
        
        // For now, use default fonts. In a real implementation, you would:
        // 1. Include Inter font or other modern fonts
        // 2. Load them from assets directory
        // fonts.font_data.insert("Inter".to_owned(), egui::FontData::from_static(include_bytes!("../../assets/Inter-Regular.ttf")));

        cc.egui_ctx.set_fonts(fonts);

        // Set dark theme
        cc.egui_ctx.set_visuals(egui::Visuals::dark());

        Self {
            chat_sessions: Vec::new(),
            current_session: None,
            input_text: String::new(),
            inference_engine: Arc::new(RwLock::new(InferenceEngine::new())),
            config: AppConfig::default(),
            show_settings: false,
            show_models: false,
            animation_time: 0.0,
            theme: Theme::Dark,
            model_manager: ModelManagerUI::new(),
            model_loaded: false,
            generating_response: false,
            streaming_rx: None,
            streaming_buffer: String::new(),
            streaming_start: None,
            system_status: SystemStatusComponent::new(),
        }
    }

    fn create_new_session(&mut self) {
        let session = ChatSession {
            id: uuid::Uuid::new_v4().to_string(),
            title: format!("Chat {}", self.chat_sessions.len() + 1),
            messages: Vec::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        
        self.chat_sessions.push(session);
        self.current_session = Some(self.chat_sessions.len() - 1);
    }

    fn send_message(&mut self, ctx: &egui::Context) {
        if self.input_text.trim().is_empty() || self.generating_response {
            return;
        }

        if self.current_session.is_none() {
            self.create_new_session();
        }

        let session_idx = self.current_session.unwrap();
        let user_message = ChatMessage {
            id: uuid::Uuid::new_v4().to_string(),
            content: self.input_text.clone(),
            role: MessageRole::User,
            timestamp: chrono::Utc::now(),
            model_used: None,
            inference_time: None,
        };

        self.chat_sessions[session_idx].messages.push(user_message.clone());
        let user_input = self.input_text.clone();
        self.input_text.clear();
        self.generating_response = true;

        // Kick off streaming generation via inference engine. If no provider is loaded,
        // the engine will fall back to a demo provider.
        let messages_snapshot = self.chat_sessions[session_idx].messages.clone();
        let engine_arc = self.inference_engine.clone();
        let (ui_tx, ui_rx) = mpsc::channel(64);
        self.streaming_rx = Some(ui_rx);
        self.streaming_buffer.clear();
        self.streaming_start = Some(Instant::now());

        // Start a background task to stream chunks
        tokio::spawn(async move {
            let mut engine = engine_arc.write().await;

            // Ensure there is at least one provider; if not, add a demo provider
            if !engine.has_active_provider() {
                let idx = engine.add_provider_sync(Box::new(BasicDemoProvider));
                let _ = engine.set_active_provider_sync(idx);
            }

            // Reasonable defaults: ~16 chars per chunk, 20ms delay
            let chunk_chars = 16usize;
            let delay_ms = 20u64;

            match engine.generate_response_stream(&messages_snapshot, chunk_chars, delay_ms) {
                Ok(mut rx) => {
                    while let Some(chunk) = rx.recv().await {
                        if ui_tx.send(chunk).await.is_err() {
                            break;
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Streaming generation failed: {}", e);
                }
            }
            // Drop tx to signal completion
        });

        // Display typing indicator; final message will be appended when streaming ends
    }

    fn render_sidebar(&mut self, _ctx: &egui::Context, ui: &mut egui::Ui) {
        ui.with_layout(egui::Layout::top_down(egui::Align::LEFT), |ui| {
            // Header with app title
            ui.add_space(20.0);
            ui.horizontal(|ui| {
                ui.add_space(20.0);
                ui.label(
                    egui::RichText::new("🤖 RIA AI Chat")
                        .size(24.0)
                        .strong()
                        .color(egui::Color32::from_rgb(100, 200, 255))
                );
            });
            
            ui.add_space(30.0);

            // New Chat button
            ui.horizontal(|ui| {
                ui.add_space(20.0);
                if ui.add_sized([200.0, 40.0], egui::Button::new("➕ New Chat")).clicked() {
                    self.create_new_session();
                }
            });

            ui.add_space(20.0);
            ui.separator();
            ui.add_space(20.0);

            // Chat sessions list
            ui.horizontal(|ui| {
                ui.add_space(20.0);
                ui.label(egui::RichText::new("Recent Chats").size(16.0).strong());
            });

            ui.add_space(10.0);

            for (i, session) in self.chat_sessions.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.add_space(20.0);
                    let selected = self.current_session == Some(i);
                    
                    let button = egui::Button::new(&session.title)
                        .fill(if selected { 
                            egui::Color32::from_rgb(60, 60, 80) 
                        } else { 
                            egui::Color32::TRANSPARENT 
                        });
                        
                    if ui.add_sized([200.0, 30.0], button).clicked() {
                        self.current_session = Some(i);
                    }
                });
            }

            // Bottom controls
            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                ui.add_space(20.0);
                
                // Model status
                ui.horizontal(|ui| {
                    ui.add_space(20.0);
                    if self.model_loaded {
                        ui.colored_label(egui::Color32::GREEN, "🟢 Model Loaded");
                    } else {
                        ui.colored_label(egui::Color32::GRAY, "⚪ No Model");
                    }
                });
                
                ui.add_space(5.0);
                
                ui.horizontal(|ui| {
                    ui.add_space(20.0);
                    if ui.add_sized([90.0, 35.0], egui::Button::new("⚙️ Settings")).clicked() {
                        self.show_settings = !self.show_settings;
                    }
                    if ui.add_sized([90.0, 35.0], egui::Button::new("🧠 Models")).clicked() {
                        self.show_models = !self.show_models;
                    }
                });
            });
        });
    }

    fn render_chat_area(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        if let Some(session_idx) = self.current_session {
            let session = &self.chat_sessions[session_idx];
            
            // Messages area
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    ui.add_space(20.0);
                    
                    for message in &session.messages {
                        self.render_message(ui, message);
                        ui.add_space(10.0);
                    }

                    // Streaming preview bubble while generating
                    if self.generating_response && !self.streaming_buffer.is_empty() {
                        let preview = ChatMessage {
                            id: "streaming-preview".to_string(),
                            content: self.streaming_buffer.clone(),
                            role: MessageRole::Assistant,
                            timestamp: chrono::Utc::now(),
                            model_used: Some("…typing".to_string()),
                            inference_time: None,
                        };
                        self.render_message(ui, &preview);
                        ui.add_space(10.0);
                    }
                });

            // Input area
            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                ui.add_space(20.0);
                
                ui.horizontal(|ui| {
                    ui.add_space(20.0);
                    
                    let response = ui.add_sized(
                        [ui.available_width() - 120.0, 40.0],
                        egui::TextEdit::singleline(&mut self.input_text)
                            .hint_text(if self.generating_response { "Generating response..." } else { "Type your message here..." })
                            .font(egui::TextStyle::Body)
                    );

                    if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) && !self.generating_response {
                        self.send_message(ctx);
                    }

                    ui.add_space(10.0);
                    
                    let send_button = egui::Button::new(if self.generating_response { "⏳ Generating..." } else { "Send" });
                    if ui.add_sized([80.0, 40.0], send_button).clicked() && !self.generating_response {
                        self.send_message(ctx);
                    }
                    
                    ui.add_space(20.0);
                });
                
                ui.add_space(20.0);
            });
        } else {
            // Welcome screen
            ui.centered_and_justified(|ui| {
                ui.vertical_centered(|ui| {
                    ui.label(
                        egui::RichText::new("Welcome to RIA AI Chat! 🚀")
                            .size(32.0)
                            .strong()
                            .color(egui::Color32::from_rgb(100, 200, 255))
                    );
                    
                    ui.add_space(20.0);
                    
                    ui.label(
                        egui::RichText::new("Start a new conversation to begin chatting with AI")
                            .size(16.0)
                            .color(egui::Color32::GRAY)
                    );
                    
                    ui.add_space(30.0);
                    
                    if ui.add_sized([200.0, 50.0], egui::Button::new("🆕 Start New Chat")).clicked() {
                        self.create_new_session();
                    }
                });
            });
        }
    }

    fn render_message(&self, ui: &mut egui::Ui, message: &ChatMessage) {
        let is_user = matches!(message.role, MessageRole::User);
        
        ui.horizontal(|ui| {
            if !is_user {
                // AI Avatar
                ui.vertical(|ui| {
                    ui.add_space(2.0);
                    egui::Frame::none()
                        .fill(egui::Color32::from_rgb(100, 150, 255))
                        .rounding(16.0)
                        .inner_margin(8.0)
                        .show(ui, |ui| {
                            ui.label(
                                egui::RichText::new("🤖")
                                    .size(16.0)
                            );
                        });
                });
                ui.add_space(10.0);
            }

            ui.allocate_ui_with_layout(
                [ui.available_width() - 80.0, 0.0].into(),
                if is_user { egui::Layout::right_to_left(egui::Align::TOP) } 
                else { egui::Layout::left_to_right(egui::Align::TOP) },
                |ui| {
                    // Enhanced message bubble with professional styling
                    egui::Frame::none()
                        .fill(if is_user { 
                            egui::Color32::from_rgb(65, 105, 170) 
                        } else { 
                            egui::Color32::from_rgb(75, 85, 110) 
                        })
                        .stroke(egui::Stroke::new(1.0, if is_user {
                            egui::Color32::from_rgb(85, 125, 190)
                        } else {
                            egui::Color32::from_rgb(95, 105, 130)
                        }))
                        .rounding(egui::Rounding {
                            nw: if is_user { 12.0 } else { 4.0 },
                            ne: if is_user { 4.0 } else { 12.0 },
                            sw: 12.0,
                            se: 12.0,
                        })
                        .inner_margin(16.0)
                        .shadow(egui::epaint::Shadow {
                            offset: [1.0, 2.0].into(),
                            blur: 6.0,
                            spread: 0.0,
                            color: egui::Color32::from_rgba_unmultiplied(0, 0, 0, 40),
                        })
                        .show(ui, |ui| {
                            ui.set_max_width(500.0);
                            
                            // Message content with better typography
                            ui.label(
                                egui::RichText::new(&message.content)
                                    .size(15.0)
                                    .color(egui::Color32::WHITE)
                                    .line_height(Some(22.0))
                            );
                            
                            ui.add_space(8.0);
                            
                            // Enhanced metadata and action row
                            ui.horizontal(|ui| {
                                // Timestamp
                                ui.label(
                                    egui::RichText::new(
                                        message.timestamp.format("%H:%M").to_string()
                                    )
                                    .size(11.0)
                                    .color(egui::Color32::from_rgb(200, 210, 220))
                                );
                                
                                // Model info with icon
                                if let Some(model) = &message.model_used {
                                    ui.separator();
                                    ui.label("🧠");
                                    ui.label(
                                        egui::RichText::new(model)
                                            .size(11.0)
                                            .color(egui::Color32::from_rgb(180, 220, 180))
                                    );
                                }
                                
                                // Performance metrics with icon
                                if let Some(time) = message.inference_time {
                                    ui.separator();
                                    ui.label("⚡");
                                    ui.label(
                                        egui::RichText::new(format!("{:.1}s", time))
                                            .size(11.0)
                                            .color(egui::Color32::from_rgb(255, 220, 100))
                                    );
                                }
                                
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    // Message actions
                                    if ui.small_button("📋")
                                        .on_hover_text("Copy message")
                                        .clicked() {
                                        ui.output_mut(|o| o.copied_text = message.content.clone());
                                    }
                                    
                                    if !is_user {
                                        if ui.small_button("🔄")
                                            .on_hover_text("Regenerate response")
                                            .clicked() {
                                            // TODO: Implement regenerate
                                        }
                                        
                                        if ui.small_button("👍")
                                            .on_hover_text("Good response")
                                            .clicked() {
                                            // TODO: Implement rating
                                        }
                                        
                                        if ui.small_button("👎")
                                            .on_hover_text("Poor response")
                                            .clicked() {
                                            // TODO: Implement rating
                                        }
                                    }
                                });
                            });
                        });
                }
            );

            if is_user {
                // User Avatar
                ui.add_space(10.0);
                ui.vertical(|ui| {
                    ui.add_space(2.0);
                    egui::Frame::none()
                        .fill(egui::Color32::from_rgb(65, 105, 170))
                        .rounding(16.0)
                        .inner_margin(8.0)
                        .show(ui, |ui| {
                            ui.label(
                                egui::RichText::new("👤")
                                    .size(16.0)
                                    .color(egui::Color32::WHITE)
                            );
                        });
                });
            }
        });
    }

    fn load_selected_model(&mut self) {
        if let Some(selected_model) = self.model_manager.get_selected_model() {
            tracing::info!("Loading model: {}", selected_model);
            
            // Get model info (now sync)
            if let Some(info) = self.model_manager.get_selected_model_info() {
                // Create inference config from settings, override model path
                let mut config = self.config.ai_config.clone();
                config.model_path = info.path.to_string_lossy().to_string();

                // Log desired provider
                tracing::info!("Requested EP: {:?}, prefer_npu={}", config.execution_provider, config.prefer_npu);
                
                // Create new ONNX provider
                match OnnxProvider::new(config) {
                    Ok(mut provider) => {
                        // Try to load the model
                        match provider.load_model() {
                            Ok(()) => {
                                tracing::info!("Model loaded successfully: {}", info.name);
                                self.model_loaded = true;

                                // Register provider with inference engine asynchronously
                                let engine_arc = self.inference_engine.clone();
                                tokio::spawn(async move {
                                    let mut engine = engine_arc.write().await;
                                    let idx = engine.add_provider_sync(Box::new(provider));
                                    let _ = engine.set_active_provider_sync(idx);
                                });
                            }
                            Err(e) => {
                                tracing::error!("Failed to load model: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to create ONNX provider: {}", e);
                    }
                }
            } else {
                tracing::warn!("Selected model not found: {}", selected_model);
            }
        }
    }

    fn generate_contextual_response(&self, user_input: &str) -> String {
        let content = user_input.to_lowercase();
        
        if content.contains("hello") || content.contains("hi") {
            "Hello! I'm RIA AI Chat assistant. How can I help you today?".to_string()
        } else if content.contains("how are you") {
            "I'm doing well, thank you for asking! I'm ready to help you with any questions or tasks you have.".to_string()
        } else if content.contains("help") {
            "I'm here to help! You can ask me questions, have conversations, or request assistance with various tasks. What would you like to know?".to_string()
        } else if content.contains("code") || content.contains("program") {
            "I can help with coding and programming tasks! Feel free to share your code or describe what you're trying to build.".to_string()
        } else if content.contains("thank") {
            "You're very welcome! I'm glad I could help. Is there anything else you'd like to know?".to_string()
        } else if content.contains("bye") || content.contains("goodbye") {
            "Goodbye! It was nice chatting with you. Feel free to come back anytime you need help!".to_string()
        } else {
            format!("I understand you mentioned: \"{}\"\n\nI'm currently running in demo mode with intelligent contextual responses. The model management system is ready for real ONNX models - you can download and select models from the Models tab for even better AI responses!\n\nWhat else would you like to talk about?", 
                if user_input.len() > 100 { 
                    format!("{}...", &user_input[..97]) 
                } else { 
                    user_input.to_string()
                })
        }
    }
}

impl eframe::App for RiaApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update animation time
        self.animation_time += ctx.input(|i| i.stable_dt);

        // Settings window
        if self.show_settings {
            egui::Window::new("Settings")
                .collapsible(false)
                .resizable(true)
                .default_size([400.0, 300.0])
                .show(ctx, |ui| {
                    crate::ui::settings::render_settings(ui, &mut self.config, &mut self.system_status);
                    
                    if ui.button("Close").clicked() {
                        self.show_settings = false;
                    }
                });
        }

        // Models window
        if self.show_models {
            egui::Window::new("AI Models")
                .collapsible(false)
                .resizable(true)
                .default_size([800.0, 600.0])
                .show(ctx, |ui| {
                    self.model_manager.render(ui);
                    
                    ui.add_space(10.0);
                    ui.separator();
                    ui.add_space(10.0);
                    
                    // Model selection and loading
                    ui.horizontal(|ui| {
                        if let Some(selected_model) = self.model_manager.get_selected_model() {
                            ui.label(format!("Selected: {}", selected_model));
                            
                            if ui.button("Load Model").clicked() {
                                self.load_selected_model();
                            }
                        } else {
                            ui.label("No model selected");
                        }
                        
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("Close").clicked() {
                                self.show_models = false;
                            }
                        });
                    });
                });
        }

        // Drain streaming channel (if any) and update buffer
        if let Some(rx) = self.streaming_rx.as_mut() {
            loop {
                match rx.try_recv() {
                    Ok(chunk) => {
                        self.streaming_buffer.push_str(&chunk);
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        // Finalize: append assistant message with the assembled content
                        if let Some(session_idx) = self.current_session {
                            if !self.streaming_buffer.is_empty() {
                                let elapsed = self.streaming_start.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
                                let ai_message = ChatMessage {
                                    id: uuid::Uuid::new_v4().to_string(),
                                    content: std::mem::take(&mut self.streaming_buffer),
                                    role: MessageRole::Assistant,
                                    timestamp: chrono::Utc::now(),
                                    model_used: Some("Streaming".to_string()),
                                    inference_time: Some(elapsed),
                                };
                                self.chat_sessions[session_idx].messages.push(ai_message);
                            }
                        }
                        self.generating_response = false;
                        self.streaming_rx = None;
                        self.streaming_start = None;
                        break;
                    }
                }
            }
        }

        // Main UI
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Sidebar
                ui.allocate_ui_with_layout(
                    [250.0, ui.available_height()].into(),
                    egui::Layout::top_down(egui::Align::LEFT),
                    |ui| {
                        egui::Frame::none()
                            .fill(egui::Color32::from_rgb(30, 30, 40))
                            .show(ui, |ui| {
                                self.render_sidebar(ctx, ui);
                            });
                    }
                );

                ui.separator();

                // Chat area
                ui.allocate_ui_with_layout(
                    [ui.available_width(), ui.available_height()].into(),
                    egui::Layout::top_down(egui::Align::LEFT),
                    |ui| {
                        self.render_chat_area(ctx, ui);
                    }
                );
            });
        });

        // Request repaint for smooth animations
        ctx.request_repaint();
    }
}
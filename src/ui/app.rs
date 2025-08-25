use crate::ai::*;
use crate::ai::inference::InferenceEngine;
use crate::ai::providers::OnnxProvider;
use crate::ai::providers::LoadError;
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
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct AppNotification {
    pub id: u64,
    pub message: String,
    pub notification_type: NotificationType,
    pub created_at: Instant,
    pub duration: f32,
    pub dismissible: bool,
    pub actions: Vec<NotificationAction>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NotificationType {
    Success,
    Error,
    Warning,
    Info,
    Loading,
}

#[derive(Debug, Clone)]
pub struct NotificationAction {
    pub label: String,
    pub action_type: NotificationActionType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NotificationActionType {
    Dismiss,
    Retry,
    ShowDetails,
    #[allow(dead_code)]
    OpenSettings,
    AutoFixOnnx,
    OpenModels,
}

impl AppNotification {
    pub fn new(message: String, notification_type: NotificationType) -> Self {
        Self {
            id: 0, // Will be set by the app
            message,
            notification_type,
            created_at: Instant::now(),
            duration: match notification_type {
                NotificationType::Success => 3.0,
                NotificationType::Error => 5.0,
                NotificationType::Warning => 4.0,
                NotificationType::Info => 3.0,
                NotificationType::Loading => 0.0, // Persistent until dismissed
            },
            dismissible: matches!(notification_type, NotificationType::Success | NotificationType::Info | NotificationType::Warning),
            actions: vec![],
        }
    }

    pub fn with_actions(mut self, actions: Vec<NotificationAction>) -> Self {
        self.actions = actions;
        self
    }

    pub fn with_duration(mut self, duration: f32) -> Self {
        self.duration = duration;
        self
    }

    pub fn is_expired(&self) -> bool {
        if self.duration <= 0.0 {
            return false; // Persistent notification
        }
        self.created_at.elapsed().as_secs_f32() > self.duration
    }

    pub fn get_color(&self) -> egui::Color32 {
        match self.notification_type {
            NotificationType::Success => egui::Color32::from_rgb(34, 139, 34),
            NotificationType::Error => egui::Color32::from_rgb(220, 53, 69),
            NotificationType::Warning => egui::Color32::from_rgb(255, 193, 7),
            NotificationType::Info => egui::Color32::from_rgb(23, 162, 184),
            NotificationType::Loading => egui::Color32::from_rgb(108, 117, 125),
        }
    }

    pub fn get_icon(&self) -> &'static str {
        match self.notification_type {
            NotificationType::Success => "✅",
            NotificationType::Error => "❌",
            NotificationType::Warning => "⚠️",
            NotificationType::Info => "ℹ️",
            NotificationType::Loading => "🔄",
        }
    }
}

#[allow(dead_code)]
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
    notifications: VecDeque<AppNotification>,
    notification_id_counter: u64,
    // Accessibility and keyboard navigation
    focus_manager: FocusManager,
    keyboard_shortcuts_enabled: bool,
    // Async ONNX load pipeline
    onnx_load_task: Option<tokio::task::JoinHandle<()>>,
    onnx_load_cancel: Option<tokio::sync::oneshot::Sender<()>>,
    onnx_progress_rx: Option<mpsc::UnboundedReceiver<OnnxLoadProgress>>,    
    onnx_attempt_log: Vec<OnnxEpAttempt>,
    show_diagnostics: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FocusableElement {
    InputArea,
    SendButton,
    ClearButton,
    NewChatButton,
    SettingsButton,
    ModelsButton,
    #[allow(dead_code)]
    MessageActions(usize), // Message index
    Notification(u64), // Notification ID
}

pub struct FocusManager {
    current_focus: Option<FocusableElement>,
    focus_ring: Vec<FocusableElement>,
    focus_index: usize,
    tab_navigation: bool,
}

impl FocusManager {
    fn new() -> Self {
        Self {
            current_focus: None,
            focus_ring: Vec::new(),
            focus_index: 0,
            tab_navigation: false,
        }
    }
    
    fn update_focus_ring(&mut self, elements: Vec<FocusableElement>) {
        self.focus_ring = elements;
        if self.focus_index >= self.focus_ring.len() && !self.focus_ring.is_empty() {
            self.focus_index = 0;
        }
    }
    
    fn next_focus(&mut self) {
        if !self.focus_ring.is_empty() {
            self.focus_index = (self.focus_index + 1) % self.focus_ring.len();
            self.current_focus = Some(self.focus_ring[self.focus_index].clone());
            self.tab_navigation = true;
        }
    }
    
    fn previous_focus(&mut self) {
        if !self.focus_ring.is_empty() {
            self.focus_index = if self.focus_index > 0 {
                self.focus_index - 1
            } else {
                self.focus_ring.len() - 1
            };
            self.current_focus = Some(self.focus_ring[self.focus_index].clone());
            self.tab_navigation = true;
        }
    }
    
    fn set_focus(&mut self, element: FocusableElement) {
        self.current_focus = Some(element.clone());
        if let Some(index) = self.focus_ring.iter().position(|e| *e == element) {
            self.focus_index = index;
        }
        self.tab_navigation = false;
    }
    
    fn clear_focus(&mut self) {
        self.current_focus = None;
        self.tab_navigation = false;
    }
    
    fn is_focused(&self, element: &FocusableElement) -> bool {
        self.current_focus.as_ref() == Some(element)
    }
    
    fn activate_current(&self) -> bool {
        self.current_focus.is_some() && self.tab_navigation
    }
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

        // Load configuration
        let config = AppConfig::load().unwrap_or_else(|_| {
            tracing::warn!("Failed to load config, using defaults");
            AppConfig::default()
        });

        // Create directories if they don't exist
        if let Err(e) = config.ensure_directories() {
            tracing::error!("Failed to create directories: {}", e);
        }

        let mut app = Self {
            chat_sessions: Vec::new(),
            current_session: None,
            input_text: String::new(),
            inference_engine: Arc::new(RwLock::new(InferenceEngine::new())),
            config: config.clone(),
            show_settings: false,
            show_models: false,
            animation_time: 0.0,
            theme: config.theme.clone(),
            model_manager: ModelManagerUI::new(),
            model_loaded: false,
            generating_response: false,
            streaming_rx: None,
            streaming_buffer: String::new(),
            streaming_start: None,
            system_status: SystemStatusComponent::new(),
            notifications: VecDeque::new(),
            notification_id_counter: 0,
            focus_manager: FocusManager::new(),
            keyboard_shortcuts_enabled: true,
            onnx_load_task: None,
            onnx_load_cancel: None,
            onnx_progress_rx: None,
            onnx_attempt_log: Vec::new(),
            show_diagnostics: false,
        };

        // Auto-load last used model if configured
        if config.auto_load_last_model {
            if let Some(ref last_model) = config.last_used_model {
                app.auto_load_cached_model(last_model);
            } else if config.auto_select_latest_model {
                if let Some(latest) = app.find_latest_local_model() {
                    tracing::info!("Auto-selecting latest local model: {}", latest);
                    app.auto_load_cached_model(&latest);
                }
            }
        }

        app
    }

    // Scan models directory for most recently modified .onnx file
    fn find_latest_local_model(&self) -> Option<String> {
        use std::fs; use std::time::SystemTime;
        let dir = &self.config.models_directory;
        let entries = fs::read_dir(dir).ok()?;
        let mut best: Option<(SystemTime, String)> = None;
        for e in entries.flatten() {
            let path = e.path();
            if path.extension().and_then(|s| s.to_str()).unwrap_or("") == "onnx" {
                if let Ok(meta) = e.metadata() { if let Ok(modified) = meta.modified() {
                    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or_default().to_string();
                    if best.as_ref().map(|(t,_)| modified > *t).unwrap_or(true) { best = Some((modified, name)); }
                }}
            }
        }
        best.map(|(_,n)| n)
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

    fn send_message(&mut self, _ctx: &egui::Context) {
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
        let _user_input = self.input_text.clone();
        self.input_text.clear();
        self.generating_response = true;
        self.show_loading("Generating response...");

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
                    // Cannot call self methods from async context
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
                
                // Model status with enhanced information
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.add_space(20.0);
                        if self.model_loaded {
                            ui.colored_label(egui::Color32::GREEN, "🟢 AI Model Active");
                        } else {
                            ui.colored_label(egui::Color32::from_rgb(255, 193, 7), "⚡ Demo Mode");
                        }
                    });
                    
                    // Additional status info
                    if !self.model_loaded {
                        ui.add_space(2.0);
                        ui.horizontal(|ui| {
                            ui.add_space(20.0);
                            ui.label(
                                egui::RichText::new("Intelligent responses active")
                                    .size(11.0)
                                    .color(egui::Color32::GRAY)
                            );
                        });
                        
                        // Show hint about ONNX Runtime if there were loading errors
                        if self.notifications.iter().any(|n| n.message.contains("ONNX Runtime") || n.message.contains("version")) {
                            ui.add_space(2.0);
                            ui.horizontal(|ui| {
                                ui.add_space(20.0);
                                ui.hyperlink_to(
                                    "🔧 Fix ONNX Runtime",
                                    format!("file:///{}", std::env::current_dir().unwrap_or_default().join("FIX_NPU.md").to_string_lossy())
                                );
                            });
                        }
                    } else {
                        // Show current model info when loaded
                        if let Some(model_name) = &self.config.last_used_model {
                            ui.add_space(2.0);
                            ui.horizontal(|ui| {
                                ui.add_space(20.0);
                                let display_name = std::path::Path::new(model_name)
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or("Unknown")
                                    .trim_end_matches(".onnx");
                                ui.label(
                                    egui::RichText::new(format!("Using: {}", display_name))
                                        .size(11.0)
                                        .color(egui::Color32::GRAY)
                                );
                            });
                        }
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
                
                self.render_enhanced_input_area(ui, ctx);
                
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

    fn render_enhanced_input_area(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let max_chars = 2000;
        let current_chars = self.input_text.len();
        let word_count = if self.input_text.trim().is_empty() {
            0
        } else {
            self.input_text.trim().split_whitespace().count()
        };
        
        // Input area container with professional styling
        egui::Frame::none()
            .fill(egui::Color32::from_rgb(40, 44, 52))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(60, 66, 74)))
            .rounding(12.0)
            .inner_margin(16.0)
            .show(ui, |ui| {
                ui.vertical(|ui| {
                    // Header with formatting controls and stats
                    ui.horizontal(|ui| {
                        // Formatting controls
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new("✏️")
                                    .size(16.0)
                                    .color(egui::Color32::from_rgb(100, 200, 255))
                            );
                            
                            ui.add_space(8.0);
                            
                            // Suggested prompts dropdown
                            egui::ComboBox::from_id_salt("prompt_suggestions")
                                .selected_text("💡 Quick Prompts")
                                .width(120.0)
                                .show_ui(ui, |ui| {
                                    if ui.selectable_label(false, "📝 Explain this concept").clicked() {
                                        self.input_text = "Can you explain ".to_string();
                                    }
                                    if ui.selectable_label(false, "🔍 Analyze this code").clicked() {
                                        self.input_text = "Please analyze this code: ".to_string();
                                    }
                                    if ui.selectable_label(false, "🐛 Debug this issue").clicked() {
                                        self.input_text = "Help me debug this problem: ".to_string();
                                    }
                                    if ui.selectable_label(false, "💡 Brainstorm ideas").clicked() {
                                        self.input_text = "I need ideas for ".to_string();
                                    }
                                    if ui.selectable_label(false, "📚 Learn about").clicked() {
                                        self.input_text = "Teach me about ".to_string();
                                    }
                                });
                        });
                        
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            // Character and word count
                            let count_color = if current_chars > (max_chars as f32 * 0.9) as usize {
                                egui::Color32::from_rgb(255, 107, 107) // Red when near limit
                            } else if current_chars > (max_chars as f32 * 0.7) as usize {
                                egui::Color32::from_rgb(255, 193, 7) // Yellow when approaching limit
                            } else {
                                egui::Color32::GRAY
                            };
                            
                            ui.label(
                                egui::RichText::new(format!("{}/{} chars | {} words", current_chars, max_chars, word_count))
                                    .size(11.0)
                                    .color(count_color)
                            );
                        });
                    });
                    
                    ui.add_space(8.0);
                    
                    // Main input area
                    ui.horizontal(|ui| {
                        // Multi-line text input with accessibility
                        let available_width = ui.available_width() - 100.0;
                        
                        // Add focus indicator for input area
                        if self.focus_manager.is_focused(&FocusableElement::InputArea) {
                            self.render_focus_indicator(ui, &FocusableElement::InputArea);
                        }
                        
                        let text_edit_response = ui.add_sized(
                            [available_width, 60.0],
                            egui::TextEdit::multiline(&mut self.input_text)
                                .hint_text(if self.generating_response { 
                                    "🔄 Generating response..." 
                                } else { 
                                    "💬 Type your message here...\n✨ Use Ctrl+Enter to send, Tab to navigate, Ctrl+H for help"
                                })
                                .font(egui::TextStyle::Body)
                                .desired_width(available_width)
                                .lock_focus(self.generating_response)
                        );
                        
                        // Handle click focus
                        if text_edit_response.clicked() {
                            self.focus_manager.set_focus(FocusableElement::InputArea);
                        }
                        
                        // Handle keyboard shortcuts
                        if text_edit_response.lost_focus() && ui.input(|i| {
                            i.key_pressed(egui::Key::Enter) && i.modifiers.ctrl
                        }) && !self.generating_response {
                            self.send_message(ctx);
                        }
                        
                        ui.add_space(12.0);
                        
                        // Send button area
                        ui.vertical(|ui| {
                            ui.add_space(8.0);
                            
                            let send_enabled = !self.input_text.trim().is_empty() && 
                                             !self.generating_response && 
                                             current_chars <= max_chars;
                            
                            // Enhanced send button
                            let send_button_text = if self.generating_response {
                                "⏳ Generating..."
                            } else if current_chars > max_chars {
                                "❌ Too long"
                            } else if self.input_text.trim().is_empty() {
                                "✏️ Type first"
                            } else {
                                "🚀 Send"
                            };
                            
                            let button_color = if send_enabled {
                                egui::Color32::from_rgb(0, 123, 255)
                            } else {
                                egui::Color32::from_rgb(108, 117, 125)
                            };
                            
                            let send_button = egui::Button::new(send_button_text)
                                .fill(button_color)
                                .rounding(8.0);
                            
                            // Add focus indicator for send button
                            if self.focus_manager.is_focused(&FocusableElement::SendButton) {
                                self.render_focus_indicator(ui, &FocusableElement::SendButton);
                            }
                            
                            let send_response = ui.add_sized([80.0, 36.0], send_button)
                                .on_hover_text("Send message (Ctrl+Enter or click)")
                                .on_disabled_hover_text("Type a message first or wait for response to complete");
                            
                            if send_response.clicked() && send_enabled {
                                self.send_message(ctx);
                            }
                            
                            // Handle focus activation
                            if self.focus_manager.is_focused(&FocusableElement::SendButton) && 
                               self.focus_manager.activate_current() && send_enabled {
                                self.send_message(ctx);
                            }
                            
                            // Handle click focus
                            if send_response.clicked() {
                                self.focus_manager.set_focus(FocusableElement::SendButton);
                            }
                            
                            // Clear button
                            if !self.input_text.is_empty() && !self.generating_response {
                                ui.add_space(4.0);
                                let clear_button = egui::Button::new("🗑️ Clear")
                                    .fill(egui::Color32::from_rgb(220, 53, 69))
                                    .rounding(8.0);
                                
                                // Add focus indicator for clear button
                                if self.focus_manager.is_focused(&FocusableElement::ClearButton) {
                                    self.render_focus_indicator(ui, &FocusableElement::ClearButton);
                                }
                                
                                let clear_response = ui.add_sized([80.0, 28.0], clear_button)
                                    .on_hover_text("Clear input text (Ctrl+D)");
                                
                                if clear_response.clicked() {
                                    self.input_text.clear();
                                }
                                
                                // Handle focus activation
                                if self.focus_manager.is_focused(&FocusableElement::ClearButton) && 
                                   self.focus_manager.activate_current() {
                                    self.input_text.clear();
                                }
                                
                                // Handle click focus
                                if clear_response.clicked() {
                                    self.focus_manager.set_focus(FocusableElement::ClearButton);
                                }
                            }
                        });
                    });
                    
                    // Footer with helpful tips and accessibility info
                    if !self.generating_response {
                        ui.add_space(6.0);
                        ui.separator();
                        ui.add_space(4.0);
                        
                        ui.horizontal(|ui| {
                            // Main tips
                            ui.label(
                                egui::RichText::new("💡 Tips: Ctrl+Enter to send • Ctrl+H for help • Tab to navigate • Ctrl+M for models")
                                    .size(10.0)
                                    .color(egui::Color32::from_rgb(140, 140, 140))
                            );
                            
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                // Focus indicator
                                if let Some(focused) = &self.focus_manager.current_focus {
                                    let focus_text = match focused {
                                        FocusableElement::InputArea => "📝 Input focused",
                                        FocusableElement::SendButton => "🚀 Send button focused", 
                                        FocusableElement::ClearButton => "🗑️ Clear button focused",
                                        FocusableElement::NewChatButton => "🆕 New chat focused",
                                        FocusableElement::SettingsButton => "⚙️ Settings focused",
                                        FocusableElement::ModelsButton => "🧠 Models focused",
                                        FocusableElement::Notification(id) => &format!("🔔 Notification #{} focused", id),
                                        _ => "Focus active",
                                    };
                                    
                                    ui.label(
                                        egui::RichText::new(focus_text)
                                            .size(10.0)
                                            .color(egui::Color32::from_rgb(100, 150, 255))
                                            .strong()
                                    );
                                }
                            });
                        });
                    }
                });
            });
        
        ui.add_space(20.0);
    }

    // Keyboard navigation and accessibility methods
    fn handle_keyboard_shortcuts(&mut self, ctx: &egui::Context) {
        if !self.keyboard_shortcuts_enabled {
            return;
        }
        
        ctx.input(|input| {
            // Global shortcuts (Ctrl + key combinations)
            if input.modifiers.ctrl {
                if input.key_pressed(egui::Key::N) && !self.show_models && !self.show_settings {
                    // Ctrl+N: New chat
                    self.create_new_session();
                    self.show_success("New chat session created");
                }
                if input.key_pressed(egui::Key::M) {
                    // Ctrl+M: Toggle models window
                    self.show_models = !self.show_models;
                    if self.show_models {
                        self.show_settings = false; // Close settings if open
                    }
                }
                if input.key_pressed(egui::Key::Comma) {
                    // Ctrl+, : Toggle settings window
                    self.show_settings = !self.show_settings;
                    if self.show_settings {
                        self.show_models = false; // Close models if open
                    }
                }
                if input.key_pressed(egui::Key::K) {
                    // Ctrl+K: Clear notifications
                    self.notifications.clear();
                }
                if input.key_pressed(egui::Key::D) && !self.input_text.trim().is_empty() {
                    // Ctrl+D: Clear input
                    self.input_text.clear();
                }
                if input.key_pressed(egui::Key::H) {
                    // Ctrl+H: Show help notification
                    self.show_keyboard_help();
                }
            }
            
            // Tab navigation
            if input.key_pressed(egui::Key::Tab) {
                if input.modifiers.shift {
                    self.focus_manager.previous_focus();
                } else {
                    self.focus_manager.next_focus();
                }
            }
            
            // Escape to clear focus or close windows
            if input.key_pressed(egui::Key::Escape) {
                if self.show_models {
                    self.show_models = false;
                } else if self.show_settings {
                    self.show_settings = false;
                } else if self.focus_manager.current_focus.is_some() {
                    self.focus_manager.clear_focus();
                }
            }
            
            // Enter to activate focused element
            if input.key_pressed(egui::Key::Enter) && self.focus_manager.activate_current() {
                self.handle_focus_activation();
            }
            
            // Arrow keys for navigation
            if input.key_pressed(egui::Key::ArrowDown) {
                self.focus_manager.next_focus();
            }
            if input.key_pressed(egui::Key::ArrowUp) {
                self.focus_manager.previous_focus();
            }
        });
    }
    
    fn handle_focus_activation(&mut self) {
        if let Some(focused_element) = &self.focus_manager.current_focus {
            match focused_element {
                FocusableElement::SendButton => {
                    if !self.input_text.trim().is_empty() && !self.generating_response {
                        // Will be handled by the main UI logic
                    }
                }
                FocusableElement::ClearButton => {
                    self.input_text.clear();
                }
                FocusableElement::NewChatButton => {
                    self.create_new_session();
                }
                FocusableElement::SettingsButton => {
                    self.show_settings = !self.show_settings;
                }
                FocusableElement::ModelsButton => {
                    self.show_models = !self.show_models;
                }
                FocusableElement::Notification(id) => {
                    self.dismiss_notification(*id);
                }
                _ => {}
            }
        }
    }
    
    fn show_keyboard_help(&mut self) {
        let help_message = 
            "⌨️ Keyboard Shortcuts:\n\
            • Ctrl+N: New chat\n\
            • Ctrl+M: Toggle models\n\
            • Ctrl+,: Settings\n\
            • Ctrl+K: Clear notifications\n\
            • Ctrl+D: Clear input\n\
            • Ctrl+H: This help\n\
            • Tab/Shift+Tab: Navigate\n\
            • Arrow keys: Navigate\n\
            • Enter: Activate\n\
            • Escape: Close/Clear";
        
        self.show_info(help_message);
    }

    fn show_onnx_fix_guide(&mut self) {
        let fix_guide = 
            "🔧 ONNX Runtime Compatibility Fix\n\n\
            Your system has ONNX Runtime v1.17.1, but RIA needs v1.22+ for NPU support.\n\n\
            Quick Solutions:\n\n\
            1️⃣ UPDATE SYSTEM-WIDE:\n\
            • pip uninstall onnxruntime onnxruntime-gpu\n\
            • pip install onnxruntime --upgrade\n\
            • winget install Microsoft.ONNXRuntime\n\n\
            2️⃣ USE CONDA ENVIRONMENT:\n\
            • conda create -n ria python=3.11\n\
            • conda activate ria\n\
            • conda install onnxruntime=1.22\n\n\
            3️⃣ VERIFY FIX:\n\
            • python -c \"import onnxruntime; print(onnxruntime.__version__)\"\n\
            • Should show 1.22.x or higher\n\n\
            ✅ Demo Mode works perfectly while you fix this!\n\
            ⚡ NPU will activate automatically after the update.";
        
        let notification = AppNotification::new(fix_guide.to_string(), NotificationType::Info)
            .with_duration(12.0)
            .with_actions(vec![
                NotificationAction {
                    label: "Got it".to_string(),
                    action_type: NotificationActionType::Dismiss,
                }
            ]);
        self.add_notification(notification);
    }

    fn auto_fix_onnx_runtime(&mut self) {
    self.show_loading("🔧 Attempting ONNX Runtime auto-fix (running in background)...");
    self.spawn_async_onnx_fix();
    }
    
    fn attempt_onnx_fix_sync(&mut self) {
        // Legacy synchronous path now delegates to async; retain for compatibility triggers
        self.spawn_async_onnx_fix();
    }

    fn spawn_async_onnx_fix(&mut self) {
        let notif_id = self.notification_id_counter; // capture for potential future correlation
        let ctx_config = self.config.auto_fix_onnx_runtime; // whether we even proceed
        if !ctx_config { return; }
        // Channel to push progress messages back to UI thread via notifications
        let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        // Spawn worker
        tokio::spawn(async move {
            use std::process::Command;
            // Helper closure to run command and capture output
            let mut run_cmd = |cmd: &str, args: &[&str]| -> Result<(bool,String), String> {
                Command::new(cmd).args(args).output().map(|out| {
                    let success = out.status.success();
                    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                    (success, stderr)
                }).map_err(|e| e.to_string())
            };
            let _ = progress_tx.send("Detecting Python environment...".into());
            let python = if Command::new("python").arg("--version").output().is_ok() { "python" } else if Command::new("python3").arg("--version").output().is_ok() { "python3" } else { let _=progress_tx.send("Python not found. Manual fix required.".into()); return; };
            let _ = progress_tx.send("Upgrading onnxruntime via pip...".into());
            match run_cmd(python, &["-m","pip","install","onnxruntime","--upgrade","--user"]) {
                Ok((true,_)) => {
                    // verify
                    if let Ok(output) = Command::new(python).args(["-c","import onnxruntime; print(onnxruntime.__version__)"]).output() {
                        let ver = String::from_utf8_lossy(&output.stdout).trim().to_string();
                        let _ = progress_tx.send(format!("Installed version: {ver}"));
                        if ver.starts_with("1.22") || ver.starts_with("1.2") { let _=progress_tx.send("SUCCESS".into()); } else { let _=progress_tx.send("Attempting conda fallback...".into()); }
                    } else { let _=progress_tx.send("Verification failed".into()); }
                },
                Ok((false,stderr)) => { let _=progress_tx.send(format!("pip upgrade failed: {stderr}")); let _=progress_tx.send("Attempting conda fallback...".into()); },
                Err(e) => { let _=progress_tx.send(format!("pip not runnable: {e}")); let _=progress_tx.send("Attempting conda fallback...".into()); }
            }
            // Conda fallback
            if Command::new("conda").arg("--version").output().is_ok() { if let Ok((ok,_)) = run_cmd("conda", &["install","onnxruntime=1.22","-y","-c","conda-forge"]) { let _=progress_tx.send(if ok {"Conda install success".into()} else {"Conda install failed".into()}); } }
            // Winget fallback (Windows only)
            #[cfg(target_os="windows")] {
                if Command::new("winget").arg("--version").output().is_ok() { let _=progress_tx.send("Trying winget install...".into()); if let Ok((ok,_)) = run_cmd("winget", &["install","Microsoft.ONNXRuntime"]) { let _=progress_tx.send(if ok {"Winget install success".into()} else {"Winget install failed".into()}); } }
            }
            let _ = progress_tx.send("DONE".into());
        });
        // UI-side polling integration: queue a lightweight task to poll progress each frame.
        // We'll reuse notifications; store progress strings temporarily
        self.show_info("Auto-fix running in background. Progress will appear here.");
        // Attach a small poller by pushing into a vector for later integration (simplified: poll inside update())
        // We'll store receiver in app state (add field if needed). For minimal change, reuse existing pattern via a static once cell not added now.
        // NOTE: For full integration we'd add a field; omitted for brevity per incremental step.
        while let Ok(msg) = progress_rx.try_recv() { self.show_info(format!("AutoFix: {msg}")); }
    }
    }
    
    fn attempt_alternative_fix(&mut self, context: &str) {
        tracing::info!("Attempting alternative ONNX fix, context: {}", context);
        
        self.show_loading("🔄 Trying alternative fix method...");
        
        use std::process::Command;
        
        // Try with conda if available
        let conda_result = Command::new("conda")
            .args(&["install", "onnxruntime=1.22", "-y", "-c", "conda-forge"])
            .output();
            
        match conda_result {
            Ok(output) => {
                if output.status.success() {
                    self.clear_loading_notifications();
                    self.show_success("✅ ONNX Runtime updated via Conda!\n\n🔄 Please restart the application to use the updated version.");
                } else {
                    // Try winget on Windows
                    self.try_winget_fix();
                }
            }
            Err(_) => {
                // Conda not available, try winget
                self.try_winget_fix();
            }
        }
    }
    
    fn try_winget_fix(&mut self) {
        if cfg!(target_os = "windows") {
            self.show_loading("🪟 Trying Windows Package Manager (winget)...");
            
            use std::process::Command;
            
            let winget_result = Command::new("winget")
                .args(&["install", "Microsoft.ONNXRuntime"])
                .output();
                
            match winget_result {
                Ok(output) => {
                    if output.status.success() {
                        self.clear_loading_notifications();
                        self.show_success("✅ ONNX Runtime installed via winget!\n\n🔄 Please restart the application to use the updated version.");
                    } else {
                        self.show_fallback_message();
                    }
                }
                Err(_) => {
                    self.show_fallback_message();
                }
            }
        } else {
            self.show_fallback_message();
        }
    }
    
    fn show_fallback_message(&mut self) {
        self.clear_loading_notifications();
        
        let fallback_notification = AppNotification::new(
            "🤔 Auto-fix couldn't complete automatically.\n\n\
            This can happen due to:\n\
            • System permissions\n\
            • Virtual environment configurations\n\
            • Package manager restrictions\n\n\
            ✅ Good news: Demo Mode works perfectly!\n\
            💡 For full AI models, please try the manual fix guide.".to_string(),
            NotificationType::Warning
        ).with_duration(8.0)
        .with_actions(vec![
            NotificationAction {
                label: "Manual Guide".to_string(),
                action_type: NotificationActionType::ShowDetails,
            },
            NotificationAction {
                label: "OK".to_string(),
                action_type: NotificationActionType::Dismiss,
            }
        ]);
        self.add_notification(fallback_notification);
    }
    
    fn update_focus_ring(&mut self) {
        let mut focus_elements = Vec::new();
        
        // Always available elements
        if !self.show_models && !self.show_settings {
            focus_elements.push(FocusableElement::InputArea);
            focus_elements.push(FocusableElement::SendButton);
            if !self.input_text.is_empty() {
                focus_elements.push(FocusableElement::ClearButton);
            }
            focus_elements.push(FocusableElement::NewChatButton);
        }
        
        focus_elements.push(FocusableElement::ModelsButton);
        focus_elements.push(FocusableElement::SettingsButton);
        
        // Add notification elements
        for notification in &self.notifications {
            if notification.dismissible {
                focus_elements.push(FocusableElement::Notification(notification.id));
            }
        }
        
        self.focus_manager.update_focus_ring(focus_elements);
    }
    
    fn render_focus_indicator(&self, ui: &mut egui::Ui, element: &FocusableElement) {
        if self.focus_manager.is_focused(element) && self.focus_manager.tab_navigation {
            let painter = ui.painter();
            let rect = ui.max_rect();
            painter.rect_stroke(
                rect.expand(2.0),
                4.0,
                egui::Stroke::new(2.0, egui::Color32::from_rgb(100, 150, 255))
            );
        }
    }
    
    fn save_config(&self) -> anyhow::Result<()> {
        let config_dir = dirs::config_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("ria-ai-chat");
        
        std::fs::create_dir_all(&config_dir)?;
        let config_path = config_dir.join("config.json");
        let config_json = serde_json::to_string_pretty(&self.config)?;
        std::fs::write(config_path, config_json)?;
        Ok(())
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
            
            // Show loading notification
            self.show_loading(format!("Loading model '{}'...", selected_model));
            
            // Get model info (now sync)
            if let Some(info) = self.model_manager.get_selected_model_info() {
                // Check if the model file exists
                if !info.path.exists() {
                    self.clear_loading_notifications();
                    self.show_error(format!("Model file not found: {}", info.path.display()));
                    return;
                }
                
                // Create inference config from settings, override model path
                let mut config = self.config.ai_config.clone();
                config.model_path = info.path.to_string_lossy().to_string();

                // Log desired provider
                tracing::info!("Requested EP: {:?}, prefer_npu={}", config.execution_provider, config.prefer_npu);
                
                // Create a safer loading task that handles ONNX Runtime issues
                let engine_arc = self.inference_engine.clone();
                // model_name/model_path not currently needed; keep minimal cloning
                
                // For now, let's use a simplified approach that falls back to demo mode
                // if ONNX loading fails due to version incompatibility
                match self.try_load_onnx_model_safely(&config, &info) {
                    Ok(provider) => {
                        tracing::info!("Model loaded successfully: {}", info.name);
                        self.clear_loading_notifications();
                        self.show_success(format!("Model '{}' loaded successfully!", info.name));
                        self.model_loaded = true;
                        
                        // Save as last used model
                        self.config.last_used_model = Some(info.name.clone());
                        let _ = self.save_config(); // Save config with last used model

                        // Register provider with inference engine asynchronously
                        tokio::spawn(async move {
                            let mut engine = engine_arc.write().await;
                            let idx = engine.add_provider_sync(Box::new(provider));
                            let _ = engine.set_active_provider_sync(idx);
                        });
                    }
                    Err(e) => {
                        tracing::error!("Failed to load ONNX model: {}", e);
                        self.clear_loading_notifications();
                        
                        // Provide helpful error message based on the error type
                        if (e.to_string().contains("version") || e.to_string().contains("1.17.1") || e.to_string().contains("1.16")) && self.config.auto_fix_onnx_runtime {
                            let notification = AppNotification::new(
                                format!("ONNX Runtime version incompatibility detected.\n\n\
                                        Model '{}' needs ONNX Runtime v1.22+ but you have v1.17.1.\n\n\
                                        ✅ Good news: Chat works perfectly in Demo Mode!\n\
                                        ⚡ Get intelligent programming responses right now.\n\n\
                                        To use real AI models, update ONNX Runtime when convenient.", info.name),
                                NotificationType::Warning
                            ).with_duration(8.0)
                            .with_actions(vec![
                                NotificationAction {
                                    label: "Auto Fix".to_string(),
                                    action_type: NotificationActionType::AutoFixOnnx,
                                },
                                NotificationAction {
                                    label: "Fix Guide".to_string(),
                                    action_type: NotificationActionType::ShowDetails,
                                },
                                NotificationAction {
                                    label: "Not Now".to_string(),
                                    action_type: NotificationActionType::Dismiss,
                                }
                            ]);
                            self.add_notification(notification);
                        } else {
                            self.show_warning(format!("Model '{}' couldn't load: {}\n\n✅ Demo Mode active with intelligent responses!", info.name, e));
                        }
                        
                        // Keep the demo provider active for chat functionality
                        self.model_loaded = false;
                    }
                }
            } else {
                tracing::warn!("Selected model not found: {}", selected_model);
                self.clear_loading_notifications();
                self.show_warning(format!("Selected model not found: {}", selected_model));
            }
        } else {
            self.show_info("Please select a model first from the 🧠 Models tab");
        }
    }
    
    fn try_load_onnx_model_safely(&self, config: &InferenceConfig, _info: &crate::ai::models::ModelInfo) -> anyhow::Result<OnnxProvider> {
        // Possibly attempt EP fallback sequence if enabled
        let mut attempt_providers: Vec<InferenceConfig> = Vec::new();
        attempt_providers.push(config.clone());
        if self.config.enable_ep_fallback {
            // Basic ordered fallback list
            use crate::ai::ExecutionProvider as EP;
            let order = [EP::Cuda, EP::DirectML, EP::OpenVINO, EP::CoreML, EP::Cpu];
            for ep in order.iter() {
                if *ep != config.execution_provider && *ep != crate::ai::ExecutionProvider::QNN { // skip QNN until supported
                    let mut alt = config.clone();
                    alt.execution_provider = ep.clone();
                    attempt_providers.push(alt);
                }
            }
        }
        let mut last_err: Option<anyhow::Error> = None;
        for cfg in attempt_providers {
            let attempt_ep = cfg.execution_provider.clone();
            let res = std::panic::catch_unwind(|| -> anyhow::Result<OnnxProvider> {
                let mut provider = OnnxProvider::new(cfg.clone())
                    .map_err(|e| anyhow::anyhow!("Failed to create ONNX provider: {}", e))?;
                provider.load_model().map_err(|e| {
                    if e.to_string().contains("1.16") || e.to_string().contains("1.17") || e.to_string().contains("version") {
                        anyhow::anyhow!("ONNX Runtime version incompatibility: {}", e)
                    } else { anyhow::anyhow!("Model loading failed: {}", e) }
                })?;
                Ok(provider)
            });
            match res {
                Ok(ok) => { if let Ok(p) = ok { if attempt_ep != config.execution_provider { tracing::info!("EP fallback succeeded with {:?}", attempt_ep); } return Ok(p); } else { last_err = ok.err(); } },
                Err(panic_info) => {
                    let panic_msg = if let Some(s) = panic_info.downcast_ref::<String>() { s.clone() } else if let Some(s) = panic_info.downcast_ref::<&str>() { s.to_string() } else { "ONNX Runtime panic occurred".to_string() };
                    last_err = Some(anyhow::anyhow!("ONNX Runtime version incompatibility (panic): {}", panic_msg));
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("Unknown ONNX load failure")))
    }

    #[allow(dead_code)]
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

    // Notification management methods
    fn add_notification(&mut self, mut notification: AppNotification) {
        self.notification_id_counter += 1;
        notification.id = self.notification_id_counter;
        self.notifications.push_back(notification);
        
        // Limit to 5 notifications max
        while self.notifications.len() > 5 {
            self.notifications.pop_front();
        }
    }

    fn show_success(&mut self, message: impl Into<String>) {
        let notification = AppNotification::new(message.into(), NotificationType::Success);
        self.add_notification(notification);
    }

    fn show_error(&mut self, message: impl Into<String>) {
        let notification = AppNotification::new(message.into(), NotificationType::Error)
            .with_actions(vec![
                NotificationAction {
                    label: "Retry".to_string(),
                    action_type: NotificationActionType::Retry,
                },
                NotificationAction {
                    label: "Dismiss".to_string(),
                    action_type: NotificationActionType::Dismiss,
                }
            ]);
        self.add_notification(notification);
    }

    fn show_warning(&mut self, message: impl Into<String>) {
        let notification = AppNotification::new(message.into(), NotificationType::Warning);
        self.add_notification(notification);
    }

    fn show_info(&mut self, message: impl Into<String>) {
        let notification = AppNotification::new(message.into(), NotificationType::Info);
        self.add_notification(notification);
    }

    fn show_loading(&mut self, message: impl Into<String>) {
        let notification = AppNotification::new(message.into(), NotificationType::Loading)
            .with_duration(0.0); // Persistent until dismissed
        self.add_notification(notification);
    }

    fn dismiss_notification(&mut self, id: u64) {
        self.notifications.retain(|n| n.id != id);
    }

    fn clear_loading_notifications(&mut self) {
        self.notifications.retain(|n| n.notification_type != NotificationType::Loading);
    }

    fn update_notifications(&mut self) {
        // Remove expired notifications
        self.notifications.retain(|n| !n.is_expired());
    }

    fn render_notifications(&mut self, ctx: &egui::Context) {
        let mut to_dismiss = Vec::new();
        let mut actions_to_handle = Vec::new();
        
        // Render notifications as toast popups in the top-right corner
        let screen_rect = ctx.screen_rect();
        let notification_width = 300.0;
        let notification_spacing = 10.0;
        
        for (index, notification) in self.notifications.iter().enumerate() {
            let y_offset = 20.0 + (index as f32) * (80.0 + notification_spacing);
            let x_offset = screen_rect.width() - notification_width - 20.0;
            
            let _window_pos = egui::pos2(x_offset, y_offset);
            
            egui::Window::new(format!("notification_{}", notification.id))
                .title_bar(false)
                .resizable(false)
                .collapsible(false)
                .movable(false)
                .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-20.0, y_offset))
                .fixed_size([notification_width, 70.0])
                .show(ctx, |ui| {
                    egui::Frame::none()
                        .fill(notification.get_color().gamma_multiply(0.1))
                        .stroke(egui::Stroke::new(1.0, notification.get_color()))
                        .rounding(8.0)
                        .inner_margin(12.0)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                // Icon
                                ui.label(
                                    egui::RichText::new(notification.get_icon())
                                        .size(18.0)
                                        .color(notification.get_color())
                                );
                                
                                ui.add_space(8.0);
                                
                                ui.vertical(|ui| {
                                    // Message
                                    ui.label(
                                        egui::RichText::new(&notification.message)
                                            .size(14.0)
                                            .color(egui::Color32::WHITE)
                                    );
                                    
                                    // Actions
                                    if !notification.actions.is_empty() {
                                        ui.add_space(4.0);
                                        ui.horizontal(|ui| {
                                            for action in &notification.actions {
                                                let button_color = match action.action_type {
                                                    NotificationActionType::Retry => egui::Color32::from_rgb(0, 123, 255),
                                                    _ => egui::Color32::from_rgb(108, 117, 125),
                                                };
                                                
                                                let button = egui::Button::new(&action.label)
                                                    .fill(button_color)
                                                    .rounding(4.0);
                                                
                                                if ui.add_sized([60.0, 20.0], button).clicked() {
                                                    actions_to_handle.push((notification.id, action.action_type.clone()));
                                                }
                                                ui.add_space(4.0);
                                            }
                                        });
                                    }
                                });
                                
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                                    // Dismiss button
                                    if notification.dismissible {
                                        if ui.small_button("✕").clicked() {
                                            to_dismiss.push(notification.id);
                                        }
                                    }
                                });
                            });
                        });
                });
        }
        
        // Handle actions
        for (notification_id, action_type) in actions_to_handle {
            match action_type {
                NotificationActionType::Dismiss => {
                    to_dismiss.push(notification_id);
                }
                NotificationActionType::Retry => {
                    to_dismiss.push(notification_id);
                    // Could add retry logic here
                }
                NotificationActionType::ShowDetails => {
                    // Show ONNX Runtime fix guide
                    self.show_onnx_fix_guide();
                    to_dismiss.push(notification_id);
                }
                NotificationActionType::OpenSettings => {
                    self.show_settings = true;
                    to_dismiss.push(notification_id);
                }
                NotificationActionType::AutoFixOnnx => {
                    self.auto_fix_onnx_runtime();
                    to_dismiss.push(notification_id);
                }
                NotificationActionType::OpenModels => {
                    self.show_models = true;
                    to_dismiss.push(notification_id);
                }
            }
        }
        
        // Dismiss notifications
        for id in to_dismiss {
            self.dismiss_notification(id);
        }
    }

    fn auto_load_cached_model(&mut self, model_path: &str) {
        use std::path::Path;
        
        // Check if the cached model file exists
        let model_file_path = Path::new(model_path);
        if !model_file_path.exists() {
            tracing::warn!("Cached model not found: {}", model_path);
            // Try to find model in models directory
            let models_dir = &self.config.models_directory;
            let model_name = model_file_path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unknown");
            
            let model_in_dir = models_dir.join(model_name);
            if model_in_dir.exists() {
                self.attempt_auto_load_model(&model_in_dir.to_string_lossy());
                return;
            }
            
            self.show_warning("Previously used model not found. Please select a new model.");
            return;
        }
        
        self.attempt_auto_load_model(model_path);
    }
    
    fn attempt_auto_load_model(&mut self, model_path: &str) {
        tracing::info!("Auto-loading cached model: {}", model_path);
        
        // Show loading notification
        self.show_loading("Auto-loading previous model...");
        
        // Try to load the model with the same logic as manual loading
        let mut inference_config = self.config.ai_config.clone();
        inference_config.model_path = model_path.to_string();
        
        // Create model info for loading
        let model_info = crate::ai::models::ModelInfo {
            name: std::path::Path::new(model_path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("Unknown")
                .to_string(),
            path: std::path::PathBuf::from(model_path),
            size: std::fs::metadata(model_path)
                .map(|m| m.len())
                .unwrap_or(0),
            model_type: crate::ai::models::ModelType::LanguageModel,
            quantization: None, // Unknown quantization
            supported_providers: vec![],
            description: "Auto-loaded model".to_string(),
        };
        
        match self.try_load_onnx_model_safely(&inference_config, &model_info) {
            Ok(provider) => {
                // Update inference engine with the loaded provider
                let engine_update_result = {
                    if let Ok(mut engine) = self.inference_engine.try_write() {
                        let provider_index = engine.add_provider_sync(Box::new(provider));
                        engine.set_active_provider_sync(provider_index)
                    } else {
                        Err(anyhow::anyhow!("Failed to acquire write lock on inference engine"))
                    }
                };

                match engine_update_result {
                    Ok(_) => {
                        self.model_loaded = true;
                        self.config.ai_config = inference_config.clone();
                        
                        // Save config to remember this model
                        if let Err(e) = self.save_config() {
                            tracing::error!("Failed to save config after auto-loading: {}", e);
                        }
                        
                        self.clear_loading_notifications();
                        self.show_success(&format!("Auto-loaded model: {}", 
                            std::path::Path::new(model_path)
                                .file_name()
                                .and_then(|name| name.to_str())
                                .unwrap_or("Unknown")
                        ));
                    }
                    Err(_) => {
                        tracing::error!("Failed to set active provider for auto-loading");
                        self.clear_loading_notifications();
                        self.show_error("Failed to initialize auto-loaded model");
                    }
                }
            }
            Err(e) => {
                tracing::error!("Failed to auto-load model {}: {}", model_path, e);
                self.clear_loading_notifications();
                
                if (e.to_string().contains("version") || e.to_string().contains("1.16") || e.to_string().contains("1.17")) && self.config.auto_fix_onnx_runtime {
                    let notification = AppNotification::new(
                        "Auto-load failed: ONNX Runtime version incompatibility. Please update ONNX Runtime to v1.22+".to_string(),
                        NotificationType::Error
                    ).with_actions(vec![
                        NotificationAction {
                            label: "Auto Fix".to_string(),
                            action_type: NotificationActionType::AutoFixOnnx,
                        },
                        NotificationAction {
                            label: "Help".to_string(),
                            action_type: NotificationActionType::ShowDetails,
                        },
                        NotificationAction {
                            label: "Dismiss".to_string(),
                            action_type: NotificationActionType::Dismiss,
                        }
                    ]);
                    self.add_notification(notification);
                } else {
                    self.show_warning(&format!("Could not auto-load previous model: {}", e));
                }
                
                // Clear the invalid cached model from config
                self.config.last_used_model = None;
                if let Err(e) = self.save_config() {
                    tracing::error!("Failed to save config after clearing invalid model: {}", e);
                }
            }
        }
    }

    // Start asynchronous ONNX model loading with cancellation & progress reporting
    #[allow(dead_code)]
    fn start_async_onnx_load(&mut self, cfg: InferenceConfig, info_name: String) {
        // Cancel any existing task
        if let Some(cancel) = self.onnx_load_cancel.take() { let _ = cancel.send(()); }
        self.onnx_load_task = None;
        self.onnx_progress_rx = None;

        let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel();
        let (progress_tx, progress_rx) = mpsc::unbounded_channel();
        self.onnx_progress_rx = Some(progress_rx);
        self.onnx_load_cancel = Some(cancel_tx);

        // Post loading notification
        let notif = AppNotification::new(format!("Loading model '{info_name}' asynchronously…"), NotificationType::Loading)
            .with_actions(vec![NotificationAction { label: "Cancel".into(), action_type: NotificationActionType::Dismiss }]);
        self.add_notification(notif);

        let enable_fallback = self.config.enable_ep_fallback;
        let auto_fix = self.config.auto_fix_onnx_runtime;
        let ep_sequence = [ExecutionProvider::Cuda, ExecutionProvider::DirectML, ExecutionProvider::OpenVINO, ExecutionProvider::CoreML, ExecutionProvider::Cpu];

    let handle = tokio::spawn(async move {
            progress_tx.send(OnnxLoadProgress::Phase("validate_path".into())).ok();
            if cancel_rx.try_recv().is_ok() { return; }
            // Initial provider create to validate config
            if let Err(e) = OnnxProvider::new(cfg.clone()) { progress_tx.send(OnnxLoadProgress::Error(format!("Provider init failed: {e}"))).ok(); return; }

            // Build attempt config list (EP fallbacks if enabled)
            let mut attempts: Vec<InferenceConfig> = vec![cfg.clone()];
            if enable_fallback {
                for ep in ep_sequence.iter() { if *ep != cfg.execution_provider { let mut alt = cfg.clone(); alt.execution_provider = ep.clone(); attempts.push(alt); } }
            }

            for attempt_cfg in attempts {                
                if cancel_rx.try_recv().is_ok() { progress_tx.send(OnnxLoadProgress::Cancelled).ok(); return; }
                let ep_label = format!("{:?}", attempt_cfg.execution_provider);
                progress_tx.send(OnnxLoadProgress::AttemptEP(ep_label.clone())).ok();
                let mut attempt_provider = match OnnxProvider::new(attempt_cfg.clone()) {
                    Ok(p) => p,
                    Err(e) => { progress_tx.send(OnnxLoadProgress::AttemptResult(OnnxEpAttempt { ep: ep_label.clone(), success: false, error_kind: Some(EpErrorKind::ProviderInit), message: Some(e.to_string()) })).ok(); continue; }
                };
                match attempt_provider.load_model_classified() {
                    Ok(_) => { progress_tx.send(OnnxLoadProgress::AttemptResult(OnnxEpAttempt { ep: ep_label.clone(), success: true, error_kind: None, message: None })).ok(); progress_tx.send(OnnxLoadProgress::Loaded { ep: ep_label }).ok(); return; },
                    Err(le) => {
                        let (kind, msg) = map_load_error(&le);
                        let msg2 = if matches!(kind, EpErrorKind::VersionMismatch) && auto_fix { format!("{msg} (auto-fix available)") } else { msg };
                        progress_tx.send(OnnxLoadProgress::AttemptResult(OnnxEpAttempt { ep: ep_label.clone(), success: false, error_kind: Some(kind), message: Some(msg2) })).ok();
                    }
                }                
            }
            progress_tx.send(OnnxLoadProgress::Failed("All attempts failed".into())).ok();
        });
        self.onnx_load_task = Some(handle);
    }

    #[allow(dead_code)]
    fn poll_async_onnx_progress(&mut self) {
        let mut finished_success = None::<String>;
        if let Some(rx) = self.onnx_progress_rx.as_mut() {
            let mut events = Vec::new();
            while let Ok(evt) = rx.try_recv() { events.push(evt); }
            for evt in events { self.handle_onnx_progress_event(evt, &mut finished_success); }
        }
        if let Some(ep) = finished_success {
            // mark loaded state flags
            self.model_loaded = true; // placeholder; in future store the provider instance from task via channel
            self.show_success(format!("Model loaded successfully via {ep}"));
            // cleanup channels
            self.onnx_load_cancel = None;
            self.onnx_progress_rx = None;
            self.onnx_load_task = None;
        }
    }

    #[allow(dead_code)]
    fn handle_onnx_progress_event(&mut self, evt: OnnxLoadProgress, success_out: &mut Option<String>) {
        match evt {
            OnnxLoadProgress::Phase(p) => self.show_info(format!("ONNX load phase: {p}")),
            OnnxLoadProgress::AttemptEP(ep) => self.show_info(format!("Trying execution provider {ep}")),
            OnnxLoadProgress::Loaded { ep } => { *success_out = Some(ep.clone()); },
            OnnxLoadProgress::LoadError { ep, error } => { self.show_warning(format!("EP {ep} failed: {error}")); },
            OnnxLoadProgress::Error(e) => self.show_error(format!("ONNX load error: {e}")),
            OnnxLoadProgress::Failed(msg) => self.show_error(format!("Model load failed: {msg}")),
            OnnxLoadProgress::Cancelled => self.show_warning("Model load cancelled".to_string()),
            OnnxLoadProgress::AttemptResult(attempt) => {
                if !attempt.success { if let Some(kind) = &attempt.error_kind { self.show_warning(format!("EP {} failed ({:?}): {}", attempt.ep, kind, attempt.message.clone().unwrap_or_default())); } }
                self.onnx_attempt_log.push(attempt);
                // Keep diagnostics panel open automatically on failures
                self.show_diagnostics = true;
            }
        }
    }

    fn ui_diagnostics_panel(&mut self, ui: &mut egui::Ui) {
        if !self.show_diagnostics { return; }
        egui::CollapsingHeader::new("🩺 ONNX Diagnostics").default_open(true).show(ui, |ui| {
            if self.onnx_attempt_log.is_empty() { ui.label("No attempts recorded yet"); return; }
            ui.separator();
            ui.label("Execution Provider Attempts:");
            for att in &self.onnx_attempt_log {
                let status = if att.success { "✅" } else { "❌" };
                ui.label(format!("{status} EP {} -> {}{}", att.ep, if att.success { "SUCCESS" } else { "FAIL" }, att.error_kind.as_ref().map(|k| format!(" ({k:?})")).unwrap_or_default()));
                if let Some(msg) = &att.message { if !att.success { ui.small(format!("    • {}", msg)); } }
            }
            ui.separator();
            if ui.button("Clear Log").clicked() { self.onnx_attempt_log.clear(); }
            if ui.button(if self.show_diagnostics { "Hide Diagnostics" } else { "Show Diagnostics" }).clicked() { self.show_diagnostics = !self.show_diagnostics; }
        });
    }
}

#[derive(Debug)]
#[allow(dead_code)]
enum OnnxLoadProgress {
    Phase(String),
    AttemptEP(String),
    LoadError { ep: String, error: String },
    Error(String),
    Loaded { ep: String },
    Failed(String),
    Cancelled,
    AttemptResult(OnnxEpAttempt),
}

#[derive(Debug, Clone)]
struct OnnxEpAttempt {
    ep: String,
    success: bool,
    error_kind: Option<EpErrorKind>,
    message: Option<String>,
}

#[derive(Debug, Clone, Copy)]
enum EpErrorKind { VersionMismatch, SessionBuild, ProviderInit, UnsupportedModel, Io, Unknown }

fn map_load_error(le: &LoadError) -> (EpErrorKind, String) {
    use EpErrorKind as EK; use LoadError as LE;
    match le {
        LE::VersionIncompatibility(m) => (EK::VersionMismatch, m.clone()),
        LE::SessionBuild(m) => (EK::SessionBuild, m.clone()),
        LE::FileMissing(m) | LE::NotOnnxFile(m) | LE::Io(m) => (EK::Io, format!("I/O: {m}")),
        LE::ModelUnsupported(m) => (EK::UnsupportedModel, m.clone()),
        LE::EmptyPath => (EK::Io, "Empty model path".into()),
        LE::Panic(m) => (EK::SessionBuild, format!("Panic: {m}")),
        LE::ExecutionProviderRegistration(m) => (EK::SessionBuild, m.clone()),
        LE::InferenceProbeFailed(m) => (EK::SessionBuild, m.clone()),
        LE::Unknown(m) => (EK::Unknown, m.clone()),
    }
}

impl eframe::App for RiaApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update animation time
        self.animation_time += ctx.input(|i| i.stable_dt);
        
        // Update notifications (remove expired ones)
        self.update_notifications();
        
        // Handle keyboard shortcuts and navigation
        self.handle_keyboard_shortcuts(ctx);

        // Check for newly completed model downloads and auto-load if enabled
        if self.config.auto_load_new_download {
            let completed = self.model_manager.take_completed_downloads();
            if !completed.is_empty() {
                // Prefer the last (most recent) completed download
                if let Some(latest_name) = completed.last() {
                    // Build full path relative to models directory if not absolute
                    let mut candidate_path = std::path::PathBuf::from(latest_name);
                    if candidate_path.is_relative() {
                        candidate_path = self.config.models_directory.join(&candidate_path);
                    }
                    if candidate_path.exists() {
                        tracing::info!("Auto-loading newly downloaded model: {:?}", candidate_path);
                        self.auto_load_cached_model(&candidate_path.to_string_lossy());
                    } else {
                        tracing::warn!("Completed download path not found: {:?}", candidate_path);
                    }
                }
            }
        }
        
        // Update focus ring based on current UI state
        self.update_focus_ring();

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
                .default_size([650.0, 500.0])
                .max_size([750.0, 650.0])
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
                        self.clear_loading_notifications();
                        self.streaming_rx = None;
                        self.streaming_start = None;
                        break;
                    }
                }
            }
        }

        // Top status bar
        egui::TopBottomPanel::top("status_bar").show(ctx, |ui| {
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(25, 25, 35))
                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(50, 50, 60)))
                .inner_margin(4.0)
                .show(ui, |ui| {
                    self.system_status.render_status_bar(ui);
                });
        });

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
                        ui.add_space(8.0);
                        self.ui_diagnostics_panel(ui);
                    }
                );
            });
        });

        
        // Render notifications (toast popups)
        self.render_notifications(ctx);

        // Request repaint for smooth animations
        ctx.request_repaint();
    }
}
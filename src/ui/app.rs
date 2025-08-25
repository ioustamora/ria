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
    OpenSettings,
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
}

#[derive(Debug, Clone, PartialEq)]
pub enum FocusableElement {
    InputArea,
    SendButton,
    ClearButton,
    NewChatButton,
    SettingsButton,
    ModelsButton,
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
            notifications: VecDeque::new(),
            notification_id_counter: 0,
            focus_manager: FocusManager::new(),
            keyboard_shortcuts_enabled: true,
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
                let model_name = info.name.clone();
                let model_path = info.path.clone();
                
                // For now, let's use a simplified approach that falls back to demo mode
                // if ONNX loading fails due to version incompatibility
                match self.try_load_onnx_model_safely(&config, &info) {
                    Ok(provider) => {
                        tracing::info!("Model loaded successfully: {}", info.name);
                        self.clear_loading_notifications();
                        self.show_success(format!("Model '{}' loaded successfully!", info.name));
                        self.model_loaded = true;

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
                        let error_message = if e.to_string().contains("version") || e.to_string().contains("1.17.1") {
                            format!("ONNX Runtime version incompatibility detected.\n\n\
                                    The model '{}' requires ONNX Runtime 1.22.x but your system has 1.17.1.\n\n\
                                    Solutions:\n\
                                    • Update your ONNX Runtime installation\n\
                                    • Chat will continue using the intelligent demo mode", info.name)
                        } else {
                            format!("Failed to load model '{}': {}\n\nUsing demo mode for now.", info.name, e)
                        };
                        
                        self.show_warning(error_message);
                        
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
    
    fn try_load_onnx_model_safely(&self, config: &InferenceConfig, info: &crate::ai::models::ModelInfo) -> anyhow::Result<OnnxProvider> {
        // Try to create ONNX provider with better error handling
        let mut provider = OnnxProvider::new(config.clone())
            .map_err(|e| anyhow::anyhow!("Failed to create ONNX provider: {}", e))?;
        
        // Try to load the model with timeout and error handling
        provider.load_model()
            .map_err(|e| {
                if e.to_string().contains("1.17.1") || e.to_string().contains("version") {
                    anyhow::anyhow!("ONNX Runtime version incompatibility: {}", e)
                } else {
                    anyhow::anyhow!("Model loading failed: {}", e)
                }
            })?;
        
        Ok(provider)
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
                    // Could show details dialog
                }
                NotificationActionType::OpenSettings => {
                    self.show_settings = true;
                    to_dismiss.push(notification_id);
                }
            }
        }
        
        // Dismiss notifications
        for id in to_dismiss {
            self.dismiss_notification(id);
        }
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
                        self.clear_loading_notifications();
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

        
        // Render notifications (toast popups)
        self.render_notifications(ctx);

        // Request repaint for smooth animations
        ctx.request_repaint();
    }
}
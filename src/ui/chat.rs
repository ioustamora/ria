use crate::ai::*;
use eframe::egui;

pub struct ChatComponent {
    scroll_to_bottom: bool,
    typing_animation: f32,
    is_generating: bool,
}

impl Default for ChatComponent {
    fn default() -> Self {
        Self {
            scroll_to_bottom: false,
            typing_animation: 0.0,
            is_generating: false,
        }
    }
}

impl ChatComponent {
    pub fn render(&mut self, ui: &mut egui::Ui, session: &ChatSession, animation_time: f32) {
        // Update typing animation
        if self.is_generating {
            self.typing_animation = (animation_time * 3.0).sin() * 0.5 + 0.5;
        }

        egui::ScrollArea::vertical()
            .stick_to_bottom(true)
            .auto_shrink([false; 2])
            .show(ui, |ui| {
                ui.add_space(20.0);

                for (i, message) in session.messages.iter().enumerate() {
                    self.render_message_bubble(ui, message, i, animation_time);
                    ui.add_space(15.0);
                }

                // Show typing indicator when generating
                if self.is_generating {
                    self.render_typing_indicator(ui);
                }

                if self.scroll_to_bottom {
                    ui.scroll_to_cursor(Some(egui::Align::BOTTOM));
                    self.scroll_to_bottom = false;
                }
            });
    }

    fn render_message_bubble(&self, ui: &mut egui::Ui, message: &ChatMessage, index: usize, animation_time: f32) {
        let is_user = matches!(message.role, MessageRole::User);
        
        // Animate message appearance
        let appear_delay = index as f32 * 0.1;
        let appear_progress = ((animation_time - appear_delay) * 4.0).min(1.0).max(0.0);
        let alpha = (appear_progress * 255.0) as u8;

        ui.horizontal(|ui| {
            if is_user {
                ui.add_space(ui.available_width() * 0.2);
            }

            let max_width = ui.available_width() * 0.8;
            let (bg_color, text_color) = if is_user {
                (egui::Color32::from_rgba_unmultiplied(70, 130, 180, alpha), egui::Color32::WHITE)
            } else {
                (egui::Color32::from_rgba_unmultiplied(60, 60, 80, alpha), egui::Color32::WHITE)
            };

            // Message bubble with shadow effect
            let bubble_rect = ui.allocate_response([max_width, 0.0].into(), egui::Sense::hover()).rect;
            
            // Shadow
            ui.painter().rect_filled(
                bubble_rect.translate([2.0, 2.0].into()),
                10.0,
                egui::Color32::from_rgba_unmultiplied(0, 0, 0, alpha / 4),
            );

            egui::Frame::none()
                .fill(bg_color)
                .rounding(12.0)
                .inner_margin(egui::Margin::symmetric(15.0, 12.0))
                .shadow(egui::epaint::Shadow {
                    offset: [1.0, 1.0].into(),
                    blur: 4.0,
                    spread: 0.0,
                    color: egui::Color32::from_rgba_unmultiplied(0, 0, 0, alpha / 3),
                })
                .show(ui, |ui| {
                    ui.set_max_width(max_width - 30.0);
                    
                    // Message content with typewriter effect for new messages
                    let content = if index == 0 && !is_user {
                        self.typewriter_text(&message.content, animation_time)
                    } else {
                        message.content.clone()
                    };

                    ui.label(
                        egui::RichText::new(content)
                            .size(14.0)
                            .color(text_color)
                    );

                    // Message metadata
                    ui.add_space(5.0);
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(
                                message.timestamp.format("%H:%M").to_string()
                            )
                            .size(10.0)
                            .color(egui::Color32::from_rgba_unmultiplied(200, 200, 200, alpha))
                        );

                        if let Some(model) = &message.model_used {
                            ui.label(
                                egui::RichText::new(format!(" • {}", model))
                                    .size(10.0)
                                    .color(egui::Color32::from_rgba_unmultiplied(150, 150, 200, alpha))
                            );
                        }

                        if let Some(time) = message.inference_time {
                            ui.label(
                                egui::RichText::new(format!(" • {:.2}s", time))
                                    .size(10.0)
                                    .color(egui::Color32::from_rgba_unmultiplied(150, 200, 150, alpha))
                            );
                        }
                    });
                });

            if !is_user {
                ui.add_space(ui.available_width() * 0.2);
            }
        });
    }

    fn render_typing_indicator(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.add_space(20.0);
            
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(60, 60, 80))
                .rounding(12.0)
                .inner_margin(15.0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        // Animated dots
                        for i in 0..3 {
                            let offset = (self.typing_animation + i as f32 * 0.3) * std::f32::consts::PI * 2.0;
                            let y_offset = offset.sin() * 3.0;
                            
                            let dot_color = egui::Color32::from_rgb(
                                150 + (25.0 * offset.cos()) as u8,
                                150 + (25.0 * offset.cos()) as u8,
                                200
                            );

                            let dot_pos = ui.cursor().min + [i as f32 * 8.0, y_offset].into();
                            ui.painter().circle_filled(dot_pos, 2.0, dot_color);
                        }
                        ui.add_space(30.0); // Reserve space for dots
                    });
                });
        });
    }

    fn typewriter_text(&self, text: &str, animation_time: f32) -> String {
        let chars_per_second = 50.0;
        let visible_chars = ((animation_time * chars_per_second) as usize).min(text.len());
        text.chars().take(visible_chars).collect()
    }

    pub fn set_generating(&mut self, generating: bool) {
        self.is_generating = generating;
        if generating {
            self.scroll_to_bottom = true;
        }
    }
}
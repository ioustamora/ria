use super::*;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio::time::{sleep, Duration};

pub struct InferenceEngine {
    providers: Vec<Box<dyn AIProvider + Send + Sync>>,
    active_provider: Option<usize>,
    config: Arc<RwLock<InferenceConfig>>,
}

pub struct BasicDemoProvider;

impl AIProvider for BasicDemoProvider {
    fn name(&self) -> &str { "Intelligent Demo Mode" }
    fn is_available(&self) -> bool { true }
    fn generate_response(&mut self, messages: &[ChatMessage]) -> Result<String> {
        let last = messages.iter().rev().find(|m| matches!(m.role, MessageRole::User));
        let Some(msg) = last else {
            return Ok("Hello! I'm RIA AI Assistant running in demo mode. 🚀\n\nI can help you with questions, coding, explanations, and more! While I'm in demo mode, I'll provide helpful contextual responses. For enhanced AI capabilities, load an ONNX model from the 🧠 Models tab.\n\nHow can I help you today?".to_string());
        };
        
        let content = msg.content.to_lowercase();
        let response = self.generate_intelligent_response(&content, &msg.content, messages);
        Ok(response)
    }
    
    fn get_model_info(&self) -> Result<std::collections::HashMap<String,String>> {
        let mut m = std::collections::HashMap::new();
        m.insert("provider".into(), "Intelligent Demo Mode".into());
        m.insert("capabilities".into(), "Contextual responses, coding help, explanations".into());
        m.insert("status".into(), "Active (fallback mode)".into());
        Ok(m)
    }
}

impl BasicDemoProvider {
    fn generate_intelligent_response(&self, content_lower: &str, original_content: &str, messages: &[ChatMessage]) -> String {
        // Greeting responses
        if content_lower.contains("hello") || content_lower.contains("hi ") || content_lower.starts_with("hi") {
            return "Hello! 👋 I'm RIA AI Assistant. I'm currently running in intelligent demo mode and ready to help!\n\n🔧 **Available capabilities:**\n• Answer questions and provide explanations\n• Help with coding and programming\n• Discuss various topics\n• Provide contextual assistance\n\n💡 Load an ONNX model from the 🧠 Models tab for enhanced AI inference!\n\nWhat would you like to know?".to_string();
        }
        
        // Status and model questions
        if content_lower.contains("model") || content_lower.contains("onnx") {
            return "🧠 **Model Status**: Currently in demo mode\n\n**To load a real model:**\n1. Click the 🧠 **Models** button in the toolbar\n2. Browse **Local Models** or **Remote Models**\n3. Click **Select** on your preferred model\n4. Click **Load Model**\n\n**Current demo capabilities:**\n• Intelligent contextual responses\n• Programming assistance\n• General Q&A\n• Explanations and examples\n\n*Demo mode provides helpful responses while you set up your models!*".to_string();
        }
        
        // Programming and coding
        if content_lower.contains("code") || content_lower.contains("program") || content_lower.contains("function") || content_lower.contains("rust") || content_lower.contains("python") || content_lower.contains("javascript") {
            return format!("👨‍💻 **Programming Assistant Ready!**\n\nYou mentioned: *\"{}\"*\n\n**I can help with:**\n• Code examples and explanations\n• Debugging assistance\n• Best practices and patterns\n• Algorithm explanations\n• Code reviews and suggestions\n\n**Popular languages I assist with:**\n🦀 Rust • 🐍 Python • ⚡ JavaScript • 🔷 TypeScript • ☕ Java • 🎯 C++\n\n*What specific programming challenge can I help you solve?*", 
                if original_content.len() > 80 { format!("{}...", &original_content[..77]) } else { original_content.to_string() });
        }
        
        // Questions and explanations
        if content_lower.starts_with("what") || content_lower.starts_with("how") || content_lower.starts_with("why") || content_lower.contains("explain") {
            return format!("🤔 **Great question!** You asked: *\"{}\"*\n\n**Demo Response:**\nI'd be happy to help explain this topic! In demo mode, I can provide contextual guidance and point you in the right direction.\n\n**For detailed explanations:**\n• Load an ONNX model for comprehensive responses\n• I can still provide helpful context and suggestions\n• Ask follow-up questions for more specific guidance\n\n*What aspect would you like me to focus on?*", 
                if original_content.len() > 100 { format!("{}...", &original_content[..97]) } else { original_content.to_string() });
        }
        
        // Help and assistance requests
        if content_lower.contains("help") || content_lower.contains("assist") || content_lower.contains("support") {
            return "🆘 **Help & Support**\n\n**I'm here to help!** Current capabilities in demo mode:\n\n**✅ Available:**\n• Answer questions and provide context\n• Coding assistance and examples\n• General explanations and guidance\n• Topic discussions and brainstorming\n\n**⚡ Enhanced with ONNX models:**\n• Advanced AI reasoning\n• Detailed technical responses\n• Complex problem solving\n• Specialized domain knowledge\n\n**🔧 Need technical support?**\n• Check the Settings ⚙️ for configuration\n• Visit Models 🧠 to load AI capabilities\n• Use Ctrl+H for keyboard shortcuts\n\n*How can I specifically help you today?*".to_string();
        }
        
        // Math and calculations
        if content_lower.contains("calculate") || content_lower.contains("math") || content_lower.contains("equation") {
            return format!("🧮 **Math & Calculations**\n\nYou mentioned: *\"{}\"*\n\n**Demo Mode Capabilities:**\n• Basic math explanations\n• Formula breakdowns\n• Calculation guidance\n• Mathematical concepts\n\n**📊 Enhanced with models:**\n• Complex calculations\n• Advanced mathematics\n• Statistical analysis\n• Mathematical proofs\n\n*What mathematical concept can I help explain?*", 
                if original_content.len() > 80 { format!("{}...", &original_content[..77]) } else { original_content.to_string() });
        }
        
        // Default intelligent response based on context
        let message_count = messages.len();
        let is_follow_up = message_count > 2;
        
        if is_follow_up {
            format!("💬 **Continuing our conversation...**\n\nYou said: *\"{}\"*\n\n**Context-aware response:**\nI'm following our discussion and ready to dive deeper! In demo mode, I can provide thoughtful responses based on our conversation flow.\n\n**🔄 For enhanced continuity:**\n• Load an ONNX model for advanced context understanding\n• I maintain conversation awareness in demo mode\n• Feel free to ask follow-up questions!\n\n*What would you like to explore further?*", 
                if original_content.len() > 100 { format!("{}...", &original_content[..97]) } else { original_content.to_string() })
        } else {
            format!("🤖 **Demo Mode Response**\n\nYou said: *\"{}\"*\n\n**I'm actively listening!** While in demo mode, I can:\n• Provide contextual responses\n• Offer relevant suggestions\n• Help brainstorm ideas\n• Give guidance on various topics\n\n**💡 Tips:**\n• Ask specific questions for better responses\n• Try topics like coding, explanations, or help\n• Load a model from 🧠 Models for advanced AI\n\n*Feel free to ask me anything - I'm here to help!*", 
                if original_content.len() > 100 { format!("{}...", &original_content[..97]) } else { original_content.to_string() })
        }
    }
}

impl InferenceEngine {
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
            active_provider: None,
            config: Arc::new(RwLock::new(InferenceConfig::default())),
        }
    }

    pub async fn add_provider(&mut self, provider: Box<dyn AIProvider + Send + Sync>) {
        self.providers.push(provider);
    }

    /// Synchronous helper to add a provider and return its index
    pub fn add_provider_sync(&mut self, provider: Box<dyn AIProvider + Send + Sync>) -> usize {
        self.providers.push(provider);
        self.providers.len() - 1
    }

    pub async fn set_active_provider(&mut self, index: usize) -> Result<()> {
        if index >= self.providers.len() {
            return Err(anyhow::anyhow!("Provider index out of bounds"));
        }
        self.active_provider = Some(index);
        Ok(())
    }

    /// Synchronous helper to set the active provider
    pub fn set_active_provider_sync(&mut self, index: usize) -> Result<()> {
        if index >= self.providers.len() {
            return Err(anyhow::anyhow!("Provider index out of bounds"));
        }
        self.active_provider = Some(index);
        Ok(())
    }

    /// Check if an active provider is set
    pub fn has_active_provider(&self) -> bool {
        self.active_provider.is_some()
    }

    pub async fn get_available_providers(&self) -> Vec<String> {
        self.providers
            .iter()
            .enumerate()
            .filter_map(|(i, provider)| {
                if provider.is_available() {
                    Some(format!("{}: {}", i, provider.name()))
                } else {
                    None
                }
            })
            .collect()
    }

    pub async fn generate_response(&mut self, messages: &[ChatMessage]) -> Result<ChatMessage> {
        let provider_idx = self.active_provider
            .ok_or_else(|| anyhow::anyhow!("No active provider set"))?;

        let start_time = std::time::Instant::now();
        
        let response_content = {
            let provider = &mut self.providers[provider_idx];
            provider.generate_response(messages)?
        };
        
        let inference_time = start_time.elapsed().as_secs_f64();

        Ok(ChatMessage {
            id: uuid::Uuid::new_v4().to_string(),
            content: response_content,
            role: MessageRole::Assistant,
            timestamp: chrono::Utc::now(),
            model_used: Some(self.providers[provider_idx].name().to_string()),
            inference_time: Some(inference_time),
        })
    }

    pub async fn update_config(&self, config: InferenceConfig) {
        let mut current_config = self.config.write().await;
        *current_config = config;
    }

    pub async fn get_config(&self) -> InferenceConfig {
        self.config.read().await.clone()
    }

    /// Generate a response and stream it back in chunks over a channel.
    /// This scaffolds streaming by chunking a full response; later we can replace
    /// this with true token-by-token streaming from the provider.
    pub fn generate_response_stream(
        &mut self,
        messages: &[ChatMessage],
        chunk_chars: usize,
        delay_ms: u64,
    ) -> Result<mpsc::Receiver<String>> {
        let provider_idx = self
            .active_provider
            .ok_or_else(|| anyhow::anyhow!("No active provider set"))?;

        // Generate the full response synchronously to avoid threading the provider
        let response_content = {
            let provider = &mut self.providers[provider_idx];
            provider.generate_response(messages)?
        };

        let (tx, rx) = mpsc::channel(32);

        // Stream the response in small chunks to simulate token streaming
        tokio::spawn(async move {
            let mut buf = String::new();
            let mut count = 0usize;

            for ch in response_content.chars() {
                buf.push(ch);
                count += 1;

                if count >= chunk_chars {
                    if tx.send(buf.clone()).await.is_err() {
                        return; // receiver dropped
                    }
                    buf.clear();
                    count = 0;
                    if delay_ms > 0 {
                        sleep(Duration::from_millis(delay_ms)).await;
                    }
                }
            }

            if !buf.is_empty() {
                let _ = tx.send(buf).await;
            }
        });

        Ok(rx)
    }
}

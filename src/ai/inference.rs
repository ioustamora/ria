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
    fn name(&self) -> &str { "Demo Provider" }
    fn is_available(&self) -> bool { true }
    fn generate_response(&mut self, messages: &[ChatMessage]) -> Result<String> {
        let last = messages.iter().rev().find(|m| matches!(m.role, MessageRole::User));
        let Some(msg) = last else {
            return Ok("Hello! I'm RIA AI. How can I help you today?".to_string());
        };
        let content = msg.content.to_lowercase();
        let resp = if content.contains("hello") || content.contains("hi") {
            "Hello! 👋 How can I help you today?".to_string()
        } else if content.contains("how are you") {
            "I'm doing great and ready to help!".to_string()
        } else if content.contains("code") || content.contains("program") {
            "I can assist with coding questions, examples, and debugging. What are you working on?".to_string()
        } else if content.contains("model") || content.contains("onnx") {
            "Model loading framework is active. Load an ONNX model from the 🧠 Models tab for real inference.".to_string()
        } else {
            format!("You said: \"{}\"\n\nI'm streaming this response chunk-by-chunk. Load a model to enable real ONNX inference.",
                    if msg.content.len()>120 { format!("{}...", &msg.content[..117]) } else { msg.content.clone() })
        };
        Ok(resp)
    }
    fn get_model_info(&self) -> Result<std::collections::HashMap<String,String>> {
        let mut m = std::collections::HashMap::new();
        m.insert("provider".into(), "Demo Provider".into());
        Ok(m)
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

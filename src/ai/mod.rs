pub mod inference;
pub mod providers;
pub mod models;
pub mod tokenizer;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub id: String,
    pub content: String,
    pub role: MessageRole,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub model_used: Option<String>,
    pub inference_time: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSession {
    pub id: String,
    pub title: String,
    pub messages: Vec<ChatMessage>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub model_path: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub execution_provider: ExecutionProvider,
    pub use_gpu: bool,
    pub use_npu: bool,
    #[serde(default)]
    pub prefer_npu: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExecutionProvider {
    Cpu,
    Cuda,
    DirectML,
    CoreML,
    OpenVINO,
    QNN, // Qualcomm NPU
    NNAPI, // Android NPU
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            max_tokens: 2048,
            temperature: 0.7,
            top_p: 0.9,
            execution_provider: ExecutionProvider::Cpu,
            use_gpu: false,
            use_npu: false,
            prefer_npu: true, // prioritize NPU-first experience on supported systems
        }
    }
}

pub trait AIProvider {
    fn name(&self) -> &str;
    fn is_available(&self) -> bool;
    fn generate_response(&mut self, messages: &[ChatMessage]) -> Result<String>;
    fn get_model_info(&self) -> Result<HashMap<String, String>>;
}
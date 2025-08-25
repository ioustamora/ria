pub mod inference;
pub mod providers;
pub mod models;
pub mod tokenizer;
pub mod sampler;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::any::Any;

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
    /// If prefer_npu is true and OpenVINO EP is selected/forced, use this device string.
    #[serde(default = "InferenceConfig::default_prefer_npu_device_string")]
    pub prefer_npu_device_string: String,
    /// Enable lightweight profiling during model load (writes simple custom profile file, not ORT native yet).
    #[serde(default)]
    pub profiling: bool,
    /// Number of warmup iterations to run immediately after session creation (adaptive probe style) to stabilize performance.
    #[serde(default)]
    pub warmup_iterations: u32,
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
            prefer_npu_device_string: Self::default_prefer_npu_device_string(),
            profiling: false,
            warmup_iterations: 0,
        }
    }
}

impl InferenceConfig {
    fn default_prefer_npu_device_string() -> String { "AUTO:NPU,CPU".to_string() }
}

pub trait AIProvider {
    fn name(&self) -> &str;
    fn is_available(&self) -> bool;
    fn generate_response(&mut self, messages: &[ChatMessage]) -> Result<String>;
    fn get_model_info(&self) -> Result<HashMap<String, String>>;
    fn as_any(&self) -> &dyn Any;
}
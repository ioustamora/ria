use crate::ai::{ExecutionProvider, InferenceConfig};
use crate::ui::app::Theme;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub theme: Theme,
    pub ai_config: InferenceConfig,
    pub animation_quality: u32,
    pub enable_animations: bool,
    pub enable_sound: bool,
    pub models_directory: PathBuf,
    pub chat_history_path: PathBuf,
    pub auto_save: bool,
    pub max_chat_history: usize,
    pub window_size: (f32, f32),
    pub window_position: Option<(f32, f32)>,
    pub last_used_model: Option<String>,
    pub auto_load_last_model: bool,
    // New automation flags
    #[serde(default)]
    pub auto_select_latest_model: bool, // If no last model, pick most recently modified .onnx
    #[serde(default)]
    pub auto_load_new_download: bool,   // Auto load a model right after successful download
    #[serde(default)]
    pub auto_fix_onnx_runtime: bool,    // Attempt automatic ONNX runtime fix on version mismatch
    #[serde(default)]
    pub enable_ep_fallback: bool,       // Future: attempt alternate EPs on failure
}

impl Default for AppConfig {
    fn default() -> Self {
        let config_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("ria-ai-chat");

        Self {
            theme: Theme::Dark,
            ai_config: InferenceConfig::default(),
            animation_quality: 2, // High quality
            enable_animations: true,
            enable_sound: false,
            models_directory: config_dir.join("models"),
            chat_history_path: config_dir.join("chat_history.json"),
            auto_save: true,
            max_chat_history: 100,
            window_size: (1200.0, 800.0),
            window_position: None,
            last_used_model: None,
            auto_load_last_model: true,
            auto_select_latest_model: true,
            auto_load_new_download: true,
            auto_fix_onnx_runtime: true,
            enable_ep_fallback: true,
        }
    }
}

impl AppConfig {
    pub fn load() -> Result<Self> {
        let config_path = Self::get_config_path()?;
        
        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let config: AppConfig = serde_json::from_str(&content)?;
            Ok(config)
        } else {
            // Create default config and save it
            let config = Self::default();
            config.save()?;
            Ok(config)
        }
    }

    pub fn save(&self) -> Result<()> {
        let config_path = Self::get_config_path()?;
        
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(&config_path, content)?;
        
        tracing::info!("Configuration saved to {:?}", config_path);
        Ok(())
    }

    fn get_config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not find config directory"))?;
        
        Ok(config_dir.join("ria-ai-chat").join("config.json"))
    }

    pub fn ensure_directories(&self) -> Result<()> {
        std::fs::create_dir_all(&self.models_directory)?;
        
        if let Some(parent) = self.chat_history_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        Ok(())
    }

    pub fn get_available_execution_providers(&self) -> Vec<ExecutionProvider> {
        let mut providers = vec![ExecutionProvider::Cpu];

        // Add platform-specific providers
        if cfg!(target_os = "windows") {
            providers.push(ExecutionProvider::DirectML);
            
            // Check for CUDA on Windows
            if Self::is_cuda_available() {
                providers.push(ExecutionProvider::Cuda);
            }
        }

        if cfg!(target_os = "macos") {
            providers.push(ExecutionProvider::CoreML);
        }

        if cfg!(target_os = "linux") {
            if Self::is_cuda_available() {
                providers.push(ExecutionProvider::Cuda);
            }
        }

        // Intel OpenVINO (cross-platform)
        if Self::is_openvino_available() {
            providers.push(ExecutionProvider::OpenVINO);
        }

        // NPU providers (platform-specific)
        if cfg!(all(target_os = "windows", target_arch = "aarch64")) {
            providers.push(ExecutionProvider::QNN);
        }

        if cfg!(target_os = "android") {
            providers.push(ExecutionProvider::NNAPI);
        }

        providers
    }

    fn is_cuda_available() -> bool {
        // Simple check for CUDA availability
        std::process::Command::new("nvidia-smi")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    fn is_openvino_available() -> bool {
        // Check for Intel OpenVINO runtime
        // This is simplified - in production you'd check for proper installation
        std::env::var("INTEL_OPENVINO_DIR").is_ok()
    }

    pub fn validate(&self) -> Result<()> {
        // Validate AI config
        if self.ai_config.max_tokens == 0 {
            return Err(anyhow::anyhow!("Max tokens must be greater than 0"));
        }

        if !(0.0..=2.0).contains(&self.ai_config.temperature) {
            return Err(anyhow::anyhow!("Temperature must be between 0.0 and 2.0"));
        }

        if !(0.0..=1.0).contains(&self.ai_config.top_p) {
            return Err(anyhow::anyhow!("Top-p must be between 0.0 and 1.0"));
        }

        // Validate paths
        if !self.ai_config.model_path.is_empty() {
            let model_path = PathBuf::from(&self.ai_config.model_path);
            if !model_path.exists() {
                return Err(anyhow::anyhow!("Model path does not exist: {}", self.ai_config.model_path));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = AppConfig::default();
        assert!(config.enable_animations);
        assert_eq!(config.animation_quality, 2);
        assert_eq!(config.ai_config.max_tokens, 2048);
    }

    #[test]
    fn test_config_validation() {
        let mut config = AppConfig::default();
        assert!(config.validate().is_ok());

        config.ai_config.max_tokens = 0;
        assert!(config.validate().is_err());

        config.ai_config.max_tokens = 1024;
        config.ai_config.temperature = 3.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_serialization() {
        let config = AppConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: AppConfig = serde_json::from_str(&json).unwrap();
        
        assert_eq!(config.animation_quality, deserialized.animation_quality);
        assert_eq!(config.enable_animations, deserialized.enable_animations);
    }
}
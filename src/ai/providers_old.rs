use super::*;
use super::tokenizer::SimpleTokenizer;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use sysinfo::System;
use ort::{Environment, ExecutionProvider as OrtExecutionProvider, Session, SessionBuilder};
use ndarray::{Array1, Array2};
use std::sync::Arc;

pub struct DeviceDetector {
    system: System,
}

impl DeviceDetector {
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        Self { system }
    }

    pub fn detect_available_providers(&self) -> Vec<ExecutionProvider> {
        let mut providers = vec![ExecutionProvider::Cpu];

        // Check for NVIDIA GPU
        if self.has_nvidia_gpu() {
            providers.push(ExecutionProvider::Cuda);
        }

        // Check for DirectML (Windows)
        if cfg!(target_os = "windows") {
            providers.push(ExecutionProvider::DirectML);
        }

        // Check for CoreML (macOS)
        if cfg!(target_os = "macos") {
            providers.push(ExecutionProvider::CoreML);
        }

        // Check for Intel OpenVINO
        if self.has_intel_processor() {
            providers.push(ExecutionProvider::OpenVINO);
        }

        // Check for NPU support (Qualcomm)
        if self.has_qualcomm_npu() {
            providers.push(ExecutionProvider::QNN);
        }

        providers
    }

    fn has_nvidia_gpu(&self) -> bool {
        // Check for NVIDIA GPU in system
        // This is a simplified check - in production you'd use proper GPU detection
        std::process::Command::new("nvidia-smi")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    fn has_intel_processor(&self) -> bool {
        self.system
            .cpus()
            .iter()
            .any(|cpu| cpu.brand().to_lowercase().contains("intel"))
    }

    fn has_qualcomm_npu(&self) -> bool {
        // Simplified NPU detection - would need more sophisticated detection in production
        cfg!(target_arch = "aarch64") && cfg!(target_os = "windows")
    }
}

pub struct OnnxProvider {
    config: InferenceConfig,
    is_loaded: bool,
    tokenizer: SimpleTokenizer,
    session: Option<Session>,
    environment: Arc<Environment>,
}

impl OnnxProvider {
    pub fn new(config: InferenceConfig) -> Result<Self> {
        let environment = Arc::new(
            Environment::builder()
                .with_name("RIA-AI-Chat")
                .with_log_level(ort::LoggingLevel::Warning)
                .build()?
        );
        
        Ok(Self {
            config,
            is_loaded: false,
            tokenizer: SimpleTokenizer::new(),
            session: None,
            environment,
        })
    }

    pub fn load_model(&mut self) -> Result<()> {
        if self.config.model_path.is_empty() {
            return Err(anyhow!("Model path is empty"));
        }

        let model_path = std::path::Path::new(&self.config.model_path);
        if !model_path.exists() {
            return Err(anyhow!("Model file does not exist: {}", self.config.model_path));
        }

        if !model_path.extension().map_or(false, |ext| ext == "onnx") {
            return Err(anyhow!("File is not an ONNX model: {}", self.config.model_path));
        }

        tracing::info!("Loading ONNX model: {}", self.config.model_path);
        
        // Build session with appropriate execution provider
        let mut session_builder = SessionBuilder::new(&self.environment)?;
        
        // Configure execution provider based on config
        match self.config.execution_provider {
            ExecutionProvider::Cuda => {
                if let Err(e) = session_builder.with_execution_providers([OrtExecutionProvider::CUDA(Default::default())]) {
                    tracing::warn!("Failed to enable CUDA provider: {}, falling back to CPU", e);
                    session_builder.with_execution_providers([OrtExecutionProvider::CPU(Default::default())])?;
                }
            }
            ExecutionProvider::DirectML => {
                if let Err(e) = session_builder.with_execution_providers([OrtExecutionProvider::DirectML(Default::default())]) {
                    tracing::warn!("Failed to enable DirectML provider: {}, falling back to CPU", e);
                    session_builder.with_execution_providers([OrtExecutionProvider::CPU(Default::default())])?;
                }
            }
            ExecutionProvider::CoreML => {
                if let Err(e) = session_builder.with_execution_providers([OrtExecutionProvider::CoreML(Default::default())]) {
                    tracing::warn!("Failed to enable CoreML provider: {}, falling back to CPU", e);
                    session_builder.with_execution_providers([OrtExecutionProvider::CPU(Default::default())])?;
                }
            }
            _ => {
                session_builder.with_execution_providers([OrtExecutionProvider::CPU(Default::default())])?;
            }
        }
        
        // Set session configuration for optimization
        session_builder.with_optimization_level(ort::GraphOptimizationLevel::All)?;
        session_builder.with_intra_threads(num_cpus::get().min(4))?; // Limit to 4 threads max
        
        // Create the session
        let session = session_builder.commit_from_file(&self.config.model_path)
            .map_err(|e| anyhow!("Failed to load ONNX model: {}", e))?;
        
        // Store the session
        self.session = Some(session);
        self.is_loaded = true;
        
        tracing::info!("ONNX model loaded successfully: {}", self.config.model_path);
        tracing::info!("Model inputs: {:?}", self.session.as_ref().unwrap().inputs().iter().map(|i| &i.name).collect::<Vec<_>>());
        tracing::info!("Model outputs: {:?}", self.session.as_ref().unwrap().outputs().iter().map(|o| &o.name).collect::<Vec<_>>());
        
        Ok(())
    }

    pub fn tokenize(&mut self, text: &str) -> Result<Vec<i64>> {
        Ok(self.tokenizer.encode(text))
    }

    pub fn detokenize(&self, tokens: &[i64]) -> Result<String> {
        Ok(self.tokenizer.decode(tokens))
    }
}

impl OnnxProvider {
    fn generate_fallback_response(&self, messages: &[ChatMessage]) -> Result<String> {
        let last_message = messages.last();
        
        let response = if let Some(msg) = last_message {
            let content = msg.content.to_lowercase();
            
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
                format!("I understand you mentioned: \"{}\"\n\nI'm currently running in demo mode with a basic response system. To get more sophisticated AI responses:\n\n1. Download an ONNX model (try the Models tab)\n2. Select it in the interface\n3. The model will provide much better responses!\n\nFor now, I can have simple conversations. What else would you like to talk about?", 
                    if msg.content.len() > 100 { 
                        format!("{}...", &msg.content[..97]) 
                    } else { 
                        msg.content.clone() 
                    })
            }
        } else {
            "Hello! I'm RIA AI Chat. How can I assist you today?".to_string()
        };

        Ok(response)
    }
}

impl AIProvider for OnnxProvider {
    fn name(&self) -> &str {
        "ONNX Runtime"
    }

    fn is_available(&self) -> bool {
        self.is_loaded
    }

    fn generate_response(&mut self, messages: &[ChatMessage]) -> Result<String> {
        if !self.is_loaded {
            return Err(anyhow!("Model not loaded"));
        }

        // Prepare input tokens from chat messages
        let _input_tokens = self.tokenizer.prepare_chat_input(messages);
        
        // For now, always use fallback response
        // Real ONNX inference would go here
        self.generate_fallback_response(messages)
    }

    fn get_model_info(&self) -> Result<HashMap<String, String>> {
        let mut info = HashMap::new();
        info.insert("provider".to_string(), "ONNX Runtime".to_string());
        info.insert("model_path".to_string(), self.config.model_path.clone());
        info.insert("execution_provider".to_string(), format!("{:?}", self.config.execution_provider));
        Ok(info)
    }
}
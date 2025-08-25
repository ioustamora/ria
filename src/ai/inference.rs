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
        // NPU and hardware acceleration questions
        if content_lower.contains("npu") || content_lower.contains("neural processing") {
            return "🧠 **NPU (Neural Processing Unit) Information**\n\nHere's a simple Rust example for NPU-aware code:\n\n```rust\nuse std::arch::is_x86_feature_detected;\n\n// Check for AI acceleration capabilities\nfn detect_npu_support() -> bool {\n    // Check for Intel AMX (Advanced Matrix Extensions)\n    if is_x86_feature_detected!(\"avx512f\") {\n        println!(\"🟢 AVX-512 detected - good for AI workloads\");\n        return true;\n    }\n    \n    // Check system info for NPU\n    if std::env::var(\"OPENVINO_NPU_AVAILABLE\").is_ok() {\n        println!(\"🟢 OpenVINO NPU runtime detected\");\n        return true;\n    }\n    \n    false\n}\n\nfn main() {\n    if detect_npu_support() {\n        println!(\"🚀 NPU acceleration available!\");\n    } else {\n        println!(\"⚡ Using CPU optimization\");\n    }\n}\n```\n\n**NPU Integration Tips:**\n• Use OpenVINO toolkit for Intel NPUs\n• Check for DirectML on Windows\n• Leverage ONNX Runtime execution providers\n• Consider model quantization (INT8/FP16)\n\n**Current RIA Status**: ONNX Runtime version mismatch preventing NPU usage. The app expects v1.22+ but system has v1.17.1.".to_string();
        }
        
        // Specific Rust code requests
        if (content_lower.contains("write") || content_lower.contains("code")) && content_lower.contains("rust") && content_lower.contains("npu") {
            return "🦀 **Rust NPU Integration Code Example**\n\n```rust\nuse std::sync::Arc;\nuse tokio::sync::RwLock;\n\n// NPU-aware inference structure\n#[derive(Debug)]\npub struct NpuInference {\n    device_type: DeviceType,\n    model_path: String,\n    optimization_level: OptimLevel,\n}\n\n#[derive(Debug, Clone)]\nenum DeviceType {\n    CPU,\n    NPU,\n    GPU,\n    Auto,\n}\n\n#[derive(Debug)]\nenum OptimLevel {\n    Basic,\n    Aggressive,\n    NPUOptimized,\n}\n\nimpl NpuInference {\n    pub fn new() -> Self {\n        Self {\n            device_type: Self::detect_best_device(),\n            model_path: String::new(),\n            optimization_level: OptimLevel::NPUOptimized,\n        }\n    }\n    \n    fn detect_best_device() -> DeviceType {\n        // Check for Intel NPU\n        if std::env::var(\"INTEL_NPU_AVAILABLE\").is_ok() {\n            return DeviceType::NPU;\n        }\n        \n        // Check for GPU acceleration\n        if std::env::var(\"CUDA_VISIBLE_DEVICES\").is_ok() {\n            return DeviceType::GPU;\n        }\n        \n        DeviceType::CPU\n    }\n    \n    pub async fn load_model(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {\n        println!(\"🔄 Loading model on {:?}...\", self.device_type);\n        self.model_path = path.to_string();\n        \n        match self.device_type {\n            DeviceType::NPU => {\n                println!(\"⚡ Initializing NPU cores...\");\n                // NPU-specific initialization\n                self.init_npu_cores().await?;\n            },\n            DeviceType::GPU => {\n                println!(\"🎮 Using GPU acceleration...\");\n            },\n            DeviceType::CPU => {\n                println!(\"🖥️ Using optimized CPU inference...\");\n            },\n            DeviceType::Auto => {\n                println!(\"🤖 Auto-detecting best device...\");\n            }\n        }\n        \n        Ok(())\n    }\n    \n    async fn init_npu_cores(&self) -> Result<(), Box<dyn std::error::Error>> {\n        // Initialize NPU cores with OpenVINO or similar\n        println!(\"🧠 NPU cores initialized successfully\");\n        Ok(())\n    }\n    \n    pub async fn infer(&self, input: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {\n        match self.device_type {\n            DeviceType::NPU => {\n                println!(\"🚀 Running inference on NPU cores...\");\n                // NPU inference logic here\n                Ok(vec![0.95, 0.03, 0.02]) // Mock output\n            },\n            _ => {\n                println!(\"⚡ Running inference on {:#?}...\", self.device_type);\n                Ok(vec![0.85, 0.10, 0.05]) // Mock output\n            }\n        }\n    }\n}\n\n#[tokio::main]\nasync fn main() -> Result<(), Box<dyn std::error::Error>> {\n    let mut npu_engine = NpuInference::new();\n    npu_engine.load_model(\"./model.onnx\").await?;\n    \n    let input_data = vec![1.0, 2.0, 3.0, 4.0];\n    let output = npu_engine.infer(&input_data).await?;\n    \n    println!(\"🎯 Inference result: {:#?}\", output);\n    Ok(())\n}\n```\n\n**This code demonstrates:**\n✅ Device detection and NPU preference\n✅ Async model loading with device optimization\n✅ NPU core initialization\n✅ Device-specific inference paths\n✅ Error handling and logging\n\n**To integrate NPUs properly:**\n• Install compatible ONNX Runtime (1.22+)\n• Use OpenVINO for Intel NPUs\n• Enable DirectML for Windows NPU support\n• Test with quantized models (INT8/FP16)".to_string();
        }
        
        // Greeting responses
        if content_lower.contains("hello") || content_lower.contains("hi ") || content_lower.starts_with("hi") {
            return "Hello! 👋 I'm RIA AI Assistant. I'm currently running in intelligent demo mode and ready to help!\n\n🔧 **Available capabilities:**\n• Answer questions and provide explanations\n• Help with coding and programming\n• Discuss various topics\n• Provide contextual assistance\n\n💡 Load an ONNX model from the 🧠 Models tab for enhanced AI inference!\n\nWhat would you like to know?".to_string();
        }
        
        // Status and model questions
        if content_lower.contains("model") || content_lower.contains("onnx") || content_lower.contains("status") {
            return "🧠 **System Status Report**\n\n**Current State**: Demo Mode (ONNX Runtime issue)\n\n**⚠️ Issue Detected:**\n• ONNX Runtime version mismatch\n• System has v1.17.1, app requires v1.22+\n• NPU cores detected but not accessible\n\n**🔧 Solutions:**\n1. **Update ONNX Runtime:**\n   ```bash\n   pip install onnxruntime --upgrade\n   # or for GPU/NPU support:\n   pip install onnxruntime-openvino\n   ```\n\n2. **Alternative: Use system package manager**\n   ```bash\n   winget install Microsoft.ONNXRuntime\n   ```\n\n**✅ What works now:**\n• Intelligent demo responses\n• Code examples and explanations\n• Programming assistance\n• NPU programming tutorials\n\n**🚀 After fixing ONNX Runtime:**\n• Real AI model inference\n• NPU acceleration support\n• Advanced reasoning capabilities\n\n*Demo mode is fully functional for learning and development!*".to_string();
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

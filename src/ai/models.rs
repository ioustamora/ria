use super::*;
use anyhow::Result;
use std::path::{Path, PathBuf};
use sha2::{Digest, Sha256};

pub struct ModelManager {
    models_dir: PathBuf,
    available_models: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub size: u64,
    pub model_type: ModelType,
    pub supported_providers: Vec<ExecutionProvider>,
    pub description: String,
    pub quantization: Option<QuantizationType>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    LanguageModel,
    ChatModel,
    CodeModel,
    MultiModal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationType {
    FP32,
    FP16,
    INT8,
    INT4,
    Q4F16, // 4-bit quantization with FP16 weights
}

impl ModelManager {
    pub fn new<P: AsRef<Path>>(models_dir: P) -> Result<Self> {
        let models_dir = models_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&models_dir)?;
        
        let mut manager = Self {
            models_dir,
            available_models: Vec::new(),
        };
        
        manager.scan_models()?;
        Ok(manager)
    }

    /// Download an auxiliary file (e.g., tokenizer JSON) to the specified destination path.
    /// This method overwrites any existing file.
    pub async fn download_aux_file(&self, url: &str, dest_path: &Path) -> Result<()> {
        use futures_util::StreamExt;
        use tokio::fs::OpenOptions;
        use tokio::io::AsyncWriteExt;

        if let Some(parent) = dest_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let client = reqwest::Client::new();
        let response = client.get(url).send().await?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to download aux file: HTTP {}",
                response.status()
            ));
        }

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(dest_path)
            .await?;

        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
        }
        file.flush().await?;

        tracing::info!("Downloaded aux file to {}", dest_path.display());
        Ok(())
    }

    pub fn scan_models(&mut self) -> Result<()> {
        self.available_models.clear();
        
        if !self.models_dir.exists() {
            return Ok(());
        }

        for entry in std::fs::read_dir(&self.models_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("onnx") {
                if let Ok(model_info) = self.analyze_model(&path) {
                    self.available_models.push(model_info);
                }
            }
        }

        Ok(())
    }

    fn analyze_model(&self, path: &Path) -> Result<ModelInfo> {
        let metadata = std::fs::metadata(path)?;
        let name = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Unknown")
            .to_string();

        // Determine model type based on filename patterns
        let model_type = if name.to_lowercase().contains("chat") {
            ModelType::ChatModel
        } else if name.to_lowercase().contains("code") {
            ModelType::CodeModel
        } else if name.to_lowercase().contains("vision") || name.to_lowercase().contains("multimodal") {
            ModelType::MultiModal
        } else {
            ModelType::LanguageModel
        };

        // Determine quantization type based on filename
        let quantization = if name.contains("fp16") {
            Some(QuantizationType::FP16)
        } else if name.contains("int8") {
            Some(QuantizationType::INT8)
        } else if name.contains("int4") {
            Some(QuantizationType::INT4)
        } else {
            Some(QuantizationType::FP32)
        };

        // All models support CPU, add others based on system capabilities
        let mut supported_providers = vec![ExecutionProvider::Cpu];
        
        // Add GPU providers based on system
        if cfg!(target_os = "windows") {
            supported_providers.push(ExecutionProvider::DirectML);
        }
        if cfg!(target_os = "macos") {
            supported_providers.push(ExecutionProvider::CoreML);
        }

        Ok(ModelInfo {
            name,
            path: path.to_path_buf(),
            size: metadata.len(),
            model_type,
            supported_providers,
            description: format!("ONNX model loaded from {}", path.display()),
            quantization,
        })
    }

    pub fn get_available_models(&self) -> &[ModelInfo] {
        &self.available_models
    }

    pub fn get_model_by_name(&self, name: &str) -> Option<&ModelInfo> {
        self.available_models.iter().find(|model| model.name == name)
    }

    pub async fn download_model(&mut self, url: &str, name: &str) -> Result<PathBuf> {
        self.download_model_with_verify(url, name, None).await
    }

    pub async fn download_model_with_verify(&mut self, url: &str, name: &str, expected_sha256: Option<&str>) -> Result<PathBuf> {
        self.download_model_with_verify_and_progress::<fn(u64, u64, f64)>(url, name, expected_sha256, None).await
    }

    pub async fn download_model_with_verify_and_progress<F>(
        &mut self, 
        url: &str, 
        name: &str, 
        expected_sha256: Option<&str>,
        mut progress_callback: Option<F>
    ) -> Result<PathBuf> 
    where
        F: FnMut(u64, u64, f64) + Send + 'static,
    {
        use tokio::io::AsyncWriteExt;
        use tokio::fs::OpenOptions;
        use futures_util::StreamExt;
        
        // Prepare paths
        let sanitized_name = crate::utils::sanitize_filename(name);
        let final_path = crate::utils::ensure_file_extension(&self.models_dir.join(&sanitized_name), "onnx");
        let part_path = final_path.with_extension("onnx.part");

        // Ensure the models directory exists
        std::fs::create_dir_all(&self.models_dir)?;

        // Determine resume offset
        let mut resume_from: u64 = 0;
        if part_path.exists() {
            if let Ok(meta) = std::fs::metadata(&part_path) {
                resume_from = meta.len();
                tracing::info!("Resuming download for {} at {} bytes", name, resume_from);
            }
        }

        // Build request (Range if resuming)
        let client = reqwest::Client::new();
        let mut req = client.get(url);
        if resume_from > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={}-", resume_from));
        }
        let response = req.send().await?;

        if !(response.status().is_success() || response.status() == reqwest::StatusCode::PARTIAL_CONTENT) {
            return Err(anyhow::anyhow!("Failed to download model: HTTP {}", response.status()));
        }

        let total_size = response.content_length();

        // Open part file for append
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&part_path)
            .await?;

        // Stream download with progress reporting
        let mut downloaded = resume_from;
        let mut stream = response.bytes_stream();
        let start_time = std::time::Instant::now();
        let mut last_update = start_time;
        
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;

            // Report progress every 100ms or so
            let now = std::time::Instant::now();
            if now.duration_since(last_update).as_millis() >= 100 {
                if let Some(total) = total_size {
                    let total_with_resume = resume_from + total;
                    let progress = (downloaded as f64) / (total_with_resume as f64);
                    let elapsed = now.duration_since(start_time).as_secs_f64();
                    let speed = if elapsed > 0.0 { (downloaded - resume_from) as f64 / elapsed } else { 0.0 };
                    
                    // Call progress callback if provided
                    if let Some(ref mut callback) = progress_callback {
                        callback(downloaded, total_with_resume, speed);
                    }
                    
                    tracing::debug!("Download progress for {}: {:.1}% ({:.1} KB/s)", name, progress * 100.0, speed / 1024.0);
                } else {
                    // Unknown total size
                    if let Some(ref mut callback) = progress_callback {
                        let elapsed = now.duration_since(start_time).as_secs_f64();
                        let speed = if elapsed > 0.0 { (downloaded - resume_from) as f64 / elapsed } else { 0.0 };
                        callback(downloaded, 0, speed);
                    }
                    tracing::debug!("Downloaded {} bytes for {}", downloaded, name);
                }
                last_update = now;
            }
        }
        file.flush().await?;

        // Verify SHA256 if provided
        if let Some(expected) = expected_sha256 {
            let mut hasher = Sha256::new();
            let mut f = std::fs::File::open(&part_path)?;
            let mut buf = [0u8; 1024 * 64];
            loop {
                let n = std::io::Read::read(&mut f, &mut buf)?;
                if n == 0 { break; }
                hasher.update(&buf[..n]);
            }
            let digest = hasher.finalize();
            let digest_hex = hex::encode(digest);
            if digest_hex.to_lowercase() != expected.to_lowercase() {
                return Err(anyhow::anyhow!("SHA256 mismatch for {}: expected {}, got {}", name, expected, digest_hex));
            }
            tracing::info!("SHA256 verified for {}", name);
        }

        // Move part to final
        tokio::fs::rename(&part_path, &final_path).await?;
        tracing::info!("Successfully downloaded model: {}", final_path.display());

        // Rescan models after download
        self.scan_models()?;
        
        Ok(final_path)
    }

    pub fn add_model_from_url(&self, _url: &str, _name: &str) -> Result<PathBuf> {
        // This is the sync version - redirect to async version
        Err(anyhow::anyhow!("Use download_model() async method instead"))
    }

    pub fn get_models_directory(&self) -> &Path {
        &self.models_dir
    }

    /// Detect pre-installed AI models on Windows Copilot+ PCs and other systems
    pub fn detect_system_models(&self) -> Vec<ModelInfo> {
        let mut detected_models = Vec::new();
        
        // Windows Copilot+ PC model locations
        if cfg!(target_os = "windows") {
            detected_models.extend(self.scan_windows_system_models());
        }
        
        // macOS system models
        if cfg!(target_os = "macos") {
            detected_models.extend(self.scan_macos_system_models());
        }
        
        // Linux system models
        if cfg!(target_os = "linux") {
            detected_models.extend(self.scan_linux_system_models());
        }
        
        // Common cross-platform locations
        detected_models.extend(self.scan_common_model_locations());
        
        detected_models
    }
    
    #[cfg(target_os = "windows")]
    fn scan_windows_system_models(&self) -> Vec<ModelInfo> {
        let mut models = Vec::new();
        
        // Phi Silica model locations on Copilot+ PCs
        let phi_locations = vec![
            // Windows AI Platform locations
            "C:\\Windows\\System32\\Phi-3\\",
            "C:\\Windows\\System32\\AI\\Models\\Phi-3\\",
            "C:\\Program Files\\Microsoft\\AI\\Models\\Phi-3\\",
            "C:\\Program Files (x86)\\Microsoft\\AI\\Models\\Phi-3\\",
            
            // Windows Copilot locations  
            "C:\\Windows\\SystemApps\\Microsoft.Copilot\\Models\\",
            "C:\\Program Files\\WindowsApps\\Microsoft.Copilot\\Models\\",
            
            // DirectML / Windows ML locations
            "C:\\Windows\\System32\\DirectML\\Models\\",
            "C:\\Program Files\\Common Files\\Microsoft\\DirectML\\Models\\",
            
            // Intel NPU / OpenVINO locations
            "C:\\Program Files\\Intel\\OpenVINO\\models\\",
            "C:\\Program Files (x86)\\Intel\\OpenVINO\\models\\",
            "C:\\Intel\\OpenVINO\\models\\",
            
            // User profile locations
            "C:\\Users\\Public\\AI\\Models\\",
        ];
        
        for location in phi_locations {
            models.extend(self.scan_directory_for_models(location, "Phi-3 Silica (System)"));
        }
        
        // Other known Windows AI model locations
        let general_locations = vec![
            "C:\\Windows\\System32\\onnxruntime\\models\\",
            "C:\\Program Files\\ONNX Runtime\\models\\",
            "C:\\Program Files\\Microsoft\\AI Platform\\models\\",
        ];
        
        for location in general_locations {
            models.extend(self.scan_directory_for_models(location, "System Model"));
        }
        
        models
    }
    
    #[cfg(target_os = "macos")]
    fn scan_macos_system_models(&self) -> Vec<ModelInfo> {
        let mut models = Vec::new();
        
        let macos_locations = vec![
            "/System/Library/PrivateFrameworks/CoreML.framework/Versions/A/Resources/Models/",
            "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/Library/CoreML/Models/",
            "/usr/local/lib/onnxruntime/models/",
            "/opt/intel/openvino/models/",
        ];
        
        for location in macos_locations {
            models.extend(self.scan_directory_for_models(location, "System Model"));
        }
        
        models
    }
    
    #[cfg(target_os = "linux")]
    fn scan_linux_system_models(&self) -> Vec<ModelInfo> {
        let mut models = Vec::new();
        
        let linux_locations = vec![
            "/usr/share/onnxruntime/models/",
            "/usr/local/share/onnxruntime/models/",
            "/opt/intel/openvino/models/",
            "/usr/lib/onnxruntime/models/",
            "/var/lib/ai/models/",
        ];
        
        for location in linux_locations {
            models.extend(self.scan_directory_for_models(location, "System Model"));
        }
        
        models
    }
    
    #[cfg(not(target_os = "windows"))]
    fn scan_windows_system_models(&self) -> Vec<ModelInfo> { Vec::new() }
    
    #[cfg(not(target_os = "macos"))]
    fn scan_macos_system_models(&self) -> Vec<ModelInfo> { Vec::new() }
    
    #[cfg(not(target_os = "linux"))]
    fn scan_linux_system_models(&self) -> Vec<ModelInfo> { Vec::new() }
    
    fn scan_common_model_locations(&self) -> Vec<ModelInfo> {
        let mut models = Vec::new();
        
        // Common development and user locations
        if let Ok(home_dir) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
            let common_locations = vec![
                format!("{}/.cache/huggingface/hub/", home_dir),
                format!("{}/AppData/Local/Microsoft/AI/Models/", home_dir),
                format!("{}/.local/share/ai/models/", home_dir),
                format!("{}/AI/Models/", home_dir),
                format!("{}/Documents/AI/Models/", home_dir),
            ];
            
            for location in common_locations {
                models.extend(self.scan_directory_for_models(&location, "User Model"));
            }
        }
        
        models
    }
    
    fn scan_directory_for_models(&self, directory: &str, model_category: &str) -> Vec<ModelInfo> {
        let mut models = Vec::new();
        let path = Path::new(directory);
        
        if !path.exists() || !path.is_dir() {
            return models;
        }
        
        // Scan for ONNX models
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let entry_path = entry.path();
                
                if entry_path.is_file() && 
                   entry_path.extension().and_then(|s| s.to_str()) == Some("onnx") {
                    
                    if let Ok(model_info) = self.analyze_system_model(&entry_path, model_category) {
                        tracing::info!("🔍 Detected system model: {} at {}", 
                                     model_info.name, entry_path.display());
                        models.push(model_info);
                    }
                } else if entry_path.is_dir() {
                    // Recursively scan subdirectories (1 level deep to avoid performance issues)
                    if let Ok(sub_entries) = std::fs::read_dir(&entry_path) {
                        for sub_entry in sub_entries.flatten() {
                            let sub_path = sub_entry.path();
                            if sub_path.is_file() && 
                               sub_path.extension().and_then(|s| s.to_str()) == Some("onnx") {
                                
                                if let Ok(model_info) = self.analyze_system_model(&sub_path, model_category) {
                                    tracing::info!("🔍 Detected system model: {} at {}", 
                                                 model_info.name, sub_path.display());
                                    models.push(model_info);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        models
    }
    
    fn analyze_system_model(&self, path: &Path, category: &str) -> Result<ModelInfo> {
        let metadata = std::fs::metadata(path)?;
        let file_name = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Unknown")
            .to_string();
            
        // Enhanced name detection for known models
        let display_name = self.get_friendly_model_name(&file_name, path);
        
        // Determine model type based on path and filename
        let model_type = self.determine_model_type(&file_name, path);
        
        // Determine quantization from filename
        let quantization = self.determine_quantization(&file_name);
        
        // Determine supported providers based on system capabilities
        let supported_providers = self.determine_system_providers(path);
        
        Ok(ModelInfo {
            name: display_name,
            path: path.to_path_buf(),
            size: metadata.len(),
            model_type,
            supported_providers,
            description: format!("{} - Detected at {}", category, path.display()),
            quantization,
        })
    }
    
    fn get_friendly_model_name(&self, filename: &str, path: &Path) -> String {
        let lower_name = filename.to_lowercase();
        let path_str = path.to_string_lossy().to_lowercase();
        
        // Detect Phi-3 Silica models
        if lower_name.contains("phi-3") || lower_name.contains("phi3") || path_str.contains("phi-3") {
            if lower_name.contains("silica") || path_str.contains("copilot") || path_str.contains("system32") {
                return "Phi-3 Silica (System)".to_string();
            } else if lower_name.contains("mini") {
                return "Phi-3 Mini".to_string();
            } else if lower_name.contains("medium") {
                return "Phi-3 Medium".to_string();
            } else {
                return "Phi-3 Model".to_string();
            }
        }
        
        // Detect other known models
        if lower_name.contains("llama") {
            return "LLaMA Model".to_string();
        }
        if lower_name.contains("qwen") {
            return "Qwen Model".to_string();
        }
        if lower_name.contains("mistral") {
            return "Mistral Model".to_string();
        }
        if lower_name.contains("codellama") || lower_name.contains("code-llama") {
            return "Code Llama Model".to_string();
        }
        
        // Use the original filename as fallback
        filename.to_string()
    }
    
    fn determine_model_type(&self, filename: &str, path: &Path) -> ModelType {
        let lower_name = filename.to_lowercase();
        let path_str = path.to_string_lossy().to_lowercase();
        
        if lower_name.contains("code") || lower_name.contains("programming") {
            ModelType::CodeModel
        } else if lower_name.contains("chat") || lower_name.contains("instruct") || 
                  lower_name.contains("silica") || path_str.contains("copilot") {
            ModelType::ChatModel
        } else if lower_name.contains("vision") || lower_name.contains("multimodal") {
            ModelType::MultiModal
        } else {
            ModelType::LanguageModel
        }
    }
    
    fn determine_quantization(&self, filename: &str) -> Option<QuantizationType> {
        let lower_name = filename.to_lowercase();
        
        if lower_name.contains("q4f16") {
            Some(QuantizationType::Q4F16)
        } else if lower_name.contains("int4") || lower_name.contains("4bit") {
            Some(QuantizationType::INT4)
        } else if lower_name.contains("int8") || lower_name.contains("8bit") {
            Some(QuantizationType::INT8)
        } else if lower_name.contains("fp16") || lower_name.contains("half") {
            Some(QuantizationType::FP16)
        } else if lower_name.contains("fp32") || lower_name.contains("float32") {
            Some(QuantizationType::FP32)
        } else {
            // Default assumption for system models
            Some(QuantizationType::INT8)
        }
    }
    
    fn determine_system_providers(&self, path: &Path) -> Vec<ExecutionProvider> {
        let mut providers = vec![ExecutionProvider::Cpu];
        let path_str = path.to_string_lossy().to_lowercase();
        
        // If in system directories, likely optimized for system hardware
        if path_str.contains("system32") || path_str.contains("copilot") || 
           path_str.contains("microsoft") || path_str.contains("intel") {
            
            // Windows system models are likely optimized for NPU/DirectML
            if cfg!(target_os = "windows") {
                providers.push(ExecutionProvider::DirectML);
                if path_str.contains("intel") || path_str.contains("openvino") {
                    providers.push(ExecutionProvider::OpenVINO);
                }
            }
            
            // macOS system models support CoreML
            if cfg!(target_os = "macos") {
                providers.push(ExecutionProvider::CoreML);
            }
        }
        
        // Add CUDA if available (check for CUDA installation)
        if self.is_cuda_available() {
            providers.push(ExecutionProvider::Cuda);
        }
        
        providers
    }
    
    fn is_cuda_available(&self) -> bool {
        // Simple check for NVIDIA GPU presence
        if cfg!(target_os = "windows") {
            std::path::Path::new("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA").exists()
        } else if cfg!(target_os = "linux") {
            std::path::Path::new("/usr/local/cuda").exists() || 
            std::path::Path::new("/usr/cuda").exists()
        } else {
            false
        }
    }

    /// Get both local and system-detected models
    pub fn get_all_available_models(&self) -> Vec<ModelInfo> {
        let mut all_models = self.available_models.clone();
        all_models.extend(self.detect_system_models());
        all_models
    }

    pub fn format_file_size(size: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = size as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        format!("{:.1} {}", size, UNITS[unit_index])
    }
}
use sysinfo::System;
use std::collections::HashMap;

pub struct SystemInfo {
    system: System,
}

impl Default for SystemInfo {
    fn default() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        Self { system }
    }
}

impl SystemInfo {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn refresh(&mut self) {
        self.system.refresh_all();
    }

    pub fn get_cpu_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        
        if let Some(cpu) = self.system.cpus().first() {
            info.insert("brand".to_string(), cpu.brand().to_string());
            info.insert("frequency".to_string(), format!("{} MHz", cpu.frequency()));
        }
        
        info.insert("core_count".to_string(), self.system.cpus().len().to_string());
        info.insert("usage".to_string(), format!("{:.1}%", self.system.global_cpu_usage()));
        
        info
    }

    pub fn get_memory_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        
        let total_memory = self.system.total_memory();
        let used_memory = self.system.used_memory();
        let free_memory = total_memory - used_memory;
        
        info.insert("total".to_string(), crate::utils::format_file_size(total_memory));
        info.insert("used".to_string(), crate::utils::format_file_size(used_memory));
        info.insert("free".to_string(), crate::utils::format_file_size(free_memory));
        info.insert("usage_percent".to_string(), 
                   format!("{:.1}%", (used_memory as f64 / total_memory as f64) * 100.0));
        
        info
    }

    pub fn get_gpu_info(&self) -> Vec<HashMap<String, String>> {
        let mut gpus = Vec::new();
        
        // Try to get NVIDIA GPU info
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args(&["--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
            .output()
        {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                for line in output_str.lines() {
                    let parts: Vec<&str> = line.split(", ").collect();
                    if parts.len() == 4 {
                        let mut gpu = HashMap::new();
                        gpu.insert("name".to_string(), parts[0].to_string());
                        gpu.insert("memory_total".to_string(), format!("{} MB", parts[1]));
                        gpu.insert("memory_used".to_string(), format!("{} MB", parts[2]));
                        gpu.insert("utilization".to_string(), format!("{}%", parts[3]));
                        gpu.insert("type".to_string(), "NVIDIA".to_string());
                        gpus.push(gpu);
                    }
                }
            }
        }

        // If no NVIDIA GPUs found, try to detect other GPUs
        if gpus.is_empty() {
            if cfg!(target_os = "windows") {
                // Try to detect DirectML-compatible GPUs on Windows
                if let Some(gpu) = self.detect_windows_gpu() {
                    gpus.push(gpu);
                }
            } else if cfg!(target_os = "macos") {
                // Detect Metal-compatible GPUs on macOS
                if let Some(gpu) = self.detect_macos_gpu() {
                    gpus.push(gpu);
                }
            }
        }

        gpus
    }

    #[cfg(target_os = "windows")]
    fn detect_windows_gpu(&self) -> Option<HashMap<String, String>> {
        // Try to get GPU info using Windows Management Instrumentation
        if let Ok(output) = std::process::Command::new("wmic")
            .args(&["path", "win32_VideoController", "get", "name"])
            .output()
        {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                for line in output_str.lines() {
                    let line = line.trim();
                    if !line.is_empty() && line != "Name" {
                        let mut gpu = HashMap::new();
                        gpu.insert("name".to_string(), line.to_string());
                        gpu.insert("type".to_string(), "DirectML".to_string());
                        return Some(gpu);
                    }
                }
            }
        }
        None
    }

    #[cfg(target_os = "macos")]
    fn detect_macos_gpu(&self) -> Option<HashMap<String, String>> {
        // Try to get GPU info using system_profiler
        if let Ok(output) = std::process::Command::new("system_profiler")
            .args(&["SPDisplaysDataType", "-json"])
            .output()
        {
            if output.status.success() {
                // Parse JSON would require serde_json, so simplified approach
                let mut gpu = HashMap::new();
                gpu.insert("name".to_string(), "Metal-compatible GPU".to_string());
                gpu.insert("type".to_string(), "Metal".to_string());
                return Some(gpu);
            }
        }
        None
    }

    #[cfg(not(target_os = "windows"))]
    fn detect_windows_gpu(&self) -> Option<HashMap<String, String>> { None }
    
    #[cfg(not(target_os = "macos"))]
    fn detect_macos_gpu(&self) -> Option<HashMap<String, String>> { None }

    pub fn get_os_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        
        info.insert("name".to_string(), System::name().unwrap_or_else(|| "Unknown".to_string()));
        info.insert("version".to_string(), System::os_version().unwrap_or_else(|| "Unknown".to_string()));
        info.insert("kernel_version".to_string(), System::kernel_version().unwrap_or_else(|| "Unknown".to_string()));
        info.insert("architecture".to_string(), std::env::consts::ARCH.to_string());
        
        info
    }

    pub fn has_npu(&self) -> bool {
        // Prefer vendor-specific detection
        self.detect_qualcomm_npu() || self.detect_intel_npu()
    }

    fn detect_qualcomm_npu(&self) -> bool {
        // Check for Qualcomm NPU (Snapdragon platforms)
        std::path::Path::new("C:\\Windows\\System32\\QnnHtp.dll").exists() ||
        std::path::Path::new("C:\\Windows\\System32\\QnnCpu.dll").exists()
    }

    fn detect_intel_npu(&self) -> bool {
        // Heuristic checks for Intel NPU / OpenVINO presence
        if cfg!(target_os = "windows") {
            std::env::var("INTEL_OPENVINO_DIR").is_ok()
                || std::path::Path::new("C:\\Program Files\\Intel\\openvino").exists()
                || std::path::Path::new("C:\\Program Files (x86)\\Intel\\openvino").exists()
        } else {
            std::env::var("INTEL_OPENVINO_DIR").is_ok()
                || std::path::Path::new("/usr/lib/libopenvino.so").exists()
                || std::path::Path::new("/opt/intel/openvino").exists()
        }
    }

    pub fn get_available_compute_devices(&self) -> Vec<String> {
        let mut devices = vec!["CPU".to_string()];
        
        // Add GPU devices
        for gpu in self.get_gpu_info() {
            if let Some(name) = gpu.get("name") {
                devices.push(format!("GPU: {}", name));
            }
        }
        
        // Add NPU if available
        if self.has_npu() {
            devices.push("NPU".to_string());
        }
        
        devices
    }

    pub fn get_system_summary(&self) -> String {
        let cpu_info = self.get_cpu_info();
        let mem_info = self.get_memory_info();
        let os_info = self.get_os_info();
        
        format!(
            "{} {} on {} CPU with {} RAM",
            os_info.get("name").unwrap_or(&"Unknown OS".to_string()),
            os_info.get("version").unwrap_or(&"".to_string()),
            cpu_info.get("brand").unwrap_or(&"Unknown CPU".to_string()),
            mem_info.get("total").unwrap_or(&"Unknown".to_string())
        )
    }
}
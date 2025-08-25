use std::path::{Path, PathBuf};

/// Discover an ONNX model for tests.
/// Priority:
/// 1. Environment variable RIA_TEST_ONNX_MODEL if it exists and file present.
/// 2. Smallest .onnx file inside repository `models/` directory.
/// Returns absolute path as String if found.
pub fn discover_test_model() -> Option<String> {
    if let Ok(p) = std::env::var("RIA_TEST_ONNX_MODEL") {
        if Path::new(&p).exists() { return Some(p); }
    }
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let models_dir = repo_root.join("models");
    if models_dir.is_dir() {
        let mut candidates: Vec<(u64, PathBuf)> = std::fs::read_dir(&models_dir).ok()?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|e| e == "onnx").unwrap_or(false))
            .filter_map(|p| {
                if let Ok(meta) = std::fs::metadata(&p) { Some((meta.len(), p)) } else { None }
            })
            .collect();
        if candidates.is_empty() { return None; }
        candidates.sort_by_key(|(len, _)| *len);
        return candidates.first().map(|(_, p)| p.to_string_lossy().into_owned());
    }
    None
}

#[cfg(test)]
mod tests_internal {
    use super::*;

    #[test]
    fn discovery_runs() {
        // Should not panic, may return None.
        let _ = discover_test_model();
    }
}

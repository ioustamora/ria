//! Negative and edge-case tests for ONNX provider
use ria_ai_chat::ai::{InferenceConfig, ExecutionProvider};
use ria_ai_chat::ai::providers::{OnnxProvider, LoadError};

#[test]
fn load_missing_file_yields_file_missing() {
    let cfg = InferenceConfig { model_path: "nonexistent_model_file_12345.onnx".into(), execution_provider: ExecutionProvider::Cpu, ..InferenceConfig::default() };
    let mut provider = OnnxProvider::new(cfg).unwrap();
    let err = provider.load_model_classified().expect_err("expected failure");
    matches!(err, LoadError::FileMissing(_));
}

#[test]
fn load_wrong_extension_yields_not_onnx() {
    // create a temp file with wrong extension
    let tmp_dir = std::env::temp_dir();
    let bad_path = tmp_dir.join("temp_model.txt");
    std::fs::write(&bad_path, b"not an onnx model").unwrap();
    let cfg = InferenceConfig { model_path: bad_path.to_string_lossy().into_owned(), execution_provider: ExecutionProvider::Cpu, ..InferenceConfig::default() };
    let mut provider = OnnxProvider::new(cfg).unwrap();
    let err = provider.load_model_classified().expect_err("expected failure");
    assert!(matches!(err, LoadError::NotOnnxFile(_)));
}

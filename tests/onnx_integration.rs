//! ONNX integration tests
//!
//! These tests validate that:
//! 1. The ONNX provider can be constructed.
//! 2. A real (or placeholder) ONNX model file is detected & loaded.
//! 3. Model signature inputs are introspected.
//! 4. Adaptive probe executes without panic for minimal input.
//!
//! The test uses the sample model path from environment variable `RIA_TEST_ONNX_MODEL`.
//! Provide a small ONNX file (e.g., a tiny distilled transformer) to exercise loading.

use ria_ai_chat::ai::{InferenceConfig, ExecutionProvider};
use ria_ai_chat::ai::providers::OnnxProvider;

fn test_model_path() -> Option<String> {
    std::env::var("RIA_TEST_ONNX_MODEL").ok()
}

#[test]
fn onnx_model_load_and_signature() {
    let Some(model_path) = test_model_path() else {
        eprintln!("SKIP: set RIA_TEST_ONNX_MODEL to run ONNX integration tests");
        return; // Treat absence as skip
    };

    assert!(std::path::Path::new(&model_path).exists(), "Test model path does not exist: {model_path}");

    let cfg = InferenceConfig { model_path: model_path.clone(), execution_provider: ExecutionProvider::Cpu, ..InferenceConfig::default() };
    let mut provider = OnnxProvider::new(cfg).expect("create provider");
    provider.load_model().expect("load model");

    // Signature should be available after load
    let sig_inputs = provider.debug_signature_input_names().expect("signature");
    assert!(!sig_inputs.is_empty(), "No inputs discovered in model signature");
}

#[test]
fn onnx_adaptive_probe_minimal_tokens() {
    let Some(model_path) = test_model_path() else {
        eprintln!("SKIP: set RIA_TEST_ONNX_MODEL to run ONNX integration tests");
        return;
    };
    let cfg = InferenceConfig { model_path: model_path.clone(), execution_provider: ExecutionProvider::Cpu, ..InferenceConfig::default() };
    let mut provider = OnnxProvider::new(cfg).expect("provider");
    provider.load_model().expect("load");

    // Build minimal fake messages to produce tokens
    use ria_ai_chat::ai::{ChatMessage, MessageRole};
    let msg = ChatMessage { id: "1".into(), content: "hello".into(), role: MessageRole::User, timestamp: chrono::Utc::now(), model_used: None, inference_time: None };
    let _ = provider.generate_response(&[msg]).expect("response generation");
}

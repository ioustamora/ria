//! Tests focused on actual inference path (run_onnx_inference)
//! Requires RIA_TEST_ONNX_MODEL env var to point to a valid small ONNX model.

use ria_ai_chat::ai::{InferenceConfig, ExecutionProvider, ChatMessage, MessageRole};
use ria_ai_chat::ai::providers::OnnxProvider;

fn test_model_path() -> Option<String> { std::env::var("RIA_TEST_ONNX_MODEL").ok() }

#[test]
fn onnx_forward_probe_success_or_framework_response() {
    let Some(model_path) = test_model_path() else { eprintln!("SKIP: set RIA_TEST_ONNX_MODEL for inference tests"); return; };
    assert!(std::path::Path::new(&model_path).exists());
    let cfg = InferenceConfig { model_path: model_path.clone(), execution_provider: ExecutionProvider::Cpu, ..InferenceConfig::default() };
    let mut provider = OnnxProvider::new(cfg).unwrap();
    provider.load_model().unwrap();

    let user = ChatMessage { id: "u1".into(), content: "Test ONNX working?".into(), role: MessageRole::User, timestamp: chrono::Utc::now(), model_used: None, inference_time: None };
    let resp = provider.generate_response(&[user]).unwrap();
    // The response should mention tokens or success markers
    assert!(resp.contains("ONNX") || resp.contains("tokens") || resp.contains("forward pass"), "Unexpected response: {resp}");
}

#[test]
fn onnx_multiple_sequential_calls() {
    let Some(model_path) = test_model_path() else { eprintln!("SKIP: set RIA_TEST_ONNX_MODEL for inference tests"); return; };
    let cfg = InferenceConfig { model_path: model_path.clone(), execution_provider: ExecutionProvider::Cpu, ..InferenceConfig::default() };
    let mut provider = OnnxProvider::new(cfg).unwrap();
    provider.load_model().unwrap();

    for i in 0..3 {
        let user = ChatMessage { id: format!("u{i}"), content: format!("hello iteration {i}"), role: MessageRole::User, timestamp: chrono::Utc::now(), model_used: None, inference_time: None };
        let resp = provider.generate_response(&[user]).unwrap();
        assert!(resp.len() > 10, "Short response at iteration {i}");
    }
}

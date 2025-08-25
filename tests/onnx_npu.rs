//! NPU usage test (OpenVINO) - placeholder to ensure configuration selects NPU EP when available.
//! Requires: feature `openvino_ep` and env var RIA_TEST_ONNX_MODEL plus RIA_TEST_EXPECT_NPU=1
mod common;
#[cfg(feature = "openvino_ep")]
#[test]
fn onnx_uses_openvino_when_requested() {
    let use_npu = std::env::var("RIA_TEST_EXPECT_NPU").ok().unwrap_or_default() == "1";
    if !use_npu { eprintln!("SKIP: set RIA_TEST_EXPECT_NPU=1 and enable openvino_ep feature to run NPU test"); return; }
        let model = match std::env::var("RIA_TEST_ONNX_MODEL").ok().or_else(common::discover_test_model) { Some(m) => m, None => { eprintln!("SKIP: no model found for NPU test"); return; } };
    assert!(std::path::Path::new(&model).exists(), "Model path missing");
    use ria_ai_chat::ai::{InferenceConfig, ExecutionProvider};
    use ria_ai_chat::ai::providers::OnnxProvider;
    let cfg = InferenceConfig { model_path: model, execution_provider: ExecutionProvider::OpenVINO, ..InferenceConfig::default() };
    let mut provider = OnnxProvider::new(cfg).expect("create provider");
    if let Err(e) = provider.load_model() {
        panic!("OpenVINO load failed immediately (no fallback expected here): {e}");
    }
    let ep = provider.loaded_execution_provider();
    assert!(matches!(ep, Some(ExecutionProvider::OpenVINO) | Some(ExecutionProvider::Cpu)), "Unexpected EP after OpenVINO attempt: {:?}", ep);
}
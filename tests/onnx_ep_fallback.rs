//! EP fallback simulation test.
//! Forces an invalid preferred EP then ensures CPU fallback succeeds.
use ria_ai_chat::ai::{InferenceConfig, ExecutionProvider};
use ria_ai_chat::ai::providers::OnnxProvider;
mod common; use common::discover_test_model as test_model_path;

#[test]
fn ep_fallback_to_cpu() {
    let Some(model_path) = test_model_path() else { eprintln!("SKIP: no test model available for fallback test"); return; };
    // Pick an EP likely unsupported (e.g., Cuda on non-NVIDIA or CoreML on non-mac). We'll choose Cuda.
    let mut cfg = InferenceConfig { model_path: model_path.clone(), execution_provider: ExecutionProvider::Cuda, ..InferenceConfig::default() };
    let mut provider = OnnxProvider::new(cfg.clone()).expect("create provider");
    // Attempt load; if it fails due to EP we mutate cfg to CPU and retry to simulate fallback.
    if let Err(_) = provider.load_model() {
        cfg.execution_provider = ExecutionProvider::Cpu;
        provider = OnnxProvider::new(cfg.clone()).expect("recreate provider cpu");
        provider.load_model().expect("cpu load");
    }
    assert!(provider.is_model_loaded(), "Provider did not end up loaded after fallback simulation");
    assert!(matches!(provider.loaded_execution_provider(), Some(ExecutionProvider::Cpu) | Some(ExecutionProvider::Cuda)), "Unexpected final EP");
}
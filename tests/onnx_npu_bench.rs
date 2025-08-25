//! Local NPU benchmark tests.
//! Run with: RIA_TEST_ONNX_MODEL=path RIA_TEST_EXPECT_NPU=1 cargo test --tests --features openvino_ep -- --nocapture
//! Not a micro-benchmark; just coarse timing to compare CPU vs OpenVINO (NPU) load + single probe.
use std::time::Instant;
mod common;

#[cfg(feature = "openvino_ep")]
fn bench_once(model: &str, ep: ria_ai_chat::ai::ExecutionProvider, iters: u32) -> (f64, f64, f64) {
    use ria_ai_chat::ai::{InferenceConfig, ExecutionProvider};
    use ria_ai_chat::ai::providers::OnnxProvider;
    let cfg = InferenceConfig { model_path: model.to_string(), execution_provider: ep, warmup_iterations: 2, profiling: true, ..InferenceConfig::default() };
    let mut provider = OnnxProvider::new(cfg).expect("create provider");
    let t0 = Instant::now();
    provider.load_model().expect("load");
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    // Build minimal chat message
    use ria_ai_chat::ai::{ChatMessage, MessageRole, AIProvider};
    let prompt = ChatMessage { id: "1".into(), content: "hello benchmark".into(), role: MessageRole::User, timestamp: chrono::Utc::now(), model_used: None, inference_time: None };
    let t1 = Instant::now();
    for _ in 0..iters { let _ = provider.generate_response(&[prompt.clone()]).expect("response"); }
    let total_infer_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let per_iter = total_infer_ms / (iters as f64);
    (load_ms, total_infer_ms, per_iter)
}

#[cfg(feature = "openvino_ep")]
#[test]
fn npu_benchmark_compare_cpu_vs_openvino() {
    let model = match std::env::var("RIA_TEST_ONNX_MODEL").ok().or_else(common::discover_test_model) { Some(m) => m, None => { eprintln!("SKIP: no model found for benchmark"); return; } };
    if std::env::var("RIA_TEST_EXPECT_NPU").ok().as_deref() != Some("1") { eprintln!("SKIP: set RIA_TEST_EXPECT_NPU=1 for NPU benchmark"); return; }
    assert!(std::path::Path::new(&model).exists(), "Model file missing");
    use ria_ai_chat::ai::ExecutionProvider;
    let iters = 10; // coarse loops
    let (cpu_load, cpu_total, cpu_per) = bench_once(&model, ExecutionProvider::Cpu, iters);
    let (ov_load, ov_total, ov_per) = bench_once(&model, ExecutionProvider::OpenVINO, iters);
    let cpu_tps = 1000.0 / cpu_per.max(0.0001);
    let ov_tps = 1000.0 / ov_per.max(0.0001);
    println!("BENCH iters={iters} | CPU load {:.1}ms total {:.1}ms per {:.2}ms ({:.1} it/s) | OpenVINO load {:.1}ms total {:.1}ms per {:.2}ms ({:.1} it/s)",
        cpu_load, cpu_total, cpu_per, cpu_tps, ov_load, ov_total, ov_per, ov_tps);
    // Basic sanity assertions: both paths worked
    assert!(cpu_load > 0.0 && ov_load > 0.0);
    assert!(cpu_total > 0.0 && ov_total > 0.0);
}

#[cfg(feature = "openvino_ep")]
#[test]
fn npu_profile_contains_device_hint() {
    let expect = std::env::var("RIA_TEST_EXPECT_NPU").ok().unwrap_or_default() == "1";
    if !expect { eprintln!("SKIP: set RIA_TEST_EXPECT_NPU=1 for profile test"); return; }
    let model = match std::env::var("RIA_TEST_ONNX_MODEL").ok().or_else(common::discover_test_model) { Some(m) => m, None => { eprintln!("SKIP: no model found for profile test"); return; } };
    assert!(std::path::Path::new(&model).exists());
    use ria_ai_chat::ai::{InferenceConfig, ExecutionProvider};
    use ria_ai_chat::ai::providers::OnnxProvider;
    let cfg = InferenceConfig { model_path: model, execution_provider: ExecutionProvider::OpenVINO, profiling: true, warmup_iterations: 1, ..InferenceConfig::default() };
    let mut provider = OnnxProvider::new(cfg).unwrap();
    provider.load_model().unwrap();
    // Locate temp profile file
    let profile_path = std::env::temp_dir().join("ria_onnx_profile.txt");
    if profile_path.exists() {
        let content = std::fs::read_to_string(&profile_path).unwrap_or_default();
        assert!(content.contains("provider=OpenVINO") || content.contains("OpenVINO"), "profile missing OpenVINO marker: {content}");
        if expect { assert!(content.contains("NPU"), "expected NPU token in profile: {content}"); }
    } else {
        eprintln!("WARN: profiling file not found (non-fatal)");
    }
}

#[cfg(not(feature = "openvino_ep"))]
#[test]
fn npu_benchmark_feature_disabled() {
    eprintln!("SKIP: openvino_ep feature not enabled");
}
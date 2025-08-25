# 🤖 RIA AI Chat

Cross‑platform Rust desktop AI chat app (egui + ONNX Runtime) with modern animated UI, local model management, execution‑provider detection, streaming simulation, accessibility, and intelligent fallback responses when no model is loaded.

> **Current Status (Aug 2025)**: UI, model management (local + remote with resumable downloads), tokenizer, execution provider selection, minimal ONNX session loading + forward probe, simulated streaming, intelligent demo fallback, notifications & accessibility all working. Full generative decoding (token-by-token real model output) is scaffolded but not yet implemented.

## ✨ Key Features

- **🎨 Modern Animated UI** (egui) with focus rings, accessibility & responsive layout
- **💬 Intelligent Chat** with contextual demo fallback (works even before model load)
- **🧠 Model Management**: Local + remote catalog, resumable + checksum‑verifiable downloads, aux file (tokenizer) fetch, auto directory scan
- **📡 Streaming Simulation**: Chunked UI updates (scaffold for real token streaming)
- **🧩 Tokenization**: Pluggable (custom simple tokenizer + optional HF JSON load)
- **⚙️ Execution Providers**: CPU, CUDA, DirectML, CoreML, OpenVINO, QNN (scaffold), NNAPI (scaffold) with NPU preference flag
- **🔍 System Detection**: CPU / GPU / NPU capability + OS info surfaced
- **🚦 Minimal Forward Probe**: Attempts real ONNX `input_ids`(+`attention_mask`) run; falls back to framework message if unsupported
- **📦 Resumable Downloads** with progress & optional SHA256 validation
- **🔔 Notification System**: Success / Error / Info / Loading with timeouts & actions
- **⌨️ Keyboard Shortcuts & Focus Navigation** (Tab / Shift+Tab, Ctrl combos)
- **🛠 Config Persistence**: JSON config, model paths, window size/position
- **🔄 Cross-Platform**: Windows, macOS, Linux (tested primary focus: Windows)

## 🚀 Quick Start

### Prerequisites

- **Rust**: Install from [rustup.rs](https://rustup.rs/)
- **Git**: For cloning the repository

### Installation & First Run

1. **Clone and run**

```powershell
git clone <repository-url>
cd ria
cargo run --release
```

2. **Load or download a model**

   - Click **🧠 Models**
   - Remote tab: choose a model (e.g. TinyLlama / Phi / Qwen) → 📥 Download (resumable)
   - Or place an existing `*.onnx` file into the `models/` folder (auto-detected)
   - Select it under Local Models and press Load

3. **Chat**

   - Without a model you still get intelligent demo responses
   - With a model loaded you get ONNX forward probe confirmation (full decoding upcoming)

### ⚡ Quick recap

No extra services required. Works offline after model download.

## 🧠 Model Management System

The app includes a comprehensive model management system with built-in popular models:

### 📁 Local Models

- Shows all `.onnx` files in your `./models/` directory
- Displays model size, type, quantization info
- One-click selection with radio buttons
- Supported execution providers indicator
- Delete functionality for cleanup

### 🌐 Remote Models

Pre-configured popular models ready for download. On Intel NPU systems, a curated catalog (OpenVINO / NPU‑friendly) loads automatically:

| Model | Size | Type | Best For |
|-------|------|------|----------|
| **Phi-3-mini-4k-instruct (INT4 Mobile)** | 2.4GB | Chat | Copilot+/Intel NPU baseline |
| **TinyLlama-1.1B-Chat (INT8 ONNX)** | 1.5GB | Chat | Quick functionality test |
| **Qwen2-0.5B-Instruct (INT8 ONNX)** | 1.0GB | Chat | Ultra-fast baseline |

You can customize the catalog at `assets/model_catalog/intel_npu_onnx.json`.

### 🔄 Automatic & Enhanced Features

- Directory scan on startup / after downloads
- Resumable HTTP range downloads (`.onnx.part` continuation)
- Optional SHA256 verification (when catalog provides hashes)
- Aux file (e.g., tokenizer JSON) download support
- Demo fallback provider if no model active
- Real-time progress + speed estimate (KB/s)
- Basic model heuristic analysis (type, quantization)

## ⚙️ Configuration

### Execution Providers

The app automatically detects available compute devices:

- **CPU**: Always available (default)
- **CUDA**: NVIDIA GPUs (requires CUDA toolkit)
- **DirectML**: Windows GPUs (built-in on Windows 10+)
- **CoreML**: macOS Metal GPUs
- **OpenVINO**: Intel CPUs and GPUs
- **QNN**: Qualcomm NPU (ARM64 Windows)

### Settings File

Configuration is stored in:
- **Windows**: `%APPDATA%\\ria-ai-chat\\config.json`
- **macOS**: `~/Library/Application Support/ria-ai-chat/config.json`
- **Linux**: `~/.config/ria-ai-chat/config.json`

### Intel NPU Priority

This app is optimized to work on Windows Copilot+ PCs with Intel NPU cores via OpenVINO. On supported machines:
- The model catalog includes a curated list of Intel NPU–friendly ONNX models (assets/model_catalog/intel_npu_onnx.json).
- The runtime will favor CPU + OpenVINO/NPU paths when available.
- You can still run on CPU-only systems; GPU paths (CUDA/DirectML/CoreML) are optional.

## 🏗️ Architecture


```text
src/
├── ai/                 # AI inference engine
│   ├── inference.rs    # Main inference logic
│   ├── providers.rs    # Execution provider management
│   ├── models.rs       # Model management
│   └── mod.rs          # AI module exports
├── ui/                 # User interface
│   ├── app.rs          # Main application
│   ├── chat.rs         # Chat components
│   ├── settings.rs     # Settings UI
│   ├── components.rs   # Reusable UI components
│   └── mod.rs          # UI module exports
├── config/             # Configuration management
│   └── mod.rs          # App configuration
├── utils/              # Utilities
│   ├── system.rs       # System information
│   ├── files.rs        # File operations
│   └── mod.rs          # Utility functions
└── main.rs             # Application entry point
```

### Key Technologies

- **[egui](https://github.com/emilk/egui)**: Immediate mode GUI framework
- **[ONNX Runtime](https://onnxruntime.ai/)**: Cross-platform ML inference
- **[OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)**: Intel NPU acceleration (planned integration path)
- **[Tokio](https://tokio.rs/)**: Async runtime for non-blocking operations
- **[Serde](https://serde.rs/)**: Serialization for configuration
- **[Tracing](https://tracing.rs/)**: Structured logging

## 🔧 Development

### Building from Source

1. **Install Rust toolchain**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

1. **Clone and build**
```bash
git clone <repository-url>
cd ria
cargo build
```

1. **Run in development mode**
```bash
cargo run
```

### Development Features

- Logging / tracing (set `RUST_LOG=debug`)
- Structured error handling (`anyhow`, `thiserror`)
- Modular providers (demo + ONNX)
- Streaming scaffold via chunked response channel

### Performance Notes

- Release profile: `lto`, `panic=abort`, single codegen unit
- Inference session optimization: Graph Level3, limited intra threads
- Async runtime (Tokio) for downloads & streaming
- Lightweight tokenizer (fallback) keeps startup fast

## 🎨 UI & Accessibility

### Modern Design Elements

- Smooth animations, dark theme (light/system planned), responsive layout
- Message bubbles with timestamps & streaming preview
- Typing indicator via streaming buffer
- Download progress + notifications
- Focus manager + keyboard navigation rings
- Accessible labels & hover hints

### Customization

- **Themes**: Dark, Light, and System themes
- **Animation Quality**: Low, Medium, High settings
- **Font Options**: Support for custom fonts (future)
- **Window Settings**: Remembers size and position

## ✅ Current Status & Achievements

### ✨ Working Now

- Cross-platform GUI (Windows primary focus currently)
- Local & remote model system w/ resume
- Intelligent fallback chat (demo provider)
- Execution provider detection & preference (NPU-first option)
- ONNX session load + minimal forward probe
- Simulated streaming in UI
- Tokenization (custom + optional HF JSON)
- Config persistence & model metadata
- Notifications + accessibility

### 🔄 Framework Components Ready

- Provider abstraction + dynamic registration
- Streaming channel pattern (replaceable with real token decode)
- Minimal forward pass detection logic
- System info & device detection
- File + download utilities (resume, hash verify)

## 🔮 Planned Improvements

### 🎯 Phase 1 (Next)

- Full generative decoding (logits -> sampling)
- True token streaming (incremental forward passes)
- Extended tokenizer integration (BPE / SentencePiece)
- Model capability introspection & validation
- Better memory usage diagnostics

### 🌐 Phase 2

- External API / cloud provider plugins
- File & document tooling (RAG groundwork)
- Plugin architecture & sandboxing
- Rich telemetry (optional)

### 🚀 Phase 3

- RAG (vector index + retrievers)
- Multi-modal (image, audio) pipelines
- Voice interface (STT / TTS)
- Mobile ports
- Encrypted cloud sync

## 📊 System Requirements

### Minimum Requirements

- **OS**: Windows 10, macOS 10.15, Ubuntu 18.04+
- **RAM**: 4GB (8GB recommended)
- **Storage**: 500MB free space
- **CPU**: Any x64 or ARM64 processor

### Recommended for AI Inference

- **RAM**: 16GB+ (for larger models)
- **GPU/NPU**: NVIDIA RTX 3060+ or Intel NPU (OpenVINO) or equivalent
- **Storage**: SSD for faster model loading
- **CPU**: Multi-core processor (8+ cores recommended)

### NPU Support

- **Windows**: ARM64 with Qualcomm NPU
- **Intel**: Core Ultra processors with AI acceleration
- **Apple**: M-series chips (future CoreML optimization)

## 🐛 Troubleshooting

### Common Issues

#### Model Loading Fails

```bash
# Check model file exists and is valid ONNX format
file models/your-model.onnx
```

#### CUDA Not Detected

```bash
# Verify NVIDIA driver and CUDA toolkit
nvidia-smi
```

#### High Memory Usage

- Use smaller models (1B-3B parameters)
- Enable memory optimization in settings
- Close other applications

#### Slow Performance

- Try alternate execution provider
- Pick smaller / more quantized model (INT8/INT4)
- Lower animation quality (Settings)

### Logs and Debugging

Logs are written to:

- **Console**: Standard output (development)
- **File**: Application data directory (future)

Enable debug logging:

```bash
RUST_LOG=debug cargo run
```

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+Enter | Send message |
| Ctrl+N | New chat session |
| Ctrl+M | Toggle Models panel |
| Ctrl+, | Toggle Settings panel |
| Ctrl+D | Clear input box |
| Ctrl+H | Show keyboard help notification |
| Ctrl+K | Clear notifications |
| Tab / Shift+Tab | Cycle focus |
| Esc | Close panel / clear focus |

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ONNX Runtime team** for the excellent ML runtime
- **egui community** for the amazing immediate mode GUI
- **Rust community** for the fantastic ecosystem
- **AI model creators** for open-source models

---

Made with ❤️ and Rust 🦀

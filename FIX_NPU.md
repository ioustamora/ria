# 🔧 Fix NPU Loading Issue

## Problem
- **NPU loading stuck at 0%**
- **ONNX Runtime version mismatch**
- System has v1.17.1, app needs v1.22+

## Solutions (Pick One)

### Option 1: Update ONNX Runtime System-wide
```powershell
# Remove old version
pip uninstall onnxruntime onnxruntime-gpu

# Install latest with NPU support
pip install onnxruntime --upgrade
pip install onnxruntime-openvino  # For Intel NPU

# Or use winget
winget install Microsoft.ONNXRuntime
```

### Option 2: Use Conda Environment
```powershell
conda create -n ria python=3.11
conda activate ria
conda install onnxruntime=1.22
```

### Option 3: Quick Test (Bundled Runtime)
```powershell
# Download compatible ONNX Runtime
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-x64-1.22.0.zip -o onnxruntime.zip
# Extract to current directory
# Run RIA from same folder
```

## Verify Fix
```powershell
python -c "import onnxruntime; print(onnxruntime.__version__)"
# Should show 1.22.x or higher
```

## After Fix
- ✅ NPU cores will load at 100%  
- ✅ Real AI model inference works
- ✅ Hardware acceleration enabled
- ✅ Exit demo mode, enter full AI mode

**Demo mode still works perfectly for learning while you fix this!**
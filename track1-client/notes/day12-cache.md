# Day 12: Model Cache and GPU-Serving Image

## 1. Core Architecture Contract
- **Decoupled Weights**: Model artifacts ko Docker image ke andar pack karne ke bajaye host volume (`artifacts/day12/model_cache`) par rakha gaya aur container me `/app/cache` par mount kiya gaya.
- **Environment Variables**:
  - `MODEL_CACHE_DIR=/app/cache`
  - `HF_HOME=/app/cache/huggingface`
  - `TORCH_HOME=/app/cache/torch`
- **Container Healthcheck**: `HEALTHCHECK` directive add kiya gaya jo `http://localhost:8765/health` ko probe karta hai aur container status ko `(healthy)` mark karta hai.

## 2. Verification Evidence
1. **Cache Persistence**:
   - Host par `weights.bin` file banayi gayi.
   - Container ko band karke dubara `docker-compose.gpu.yml` se chalaya gaya.
   - Naye container ne wahi cache bina re-download kiye execute aur read kiya.
2. **Health Probe**:
   - `GET /health` route ne `{"status": "healthy", "cache_mounted": true, "gpu_available": false}` return kiya.

## 3. Failure & Diagnosis Test (Hardware Boundary)
- **Command Tested**: `docker run --rm --gpus all track1-client:day12`
- **Observed Error**: `could not select device driver "" with capabilities: [[gpu]]`
- **Root Cause & Resolution**: Local laptop AMD Radeon graphics par hai (NVIDIA driver nahi hai). Container architecture graceful fallback provide karti hai CPU mode me, jabke GPU execution cloud runtime (Google Colab T4) par dedicate hoti hai.

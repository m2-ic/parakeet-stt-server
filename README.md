# Parakeet STT Server

FastAPI server for **Parakeet-TDT-0.6B-v3** (NVIDIA NeMo) speech-to-text, optimized for multi-GPU inference.

## Performance (8x H100 80GB)

| Req/s | Avg Latency | P95 |
|-------|-------------|-----|
| 10 | 106ms | 128ms |
| 30 | 110ms | 134ms |
| 50 | 115ms | 154ms |
| 70 | 129ms | 228ms |
| 80 | 393ms | 763ms |

Stable up to **~70 req/s** with random audio files (10-20s duration), no batching.

## Setup

### 1. Install dependencies

```bash
# PyTorch + CUDA (adapt to your CUDA version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# NeMo ASR
pip install nemo_toolkit[asr]

# Server
pip install fastapi uvicorn python-multipart
```

### 2. Start the server

```bash
# Single GPU
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python3 -m uvicorn server:app --host 0.0.0.0 --port 8000

# Multi-GPU (1 worker per GPU, recommended)
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 --workers 8
```

The model (~600MB) is downloaded automatically on first startup from HuggingFace.

### 3. Test

```bash
# Health check
curl http://localhost:8000/health

# Transcribe
curl -X POST http://localhost:8000/transcribe -F "file=@audio.wav"
```

## API

### `GET /health`

Returns server status, GPU info, worker PID.

### `POST /transcribe`

Upload an audio file (WAV/MP3/FLAC/OGG), returns transcription.

**Request:** `multipart/form-data` with `file` field.

**Response:**
```json
{
  "text": "transcribed text here",
  "duration_sec": 10.0,
  "processing_time_ms": 65.3,
  "gpu_id": 0
}
```

## Load Testing

```bash
# Install aiohttp
pip install aiohttp

# Run load test (50 sequential + ramp to find breaking point)
python3 loadtest.py
```

Edit `AUDIO_DIR` and `rates` in `loadtest.py` to customize.

## Architecture

- Each uvicorn worker claims a GPU via filesystem lock (`/tmp/parakeet_gpu_assignments/`)
- Requests are load-balanced across workers by uvicorn
- No batching: each request is processed individually on its assigned GPU
- Manual inference pipeline (preprocessor → encoder → RNNT decoder) for maximum control

## Workarounds (PyTorch 2.10 + NeMo)

Three fixes are required for NeMo's Parakeet to work with PyTorch 2.10:

1. **`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`** — NeMo model loading crashes without this env var (PyTorch 2.10 changed the default for `torch.load` to `weights_only=True`, NeMo hasn't adapted yet)

2. **CUDA graphs disabled** — NeMo's RNNT greedy decoder uses CUDA graphs by default, which are broken with PyTorch 2.10 API changes. Without disabling them, the first inference call crashes:
   ```python
   decoding_cfg["greedy"]["use_cuda_graph_decoder"] = False
   decoding_cfg["greedy"]["loop_labels"] = False
   model.change_decoding_strategy(OmegaConf.create(decoding_cfg))
   ```

3. **Manual inference pipeline** — Instead of `model.transcribe()`, we run the pipeline manually (preprocessor → encoder → RNNT decoder). This avoids additional NeMo-internal issues and gives direct control over tensor placement:
   ```python
   processed, proc_len = model.preprocessor(input_signal=waveform, length=length)
   encoded, encoded_len = model.encoder(audio_signal=processed, length=proc_len)
   result = model.decoding.rnnt_decoder_predictions_tensor(encoded, encoded_len, return_hypotheses=True)
   ```

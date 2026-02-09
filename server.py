"""
FastAPI server for Parakeet-TDT-0.6B-v3 Speech-to-Text.
Each worker gets assigned a GPU based on its PID.

Run with:
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 --workers 8
"""

import os
import time
import tempfile
from contextlib import asynccontextmanager
from typing import List

import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, HTTPException

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

MODEL = None
GPU_ID = None


def load_model(device: str):
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf

    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    model = model.to(device)
    model.eval()
    decoding_cfg = OmegaConf.to_container(model.cfg.decoding, resolve=True)
    decoding_cfg["greedy"]["use_cuda_graph_decoder"] = False
    decoding_cfg["greedy"]["loop_labels"] = False
    model.change_decoding_strategy(OmegaConf.create(decoding_cfg))
    return model


def transcribe_single(audio_path: str) -> str:
    device = f"cuda:{GPU_ID}"
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform.to(device)
    length = torch.tensor([waveform.shape[1]], dtype=torch.long, device=device)

    with torch.no_grad():
        processed, proc_len = MODEL.preprocessor(input_signal=waveform, length=length)
        enc_out = MODEL.encoder(audio_signal=processed, length=proc_len)
        if isinstance(enc_out, tuple):
            encoded, encoded_len = enc_out
        else:
            encoded, encoded_len = enc_out, proc_len

        result = MODEL.decoding.rnnt_decoder_predictions_tensor(
            encoded, encoded_len, return_hypotheses=True
        )
        hyps = result[0] if isinstance(result, tuple) else result
        if hyps and hasattr(hyps[0], "text"):
            return hyps[0].text
        elif hyps:
            return str(hyps[0])
    return ""


def get_audio_duration(audio_path: str) -> float:
    waveform, sr = torchaudio.load(audio_path)
    return waveform.shape[1] / sr


async def save_upload(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or ".wav")[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await upload.read()
        tmp.write(content)
        tmp.close()
        return tmp.name
    except Exception:
        tmp.close()
        os.unlink(tmp.name)
        raise


# Worker-local GPU assignment file
_GPU_ASSIGNMENT_DIR = "/tmp/parakeet_gpu_assignments"


def _claim_gpu() -> int:
    """Each worker claims the next available GPU via filesystem lock."""
    os.makedirs(_GPU_ASSIGNMENT_DIR, exist_ok=True)
    num_gpus = torch.cuda.device_count()
    for gpu_id in range(num_gpus):
        lock_file = os.path.join(_GPU_ASSIGNMENT_DIR, f"gpu_{gpu_id}.lock")
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return gpu_id
        except FileExistsError:
            # Check if the PID in the lock file is still alive
            try:
                with open(lock_file) as f:
                    pid = int(f.read().strip())
                os.kill(pid, 0)  # Check if process exists
            except (ProcessLookupError, ValueError, OSError):
                # Stale lock, reclaim it
                os.unlink(lock_file)
                fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return gpu_id
    raise RuntimeError(f"No GPU available (all {num_gpus} claimed)")


def _release_gpu(gpu_id: int):
    lock_file = os.path.join(_GPU_ASSIGNMENT_DIR, f"gpu_{gpu_id}.lock")
    try:
        os.unlink(lock_file)
    except FileNotFoundError:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, GPU_ID
    # Clean stale locks on first startup
    GPU_ID = _claim_gpu()
    device = f"cuda:{GPU_ID}"
    print(f"Worker PID {os.getpid()} → GPU {GPU_ID} ({torch.cuda.get_device_name(GPU_ID)})")
    print(f"  Loading model on {device}...")
    MODEL = load_model(device)
    print(f"  Worker PID {os.getpid()} ready on GPU {GPU_ID}.")
    yield
    _release_gpu(GPU_ID)
    MODEL = None
    torch.cuda.empty_cache()


app = FastAPI(title="Parakeet STT Server", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok" if MODEL is not None else "loading",
        "model": "nvidia/parakeet-tdt-0.6b-v3",
        "gpu_id": GPU_ID,
        "gpu": torch.cuda.get_device_name(GPU_ID) if GPU_ID is not None else None,
        "pid": os.getpid(),
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(503, "Model not loaded yet")

    tmp_path = await save_upload(file)
    try:
        duration = get_audio_duration(tmp_path)
        t0 = time.perf_counter()
        text = transcribe_single(tmp_path)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    finally:
        os.unlink(tmp_path)

    return {
        "text": text,
        "duration_sec": round(duration, 3),
        "processing_time_ms": round(elapsed_ms, 1),
        "gpu_id": GPU_ID,
    }

---
mode: agent
---

# LTX2 API Server — Repair & Debugging Guide

Use this skill when working on `/home/mahesh/projects/animation/LTX2-api-server`.

## Workspace Layout

| Path | Purpose |
|---|---|
| `/home/mahesh/projects/animation/LTX2-api-server/server.py` | FastAPI inference server |
| `/home/mahesh/projects/animation/LTX-2/` | LTX-Video 2.3 source (editable pip install) |
| `models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors` | Main 22B distilled checkpoint |
| `models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | Spatial upsampler checkpoint |
| `outputs/` | Generated `.mp4` files |
| `.env` | Contains `LTX_API_KEY` |
| `venv/` | Python virtual environment |

## Hardware

- **GPU 0**: Tesla T4, 15 360 MiB, sm75 (Turing) — primary inference GPU  
- **GPU 1**: Titan Xp, 12 288 MiB — idle; model-splitting across both GPUs is NOT viable  
- **PyTorch**: 2.6.0+cu124  
- **xformers**: NOT installed  
- **FlashAttention 2/3**: NOT installed  
- **SDPA backends available**: math_sdp ✅, mem_efficient_sdp ✅ (float16 only on sm75), flash_sdp ❌ (requires sm80+)

---

## Architecture Overview

The pipeline has two diffusion stages:

1. **Stage 1** — half-resolution (512×288 for a 1024×576 target), 8 denoising steps  
2. **Spatial upsample** — `VideoUpsampler` doubles H/W via a Conv3D-based model  
3. **Stage 2** — full-resolution (1024×576), 3 denoising steps  
4. **Decode** — `VideoDecoder` + `AudioDecoder`

The diffusion transformer is an FP8-cast-quantised 22B model kept on GPU 0.  
All auxiliary components (`ImageConditioner`, `VideoUpsampler`, `VideoDecoder`, `AudioDecoder`) live on CPU and are moved to GPU only for their specific step via `ModelManager._on_gpu()`.

---

## Known Bugs (all fixed in current codebase)

### Bug 1 — DEFAULT_HEIGHT wrong
`DEFAULT_HEIGHT` was 512; must be **448** to keep aspect ratio for half-res stage.

### Bug 2 — `_on_gpu` not a context manager
`_on_gpu()` must use `@contextmanager` and `yield`.

### Bug 3 — `streaming_prefetch_count` missing
`DiffusionStage` call must pass `streaming_prefetch_count=2`; omitting it causes slow disk I/O during inference.

### Bug 4 — `_split_av_encoding` wrong slice indices
The combined API encoding must be split into `(video_connector_dim=4096, audio_connector_dim=2048)`.  
The 22B model has `caption_proj_before_connector=True` so there is **no** `caption_projection` inside the transformer.

### Bug 5 — Builder-wrapper `_device` not swapped in `_on_gpu`
`VideoUpsampler`, `VideoDecoder`, `AudioDecoder`, `ImageConditioner` are plain Python objects with a `_device` attribute — they must be detected via `hasattr(m, '_device')` and updated, not via `isinstance(m, torch.nn.Module)`.

### Bug 6 — AudioDecoder must run in float32 on CPU
`AudioDecoder` fails with bfloat16 on CPU (no CPU bfloat16 autocast support for cuDNN audio ops).  
Fix: save/restore `_audio_decoder._dtype`, set it to `torch.float32` around the decode call.

### Bug 7 — Stage 2 OOM: 10.12 GiB attention matrix *(the critical one)*

**Root cause**: At 1024×576, 121 frames the transformer has ~10 677 video tokens.  
`scaled_dot_product_attention` tries to materialise a `[1, 24, 10677, 10677]` float32 score matrix = **10.12 GiB**.

`mem_efficient_sdp` (CUTLASS, O(N) memory) would avoid this, but on **sm75 (T4) it only works with float16 — not bfloat16**.  
The model must stay in **bfloat16** because `QuantizationPolicy.fp8_cast()` requires bfloat16 compute dtype; changing `self._dtype` to float16 breaks the FP8 upcast kernel and the upsampler's cuDNN Conv3D (`CUDNN_STATUS_NOT_INITIALIZED`).

**Fix** — cast q/k/v/mask to float16 inside `PytorchAttention.__call__` just for the SDPA call, then cast output back:

```python
# File: packages/ltx-core/src/ltx_core/model/transformer/attention.py
# Class: PytorchAttention.__call__  — after the mask reshape block, before SDPA:

orig_dtype = v.dtype
q = q.to(torch.float16)
k = k.to(torch.float16)
v = v.to(torch.float16)
if mask is not None:
    mask = mask.to(torch.float16)

out = torch.nn.functional.scaled_dot_product_attention(
    q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
)
out = out.to(orig_dtype)
```

This forces `mem_efficient_sdp` selection on T4 without changing the model's bfloat16 compute dtype.

### Bug 8 — Stage 1 latent not freed before Stage 2
After spatial upsample, the half-res `video_state` and `audio_state` must be deleted before Stage 2 to avoid holding both latents in VRAM simultaneously:

```python
s1_audio_latent = audio_state.latent   # keep reference on CUDA (needed by Stage 2)
del video_state, audio_state
gc.collect()
torch.cuda.empty_cache()
```
Pass `initial_latent=s1_audio_latent` to the Stage 2 audio `ModalitySpec`.

---

## Supported Resolutions

| Resolution | Frames | Feasible? | Notes |
|---|---|---|---|
| 512×288 | any | ✅ | Stage 1 only |
| 1024×576 | 121 (5 s) | ✅ | Needs Bug 7 fix |
| 1280×704 | 121 | ❌ | 23.63 GiB attention — impossible on T4 |

---

## Starting the Server

```bash
cd /home/mahesh/projects/animation/LTX2-api-server
/home/mahesh/projects/animation/LTX2-api-server/venv/bin/uvicorn server:app \
    --host 0.0.0.0 --port 8000 --workers 1 \
    > /tmp/ltx_server.log 2>&1 &
```

**Do NOT** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — it causes the allocator to retain ~14 GB of reserved pages between layers, amplifying OOM.

---

## Submitting a Job

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "...",
    "width": 1024, "height": 576,
    "num_frames": 121, "fps": 24,
    "seed": 42
  }'
# Returns {"job_id":"<uuid>","status":"queued"}

# Poll status:
curl -s http://localhost:8000/status/<job_id>

# Download result:
curl -s -o result.mp4 http://localhost:8000/download/<job_id>
```

Logs: `tail -f /tmp/ltx_server.log`

---

## Diagnostic Checklist

| Symptom | Likely Cause | Fix |
|---|---|---|
| `RuntimeError: Tried to allocate 10.12 GiB` in `scaled_dot_product_attention` | Bug 7 — math_sdp selected for bfloat16 on sm75 | Apply float16 cast in `PytorchAttention` |
| `RuntimeError: Expected query, key, and value to have the same dtype` (float32 vs float16) | `self._dtype=float16` + RMSNorm upcast | Revert dtype to bfloat16; use the cast-inside-SDPA approach |
| `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` in upsampler | `self._dtype=float16` breaks Conv3D on T4 | Revert dtype to bfloat16 |
| `ValueError: target_weight dtype must be bfloat16` | FP8 stochastic rounding kernel requires bfloat16 | Revert dtype to bfloat16 |
| Stage 2 OOM after Stage 1 succeeds | Half-res latent not freed | Add Bug 8 cleanup |
| `CUDA error: device-side assert triggered` in `GaussianNoiser` | Audio latent moved to CPU before Stage 2 | Keep `s1_audio_latent = audio_state.latent` on CUDA, only delete the Python object |
| Very slow generation (>10 min/step) | `streaming_prefetch_count` not set | Pass `streaming_prefetch_count=2` to `DiffusionStage` |

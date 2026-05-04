# LTX-2.3 Self-Hosted API Server

A self-hosted **FastAPI inference server** for [LTX-Video 2.3](https://github.com/Lightricks/LTX-Video) (22B distilled model) that runs locally on an NVIDIA Tesla T4 (16 GB VRAM). Text encoding is offloaded to the free LTX cloud API so the full GPU budget stays available for diffusion.

---

## Sample Outputs

**Demo video**

[![Demo](assets/thumb_demo.jpg)](assets/demo.mp4)

---

## Features

- **Text-to-video** generation via simple JSON API
- **Image-conditioned video** — anchor start frame, end frame, or both
- **Two-stage pipeline** — Stage 1 diffusion → spatial 2× upsampler → Stage 2 refinement
- **FP8 quantisation + layer streaming** to fit the 22B transformer in 16 GB VRAM
- **LTX-Desktop compatibility** — `/v1/*` drop-in for the official LTX Console API
- **Async job queue** — submit jobs and poll for results
- **T4/sm75 attention patch** — `float16` cast in `PytorchAttention` enables `mem_efficient_sdp` on Turing GPUs (vendored in `vendor/ltx-core`)

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA T4 (16 GB) | A10G / A100 (24+ GB) |
| CUDA | 11.8 | 12.1+ |
| RAM | 32 GB | 64 GB |
| Disk | 30 GB free | 50 GB free |

> **Note:** The T4 is the minimum viable GPU. Higher-end cards skip the FP8 quantisation and layer streaming and generate significantly faster.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/aiarcade/ltx2-api-server.git
cd ltx2-api-server
```

### 2. Create a Python virtual environment

Requires **Python 3.10–3.12**.

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Then install the vendored LTX packages (no external LTX-2 repo needed):

```bash
pip install --no-deps -e vendor/ltx-core
pip install --no-deps -e vendor/ltx-pipelines
```

> `server-start.sh` runs these two commands automatically, so on subsequent starts you only need the shell script.

### 4. Download the model weights

The model is hosted on Hugging Face. You need two files:

| File | Size | Description |
|------|------|-------------|
| `ltx-2.3-22b-distilled-1.1.safetensors` | ~23 GB | Main transformer checkpoint |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | ~1 GB | 2× spatial upsampler |

#### Option A — Hugging Face CLI (recommended)

```bash
pip install huggingface_hub
mkdir -p models/ltx-2.3

huggingface-cli download Lightricks/LTX-Video \
  ltx-2.3-22b-distilled-1.1.safetensors \
  --local-dir models/ltx-2.3

huggingface-cli download Lightricks/LTX-Video \
  ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --local-dir models/ltx-2.3
```

#### Option B — Python

```python
from huggingface_hub import hf_hub_download

for filename in [
    "ltx-2.3-22b-distilled-1.1.safetensors",
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
]:
    hf_hub_download(
        repo_id="Lightricks/LTX-Video",
        filename=filename,
        local_dir="models/ltx-2.3",
    )
```

Expected directory layout after download:

```
models/
└── ltx-2.3/
    ├── ltx-2.3-22b-distilled-1.1.safetensors      (~23 GB)
    └── ltx-2.3-spatial-upscaler-x2-1.1.safetensors (~1 GB)
```

### 5. Get a free LTX API key

Text encoding is handled by the LTX cloud API (free tier, no generation credits consumed — only the embedding endpoint is used).

1. Sign up at [console.ltx.video](https://console.ltx.video)
2. Create an API key
3. Copy `.env.example` → `.env` and paste your key:

```bash
cp .env.example .env
# Edit .env and set LTX_API_KEY=your_key_here
```

---

## Running the Server

```bash
./server-start.sh        # starts on port 8000
LTX_PORT=8080 ./server-start.sh   # custom port
```

```bash
./server-stop.sh         # graceful shutdown
```

The script will:
1. Auto-install vendored packages if needed
2. Load `.env`
3. Start uvicorn and wait for the health check

**Health check:**
```bash
curl http://localhost:8000/health
# {"status":"ok","model":"ltx-2.3","model_loaded":true,"pending_jobs":0}
```

Interactive API docs available at `http://localhost:8000/docs`.

---

## API Reference

### Text-to-Video

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a samurai warrior meditates in a moonlit bamboo forest",
    "width": 1024,
    "height": 576,
    "num_frames": 121,
    "fps": 24,
    "seed": 42
  }'
# → {"job_id": "abc123..."}
```

### Image-Conditioned Video (start + end frame)

```bash
curl -X POST http://localhost:8000/generate-frames \
  -F "prompt=a bunny dances by a sparkling river" \
  -F "width=1280" \
  -F "height=704" \
  -F "num_frames=241" \
  -F "fps=24" \
  -F "start_frame=@/path/to/first_frame.png;type=image/png" \
  -F "end_frame=@/path/to/last_frame.png;type=image/png"
# → {"job_id": "xyz789..."}
```

You can supply `start_frame` only, `end_frame` only, or both.

### Poll job status

```bash
curl http://localhost:8000/status/{job_id}
# → {"status":"generating","progress":0.62,...}
# status values: queued | generating | completed | failed
```

### Download result

```bash
curl -o output.mp4 http://localhost:8000/download/{job_id}
```

### Supported resolutions

| Resolution | Aspect | Notes |
|-----------|--------|-------|
| 512×320 | 16:10 | Fast test, ~2 min |
| 704×448 | ~16:9 | Default |
| 768×512 | 3:2 | |
| 1024×576 | 16:9 | Good quality |
| 1280×704 | ~16:9 | 720p, ~45 min on T4 |

All dimensions must be multiples of 64.

---

## Demo Client

A ready-made CLI client is included:

```bash
# Health check
python3 demo_client.py health

# Text-to-video
python3 demo_client.py text "a cat walks across a rooftop at sunset"

# Start frame anchoring
python3 demo_client.py start-frame "camera pans right" ./my_frame.png

# Both frames
python3 demo_client.py both-frames "smooth transition" ./start.png ./end.png

# Check status
python3 demo_client.py status abc123-...
```

Set `SERVER_URL` env var to point to a remote server:

```bash
SERVER_URL=http://192.168.1.10:8000 python3 demo_client.py health
```

---

## LTX-Desktop Compatibility

This server implements the LTX Console `/v1/*` API so you can point the **LTX-Desktop** Mac/Windows app directly at your local server:

1. Open LTX-Desktop settings
2. Set API URL to `http://<your-server-ip>:8000`
3. Set API Key to the value of `LTX_CONSOLE_API_KEY` in your `.env` (leave blank to skip auth)

---

## Configuration

All settings are controlled via environment variables (set in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LTX_API_KEY` | *(required)* | Free key from [console.ltx.video](https://console.ltx.video) |
| `LTX_MODEL_DIR` | `./models/ltx-2.3` | Path to model weights directory |
| `LTX_OUTPUT_DIR` | `./outputs` | Where generated videos are saved |
| `LTX_PORT` | `8000` | Server port (set in shell, not .env) |
| `LTX_STREAMING_PREFETCH` | `2` | GPU layer prefetch count (keep at 2 for 16 GB GPUs) |
| `LTX_MAX_JOBS` | `100` | Max jobs kept in history |
| `LTX_CONSOLE_API_KEY` | *(empty)* | Bearer token for `/v1/*` LTX-Desktop endpoints |

---

## Project Structure

```
ltx2-api-server/
├── server.py                  # FastAPI server — main entrypoint
├── server-start.sh            # Start script (handles venv + .env)
├── server-stop.sh             # Graceful stop script
├── demo_client.py             # CLI demo client
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── assets/                    # Sample output videos (tracked by git)
│   ├── sample_samurai.mp4
│   └── sample_5s.mp4
├── vendor/                    # Vendored LTX packages (patched for T4)
│   ├── ltx-core/              # Core model + T4 attention.py patch
│   └── ltx-pipelines/        # Two-stage distilled pipeline
├── models/                    # Model weights (not in git — download separately)
│   └── ltx-2.3/
└── outputs/                   # Generated videos (not in git)
```

### T4 Attention Patch

`vendor/ltx-core/src/ltx_core/model/transformer/attention.py` contains a critical fix for Turing (sm75) GPUs. PyTorch's `scaled_dot_product_attention` only selects `mem_efficient_sdp` for `float16` on sm75 — not `bfloat16`. The patch casts `q/k/v` to `float16` before the SDPA call and casts the output back, while keeping the rest of the model in `bfloat16` (required for FP8 quantisation).

---

## Troubleshooting

**CUDA out of memory**
- Reduce `width`/`height` or `num_frames`
- Ensure no other processes are using the GPU: `nvidia-smi`
- Restart the server to clear any residual allocations

**`LTX_API_KEY is not set` error**
- Copy `.env.example` to `.env` and add your key
- Make sure to start the server via `./server-start.sh` (which sources `.env`), not `uvicorn` directly

**Generation stuck at 0% for >5 minutes**
- The model is loading for the first time on the first job — this is normal (can take 5–10 min on T4)
- Watch `tail -f /tmp/ltx_server.log` for progress

**`mem_efficient_sdp` not selected on T4**
- The vendored `attention.py` patch handles this automatically
- If you reinstalled packages from upstream, re-run `pip install --no-deps -e vendor/ltx-core`

---

## License

The server code (`server.py`, `server-start.sh`, `server-stop.sh`, `demo_client.py`) is released under the **MIT License**.

The vendored packages in `vendor/` are from the [LTX-Video](https://github.com/Lightricks/LTX-Video) project by Lightricks and are subject to their original license. The only modification is the T4 attention patch in `vendor/ltx-core/src/ltx_core/model/transformer/attention.py`.

The LTX-Video model weights are subject to the [Lightricks LTX-Video License](https://huggingface.co/Lightricks/LTX-Video).

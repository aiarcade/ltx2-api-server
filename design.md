# LTX-2.3 Inference Server for Tesla T4

A self-hosted FastAPI inference server for [LTX-Video 2.3](https://github.com/Lightricks/LTX-2) optimized for **NVIDIA Tesla T4 (16GB VRAM)**. Text encoding is offloaded to the **free LTX API**, leaving the full 16GB available for local diffusion inference.

---

## How It Works

```
Your T4 (local)                  LTX Free API (cloud)
────────────────────             ─────────────────────
prompt (text)       ──────────►  Gemma-3-12B encoder
                    ◄──────────  embeddings (tensors)
Diffusion transformer
runs locally on T4
```

The Gemma-3-12B text encoder alone requires ~24GB at FP16 — far beyond the T4's capacity. By offloading text encoding to Lightricks' free API endpoint, the full 16GB is reserved for the diffusion transformer with FP8 quantization.

---

## Requirements

| Component | Requirement |
|---|---|
| GPU | NVIDIA Tesla T4 (16GB VRAM) |
| CUDA | 12.1+ |
| Python | 3.10+ |
| Disk | ~25GB (FP8 quantized weights) |
| OS | Linux (Ubuntu 20.04+) |
| LTX API Key | Free — [console.ltx.studio](https://console.ltx.studio) |

---

## Installation

### 1. Clone the LTX-2 repository

```bash
git clone https://github.com/Lightricks/LTX-2.git
cd LTX-2
python -m venv venv && source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e packages/ltx-core
pip install -e packages/ltx-pipelines
pip install accelerate transformers xformers fastapi uvicorn python-multipart
```

### 3. Download model weights

```bash
pip install huggingface_hub

python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Lightricks/LTX-2.3",
    local_dir="./models/ltx-2.3"
)
EOF
```

> The FP8 quantized weights are ~25GB. Full BF16 weights are ~44GB and will not fit on a T4.

### 4. Get a free LTX API key

1. Go to [console.ltx.studio](https://console.ltx.studio)
2. Sign up and generate an API key
3. Text encoding is **always free** — only video generation API calls are paid

### 5. Add your API key

```bash
export LTX_API_KEY="your_key_here"
```

Or create a `.env` file:

```
LTX_API_KEY=your_key_here
```

---

## Running the Server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
```

The server loads the model on startup. First launch may take 2–3 minutes to initialize.

---

## API Reference

### `POST /generate`

Submit a video generation job. Returns a `job_id` immediately; generation runs in the background.

**Request body:**

```json
{
  "prompt": "A fox running through a snowy forest, slow motion, cinematic",
  "negative_prompt": "blurry, low quality, watermark",
  "width": 704,
  "height": 480,
  "num_frames": 65,
  "fps": 30,
  "seed": 42
}
```

**Response:**

```json
{
  "job_id": "a3f2c1d4-...",
  "status": "queued"
}
```

---

### `GET /status/{job_id}`

Poll job status.

**Possible status values:** `queued` → `encoding_prompt` → `generating` → `completed` / `failed`

**Response (completed):**

```json
{
  "status": "completed",
  "url": "/download/a3f2c1d4-..."
}
```

---

### `GET /download/{job_id}`

Download the generated MP4 file.

```bash
curl -o output.mp4 http://localhost:8000/download/<job_id>
```

---

### `GET /health`

Check server status.

```json
{ "status": "ok", "model": "ltx-2.3" }
```

---

## Example Usage

```bash
# Submit a job
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat walking through a neon-lit city at night, cinematic 24mm lens",
    "num_frames": 65,
    "width": 704,
    "height": 480
  }'

# Poll until completed
curl http://localhost:8000/status/<job_id>

# Download
curl -o result.mp4 http://localhost:8000/download/<job_id>
```

---

## T4 Resolution & Frame Budget

Stay within these limits to avoid out-of-memory errors:

| Resolution | Max Frames | Duration @ 30fps | Est. Generation Time |
|---|---|---|---|
| 512 × 288 | 129 | ~4 sec | ~3–4 min |
| 704 × 480 | 97 | ~3 sec | ~4–5 min |
| 768 × 512 | 65 | ~2 sec | ~4–5 min |
| 1024 × 576 | 49 | ~1.5 sec | ~6–8 min |

Start at lower resolutions and increase once you've confirmed stable inference.

---

## Performance Tips

- **Use DistilledPipeline** — 8 diffusion steps instead of 40, significantly faster
- **FP8 quantization** — reduces model VRAM from ~44GB to ~12GB
- **FP16 dtype** — T4 (Turing/sm_75) lacks BF16 tensor cores; FP16 is the right choice
- **xFormers** — memory-efficient attention, important for Turing GPUs
- **Free text encoding API** — eliminates Gemma-3-12B (~24GB) from local VRAM entirely
- **TeaCache** — training-free 2× speed boost if supported by your ltx-pipelines version

---

## Troubleshooting

**OOM (Out of Memory) error:**
- Reduce `num_frames` first, then reduce resolution
- Call `torch.cuda.empty_cache()` between jobs
- Switch to `enable_sequential_cpu_offload()` for more aggressive offloading

**Text encoding API failure:**
- Check your `LTX_API_KEY` is set correctly
- Verify connectivity to `api.ltx.studio`
- The free tier has rate limits — add retry logic for high-throughput workloads

**Slow first generation:**
- Normal — CUDA kernels compile on first run
- Subsequent generations are significantly faster

**`bfloat16` errors:**
- T4 does not support BF16 tensor cores — ensure `torch_dtype=torch.float16`

---

## Project Structure

```
.
├── server.py          # FastAPI inference server
├── outputs/           # Generated MP4 files
├── models/
│   └── ltx-2.3/       # Downloaded model weights
└── README.md
```

---

## License

Model weights are subject to the [LTX-Video Model License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE). Commercial use is permitted — verify current terms before production deployment.

This server code is provided under the MIT License.

---

## Credits

- [LTX-2 by Lightricks](https://github.com/Lightricks/LTX-2) — open-source audio-video generation model
- Free text encoding endpoint provided by [LTX Studio](https://ltx.studio)
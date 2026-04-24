"""
LTX-2.3 Inference Server for Tesla T4
======================================
A self-hosted FastAPI server that runs LTX-2.3 video generation locally on an
NVIDIA Tesla T4 (16 GB VRAM). Text encoding is offloaded to the free LTX API,
keeping the full GPU budget available for the diffusion transformer.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import asyncio
import base64
import gc
import io
import logging
import os
import tempfile
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

import httpx
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ltx-server")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = os.getenv("LTX_MODEL_DIR", "./models/ltx-2.3")
OUTPUT_DIR = os.getenv("LTX_OUTPUT_DIR", "./outputs")
LTX_API_KEY = os.getenv("LTX_API_KEY", "")
LTX_API_BASE = os.getenv("LTX_API_BASE", "https://api.ltx.video")
MAX_JOBS_HISTORY = int(os.getenv("LTX_MAX_JOBS", "100"))
# Streaming: build transformer on CPU and stream N layers ahead to GPU.
# Required on 16 GB GPUs — the 22B transformer cannot be loaded all at once.
STREAMING_PREFETCH_COUNT = int(os.getenv("LTX_STREAMING_PREFETCH", "2"))
# Optional Bearer token required by /v1/* endpoints (LTX-Desktop compat layer).
# If empty, any (or no) Authorization header is accepted.
# Set to the same value you put in LTX-Desktop's API-key field so both match.
CONSOLE_API_KEY: str = os.getenv("LTX_CONSOLE_API_KEY", "")

# T4 VRAM-safe defaults (width & height must be multiples of 64)
# Valid examples: 512×256, 640×384, 704×448, 768×512, 1024×576
DEFAULT_WIDTH = 704
DEFAULT_HEIGHT = 448

# Camera-motion prompt suffixes (injected server-side for /v1/* endpoints).
_CAMERA_MOTION_PROMPTS: dict[str, str] = {
    "none": "",
    "static": ", static camera, locked off shot, no camera movement",
    "focus_shift": ", focus shift, rack focus, changing focal point",
    "dolly_in": ", dolly in, camera pushing forward, smooth forward movement",
    "dolly_out": ", dolly out, camera pulling back, smooth backward movement",
    "dolly_left": ", dolly left, camera tracking left, lateral movement",
    "dolly_right": ", dolly right, camera tracking right, lateral movement",
    "jib_up": ", jib up, camera rising up, upward crane movement",
    "jib_down": ", jib down, camera lowering down, downward crane movement",
}
DEFAULT_NUM_FRAMES = 65
DEFAULT_FPS = 30
DEFAULT_SEED = 42

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------
class JobStatus(str, Enum):
    QUEUED = "queued"
    ENCODING_PROMPT = "encoding_prompt"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    prompt: str
    negative_prompt: str
    width: int
    height: int
    num_frames: int
    fps: int
    seed: int
    status: JobStatus = JobStatus.QUEUED
    error: str | None = None
    output_path: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    # Optional image-conditioning frames (absolute paths to temp PNG files).
    # start_frame_path → frame_idx=0 (anchors the first frame)
    # end_frame_path   → frame_idx=num_frames-1 (anchors the last frame)
    start_frame_path: str | None = None
    end_frame_path: str | None = None


# Bounded job store (oldest evicted first)
jobs: OrderedDict[str, Job] = OrderedDict()

# Events set by the worker when a /v1/* synchronous job finishes.
_v1_done: dict[str, asyncio.Event] = {}

# In-memory upload store for /v1/upload  (upload_id -> raw bytes)
_v1_uploads: dict[str, bytes] = {}


def _register_job(job: Job) -> None:
    while len(jobs) >= MAX_JOBS_HISTORY:
        evicted_id, evicted = jobs.popitem(last=False)
        # Clean up old output files
        if evicted.output_path and Path(evicted.output_path).exists():
            try:
                Path(evicted.output_path).unlink()
            except OSError:
                pass
    jobs[job.job_id] = job


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = Field(default=DEFAULT_WIDTH, ge=256, le=1280)
    height: int = Field(default=DEFAULT_HEIGHT, ge=256, le=1280)
    num_frames: int = Field(default=DEFAULT_NUM_FRAMES, ge=9, le=257)
    fps: int = Field(default=DEFAULT_FPS, ge=1, le=60)
    seed: int | None = None
    # Optional base64-encoded PNG/JPG frames for image conditioning.
    # start_frame_b64 anchors the first frame of the video.
    # end_frame_b64   anchors the last frame of the video.
    start_frame_b64: str | None = None
    end_frame_b64: str | None = None


class GenerateResponse(BaseModel):
    job_id: str
    status: str


class StatusResponse(BaseModel):
    status: str
    url: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Model holder (lazy-loaded on first request)
# ---------------------------------------------------------------------------
class ModelManager:
    """Manages the lifecycle of the LTX-2.3 pipeline components.

    The pipeline is loaded lazily on the first generation request to keep
    startup fast when the server is just health-checked.  Because the Gemma
    text encoder is replaced by an API call, we load only:
      - the EmbeddingsProcessor (small, runs on GPU)
      - the DiffusionStage (transformer, FP8-quantized)
      - the ImageConditioner (video encoder)
      - the VideoUpsampler
      - the VideoDecoder
      - the AudioDecoder
    """

    def __init__(self) -> None:
        self._ready = False
        self._lock = asyncio.Lock()
        # Pipeline components — populated by ``_load()``
        self._diffusion_stage = None
        self._image_conditioner = None
        self._upsampler = None
        self._video_decoder = None
        self._audio_decoder = None
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None

    @property
    def ready(self) -> bool:
        return self._ready

    async def ensure_loaded(self) -> None:
        """Ensure model is loaded, thread-safe."""
        if self._ready:
            return
        async with self._lock:
            if self._ready:
                return
            await asyncio.get_event_loop().run_in_executor(None, self._load)

    def _load(self) -> None:
        """Load all pipeline components (runs in a thread-pool worker)."""
        logger.info("Loading LTX-2.3 pipeline components …")
        start = time.time()

        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines.utils.blocks import (
            AudioDecoder,
            DiffusionStage,
            ImageConditioner,
            VideoDecoder,
            VideoUpsampler,
        )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # T4 (Turing / sm_75) has FP16 tensor cores but NOT BF16 tensor cores.
        self._dtype = torch.bfloat16

        # Detect the distilled checkpoint
        checkpoint_path = _find_checkpoint(MODEL_DIR)
        spatial_upsampler_path = _find_file(MODEL_DIR, "spatial-upscaler")
        logger.info("Checkpoint: %s", checkpoint_path)
        logger.info("Spatial upsampler: %s", spatial_upsampler_path)

        # We use FP8 cast quantization for the transformer to fit in 16 GB.
        quantization = QuantizationPolicy.fp8_cast()

        # NOTE: On a 16 GB T4 the FP8 transformer alone is ~12 GB.  Loading all
        # auxiliary components (ImageConditioner, VideoUpsampler, VideoDecoder,
        # AudioDecoder) onto the GPU simultaneously leaves no room for the
        # transformer during inference.  The solution is to keep auxiliary
        # components on CPU and move each to GPU only for its specific step
        # (see _on_gpu() and generate()).
        cpu = torch.device("cpu")

        # --- Diffusion transformer — stays on GPU (loads lazily via gpu_model) ---
        self._diffusion_stage = DiffusionStage(
            checkpoint_path=checkpoint_path,
            dtype=self._dtype,
            device=self._device,
            loras=(),
            quantization=quantization,
        )

        # --- Auxiliary components — CPU-resident, moved to GPU on demand ---
        self._image_conditioner = ImageConditioner(
            checkpoint_path=checkpoint_path,
            dtype=self._dtype,
            device=cpu,
        )

        self._upsampler = VideoUpsampler(
            checkpoint_path=checkpoint_path,
            upsampler_path=spatial_upsampler_path,
            dtype=self._dtype,
            device=cpu,
        )

        self._video_decoder = VideoDecoder(
            checkpoint_path=checkpoint_path,
            dtype=self._dtype,
            device=cpu,
        )

        self._audio_decoder = AudioDecoder(
            checkpoint_path=checkpoint_path,
            dtype=self._dtype,
            device=cpu,
        )

        # --- Connector dims for AV encoding split -----------------------
        # For the 22B model: caption_proj_before_connector=True means the
        # transformer has no caption_projection, so the API's combined encoding
        # must be split into (video_connector_dim, audio_connector_dim).
        import json
        from safetensors import safe_open
        with safe_open(checkpoint_path, framework="pt") as _f:
            _cfg = json.loads(_f.metadata().get("config", "{}"))
        _t = _cfg.get("transformer", {})
        self._video_connector_dim = (
            _t.get("connector_num_attention_heads", 32) *
            _t.get("connector_attention_head_dim", 128)
        )
        self._audio_connector_dim = (
            _t.get("audio_connector_num_attention_heads",
                   _t.get("connector_num_attention_heads", 32)) *
            _t.get("audio_connector_attention_head_dim",
                   _t.get("connector_attention_head_dim", 128))
        )
        logger.info(
            "Connector dims: video=%d, audio=%d",
            self._video_connector_dim, self._audio_connector_dim,
        )

        self._ready = True
        elapsed = time.time() - start
        logger.info("Pipeline loaded in %.1f s", elapsed)

    # ------------------------------------------------------------------
    # GPU offload helper
    # ------------------------------------------------------------------

    @contextmanager
    def _on_gpu(self, *modules):
        """Temporarily move *modules* to GPU, yield, then return them to CPU.

        Handles both nn.Module subclasses (call .to(device)) and the pipeline
        builder-wrapper classes (VideoUpsampler, VideoDecoder, AudioDecoder,
        ImageConditioner) which are plain Python objects that store a _device
        attribute used when they build sub-models on demand.
        """
        cpu = torch.device("cpu")
        try:
            for m in modules:
                if isinstance(m, torch.nn.Module):
                    m.to(self._device)
                elif hasattr(m, "_device"):
                    m._device = self._device
            yield
        finally:
            for m in modules:
                if isinstance(m, torch.nn.Module):
                    m.to(cpu)
                elif hasattr(m, "_device"):
                    m._device = cpu
            gc.collect()
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def generate(self, job: Job) -> None:
        """Run full generation for *job* (blocking, call from worker thread)."""
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.args import ImageConditioningInput
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
        from ltx_pipelines.utils.denoisers import SimpleDenoiser
        from ltx_pipelines.utils.helpers import assert_resolution, combined_image_conditionings
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_pipelines.utils.types import ModalitySpec

        device = self._device
        dtype = self._dtype

        # ------ 1. Text encoding via LTX API ----------------------------
        job.status = JobStatus.ENCODING_PROMPT
        logger.info("[%s] Encoding prompt via LTX API …", job.job_id)
        video_context, audio_context = self._encode_prompt_via_api(job.prompt)

        # ------ 2. Diffusion (two-stage distilled) -----------------------
        job.status = JobStatus.GENERATING
        logger.info("[%s] Running diffusion (seed=%d, %dx%d, %d frames) …",
                     job.job_id, job.seed, job.width, job.height, job.num_frames)

        assert_resolution(height=job.height, width=job.width, is_two_stage=True)

        generator = torch.Generator(device=device).manual_seed(job.seed)
        noiser = GaussianNoiser(generator=generator)

        # --- Stage 1: half-resolution ------------------------------------
        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(device)
        s1_w, s1_h = job.width // 2, job.height // 2

        # Move image conditioner to GPU only for this step, then back to CPU
        # Build image-conditioning inputs from start/end frame paths on the job.
        # Both stages get the same temporal frame indices; the conditioner
        # handles spatial rescaling to (s1_h, s1_w) or (job.height, job.width).
        # strength=0.7 matches the LTXVAddGuide default used in ComfyUI workflows.
        # At 1.0 the model pixel-locks the frame and fights the prompt;
        # at 0.7 it respects the characters / composition while following the motion.
        FRAME_STRENGTH = 0.7
        frame_inputs: list = []
        if job.start_frame_path:
            frame_inputs.append(
                ImageConditioningInput(path=job.start_frame_path, frame_idx=0, strength=FRAME_STRENGTH)
            )
        if job.end_frame_path:
            frame_inputs.append(
                ImageConditioningInput(path=job.end_frame_path, frame_idx=job.num_frames - 1, strength=FRAME_STRENGTH)
            )

        with self._on_gpu(self._image_conditioner):
            stage_1_conditionings = self._image_conditioner(
                lambda enc: combined_image_conditionings(
                    images=frame_inputs,
                    height=s1_h,
                    width=s1_w,
                    video_encoder=enc,
                    dtype=dtype,
                    device=device,
                )
            )
        logger.info("[%s] Stage 1 conditionings computed (frames=%d)", job.job_id, len(frame_inputs))

        # Transformer loads/unloads GPU memory transiently via gpu_model()
        video_state, audio_state = self._diffusion_stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_1_sigmas,
            noiser=noiser,
            width=s1_w,
            height=s1_h,
            frames=job.num_frames,
            fps=float(job.fps),
            video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings),
            audio=ModalitySpec(context=audio_context),
            streaming_prefetch_count=STREAMING_PREFETCH_COUNT,
        )
        logger.info("[%s] Stage 1 diffusion complete", job.job_id)

        # --- Stage 2: upsample + refine ----------------------------------
        with self._on_gpu(self._upsampler):
            upscaled_video_latent = self._upsampler(video_state.latent[:1])

        # Free Stage 1 video latent (half-res) from CUDA before Stage 2 to
        # avoid holding both half-res and full-res latents in VRAM at once.
        s1_audio_latent = audio_state.latent
        del video_state, audio_state
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("[%s] Spatial upsample complete", job.job_id)

        stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(device)

        with self._on_gpu(self._image_conditioner):
            stage_2_conditionings = self._image_conditioner(
                lambda enc: combined_image_conditionings(
                    images=frame_inputs,
                    height=job.height,
                    width=job.width,
                    video_encoder=enc,
                    dtype=dtype,
                    device=device,
                )
            )
        logger.info("[%s] Stage 2 conditionings computed", job.job_id)

        video_state, audio_state = self._diffusion_stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=job.width,
            height=job.height,
            frames=job.num_frames,
            fps=float(job.fps),
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=audio_context,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=s1_audio_latent,
            ),
            streaming_prefetch_count=STREAMING_PREFETCH_COUNT,
        )
        logger.info("[%s] Stage 2 diffusion complete", job.job_id)

        # --- Decode video + audio ----------------------------------------
        tiling_config = TilingConfig.default()
        with self._on_gpu(self._video_decoder):
            decoded_video = self._video_decoder(video_state.latent, tiling_config, generator)
        logger.info("[%s] Video decoded", job.job_id)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Audio decoder runs on CPU in float32 — cuDNN OOMs if run on GPU right
        # after video decode, and bfloat16 autocast is not supported on CPU.
        saved_dtype = self._audio_decoder._dtype
        self._audio_decoder._dtype = torch.float32
        try:
            decoded_audio = self._audio_decoder(audio_state.latent.cpu().float())
        finally:
            self._audio_decoder._dtype = saved_dtype
        logger.info("[%s] Audio decoded", job.job_id)

        # --- Write MP4 ---------------------------------------------------
        output_path = str(Path(OUTPUT_DIR) / f"{job.job_id}.mp4")
        video_chunks_number = get_video_chunks_number(job.num_frames, tiling_config)
        encode_video(
            video=decoded_video,
            fps=job.fps,
            audio=decoded_audio,
            output_path=output_path,
            video_chunks_number=video_chunks_number,
        )

        job.output_path = output_path
        logger.info("[%s] Video saved → %s", job.job_id, output_path)

        # Reclaim VRAM between jobs
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # LTX API text encoding
    # ------------------------------------------------------------------
    def _encode_prompt_via_api(
        self,
        prompt: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Call the LTX free API to obtain text embeddings, then run the
        local EmbeddingsProcessor to produce the final video/audio context
        tensors that the diffusion transformer expects.

        The LTX API ``/v1/embed`` endpoint accepts a prompt and a model_id
        (extracted from the checkpoint metadata) and returns serialized
        hidden-states + attention mask.  We then feed those through the
        small ``EmbeddingsProcessor`` network locally.

        If the dedicated embed endpoint is not available, we fall back to a
        local-only path that loads Gemma on the fly (this will only work on
        GPUs with ≥48 GB VRAM).
        """
        device = self._device
        dtype = self._dtype

        # ---- Try the LTX cloud text-encoding API -----------------------
        if LTX_API_KEY:
            try:
                return self._cloud_encode(prompt, device, dtype)
            except Exception as exc:
                logger.warning(
                    "LTX API text encoding failed (%s), falling back to local Gemma.",
                    exc,
                )

        # ---- Fallback: local Gemma (needs ~24 GB extra VRAM) -----------
        return self._local_encode(prompt, device, dtype)

    def _split_av_encoding(
        self,
        combined: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Split a packed AV conditioning tensor into (video_encoding, audio_encoding).

        The LTX 2.3 cloud API returns a single tensor of shape
        [batch, seq_len, video_dim + audio_dim] where both connector outputs are
        concatenated along the feature axis.  For the 22B distilled checkpoint the
        split point is 4096 (video) / 2048 (audio) = 6144 total.

        If the combined dim does not equal 6144 we return the full tensor as
        video_encoding and None for audio_encoding (graceful degradation).
        """
        total_dim = combined.shape[-1]
        # Derive split from checkpoint config
        video_dim = self._video_connector_dim
        audio_dim = self._audio_connector_dim
        if total_dim == video_dim + audio_dim:
            return combined[..., :video_dim], combined[..., video_dim:]
        if total_dim == video_dim:
            # Already split externally or returned alone
            return combined, None
        logger.warning(
            "Unexpected combined encoding dim %d (expected %d + %d = %d or %d). "
            "Using full tensor as video encoding and None for audio.",
            total_dim, video_dim, audio_dim, video_dim + audio_dim, video_dim,
        )
        return combined, None

    def _cloud_encode(
        self,
        prompt: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode *prompt* using the LTX free cloud API.

        Uses the ``/v1/prompt-embedding`` endpoint (same as the official
        ComfyUI GemmaAPITextEncode node).  The API accepts:

          POST /v1/prompt-embedding
          {
              "prompt": "...",
              "model_id": "<encrypted_wandb_properties from checkpoint>",
              "enhance_prompt": false
          }

        It returns a binary (pickled) conditioning tuple that is directly
        usable by the diffusion pipeline.
        """
        import pickle

        # Extract model_id from the checkpoint metadata
        checkpoint_path = _find_checkpoint(MODEL_DIR)
        model_id = _get_model_id(checkpoint_path)

        if not model_id:
            raise RuntimeError(
                "Could not extract model_id (encrypted_wandb_properties) from "
                f"checkpoint metadata at {checkpoint_path}"
            )

        headers = {
            "Authorization": f"Bearer {LTX_API_KEY}",
            "Content-Type": "application/json",
        }

        url = f"{LTX_API_BASE}/v1/prompt-embedding"
        payload: dict[str, Any] = {
            "prompt": prompt,
            "model_id": model_id,
            "enhance_prompt": False,
        }

        logger.info("Calling LTX API: %s (model_id=%s…)", url, model_id[:20])

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, headers=headers, json=payload)

            if resp.status_code == 401:
                raise RuntimeError(
                    "Invalid LTX API key. Get a new key at https://console.ltx.video/"
                )
            resp.raise_for_status()

        # The API returns pickled conditioning data in ComfyUI format:
        # [[video_tensor, {"pooled_output": None, "attention_mask": mask, "audio_encoding": tensor, ...}]]
        conditioning = pickle.loads(resp.content)

        logger.info("Conditioning type: %s, len: %s", type(conditioning).__name__,
                     len(conditioning) if hasattr(conditioning, '__len__') else 'N/A')

        # Parse ComfyUI conditioning format:
        # conditioning = [[tensor(1, seq_len, dim), {"attention_mask": ..., "audio_encoding": ..., ...}]]
        try:
            cond_item = conditioning[0]  # First conditioning entry
            video_encoding = cond_item[0]  # The tensor
            cond_meta = cond_item[1] if len(cond_item) > 1 else {}

            if not isinstance(video_encoding, torch.Tensor):
                raise ValueError(f"Expected tensor, got {type(video_encoding)}")

            # Log all keys in cond_meta to understand the full API response
            meta_summary = {
                k: (list(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__)
                for k, v in cond_meta.items()
            }
            logger.info("cond_meta keys: %s", meta_summary)

            video_encoding = video_encoding.to(device=device, dtype=dtype)

            # For the LTX 2.3 (22B) model, the API packs both video and audio
            # connector outputs into a single tensor along the feature dimension:
            #   [batch, seq_len, video_dim + audio_dim] = [1, 1024, 6144]
            # where video_dim = connector_num_attention_heads * connector_attention_head_dim
            #                 = 32 * 128 = 4096
            # and   audio_dim = audio_connector_num_attention_heads * audio_connector_attention_head_dim
            #                 = 32 * 64  = 2048
            # The transformer uses these directly (caption_proj_before_connector=True → no
            # caption_projection in transformer), so we must split them here.
            video_encoding, audio_encoding = self._split_av_encoding(video_encoding)

            logger.info(
                "Text encoding received: video_encoding=%s, audio_encoding=%s, attention_mask=%s",
                list(video_encoding.shape),
                list(audio_encoding.shape) if audio_encoding is not None else "None",
                list(cond_meta.get("attention_mask", torch.tensor([])).shape)
                if isinstance(cond_meta.get("attention_mask"), torch.Tensor) else "None",
            )

            return video_encoding, audio_encoding

        except (IndexError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Failed to parse LTX API conditioning response: {exc}. "
                f"Response type: {type(conditioning).__name__}, "
                f"len: {len(conditioning) if hasattr(conditioning, '__len__') else 'N/A'}"
            ) from exc

    def _local_encode(
        self,
        prompt: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode *prompt* locally using the full Gemma text encoder.

        This requires a local Gemma installation and ~24 GB of VRAM on top
        of the diffusion model.  It is a fallback path only.
        """
        gemma_root = os.getenv("GEMMA_ROOT", "")
        if not gemma_root:
            raise RuntimeError(
                "LTX_API_KEY is not set and GEMMA_ROOT is not configured. "
                "Either set LTX_API_KEY for cloud text encoding or set GEMMA_ROOT "
                "for local Gemma inference."
            )

        from ltx_pipelines.utils.blocks import PromptEncoder

        checkpoint_path = _find_checkpoint(MODEL_DIR)
        prompt_encoder = PromptEncoder(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            dtype=dtype,
            device=device,
        )
        (ctx,) = prompt_encoder([prompt])
        return ctx.video_encoding, ctx.audio_encoding



model_manager = ModelManager()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _find_checkpoint(model_dir: str) -> str:
    """Find the distilled checkpoint .safetensors file in *model_dir*."""
    candidates = sorted(Path(model_dir).glob("*distilled*.safetensors"))
    if candidates:
        return str(candidates[0])
    # Fall back to any .safetensors
    candidates = sorted(Path(model_dir).glob("*.safetensors"))
    if candidates:
        return str(candidates[0])
    raise FileNotFoundError(f"No .safetensors checkpoint found in {model_dir}")


def _find_file(model_dir: str, pattern: str) -> str:
    """Find a file matching *pattern* in *model_dir*."""
    candidates = sorted(Path(model_dir).glob(f"*{pattern}*"))
    if candidates:
        return str(candidates[0])
    raise FileNotFoundError(f"No file matching '{pattern}' found in {model_dir}")


def _get_model_id(checkpoint_path: str) -> str | None:
    """Read the model_id from safetensors metadata."""
    try:
        from safetensors import safe_open
        with safe_open(checkpoint_path, framework="pt") as f:
            metadata = f.metadata() or {}
        return metadata.get("encrypted_wandb_properties") or metadata.get("model_id")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Background generation worker
# ---------------------------------------------------------------------------
_generation_queue: asyncio.Queue[Job] = asyncio.Queue()


async def _worker() -> None:
    """Process generation jobs sequentially (T4 can only handle one at a time)."""
    while True:
        job = await _generation_queue.get()
        try:
            await model_manager.ensure_loaded()
            # Run blocking inference in a thread-pool executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _run_job, job)
        except Exception as exc:
            logger.exception("[%s] Generation failed", job.job_id)
            job.status = JobStatus.FAILED
            job.error = str(exc)
        finally:
            job.completed_at = time.time()
            _generation_queue.task_done()
            # Wake any /v1/* caller waiting for this job
            event = _v1_done.pop(job.job_id, None)
            if event is not None:
                event.set()


def _run_job(job: Job) -> None:
    """Execute a single generation job (runs in thread-pool)."""
    try:
        with torch.inference_mode():
            model_manager.generate(job)
        job.status = JobStatus.COMPLETED
    finally:
        # Clean up temp frame files regardless of success or failure.
        for p in (job.start_frame_path, job.end_frame_path):
            if p and Path(p).exists():
                try:
                    Path(p).unlink()
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LTX-2.3 Inference Server",
    description="Self-hosted video generation server for LTX-2.3 on Tesla T4",
    version="1.0.0",
)


@app.on_event("startup")
async def _startup() -> None:
    """Launch the background worker task."""
    asyncio.create_task(_worker())
    logger.info("LTX-2.3 server started.  Model will load on first /generate request.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": "ltx-2.3",
        "model_loaded": model_manager.ready,
        "pending_jobs": _generation_queue.qsize(),
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """Submit a video generation job.

    Returns a ``job_id`` immediately.  Poll ``/status/{job_id}`` until the
    status is ``completed``, then download from ``/download/{job_id}``.
    """
    if not LTX_API_KEY and not os.getenv("GEMMA_ROOT"):
        raise HTTPException(
            status_code=500,
            detail=(
                "Server misconfigured: set LTX_API_KEY (recommended) or GEMMA_ROOT "
                "environment variable."
            ),
        )

    # Enforce resolution divisibility (two-stage pipeline requires multiples of 64)
    if req.width % 64 != 0 or req.height % 64 != 0:
        raise HTTPException(
            status_code=400,
            detail="Width and height must be multiples of 64 for the two-stage pipeline.",
        )

    # Enforce frame count: must be 8k+1 (e.g. 9, 17, 25, … 65, 97, 129)
    if (req.num_frames - 1) % 8 != 0:
        raise HTTPException(
            status_code=400,
            detail="num_frames must satisfy (num_frames - 1) % 8 == 0 (e.g. 9, 17, 25, 33, 41, 49, 57, 65, 97, 129).",
        )

    seed = req.seed if req.seed is not None else DEFAULT_SEED

    # Decode optional base64 frames → temp PNG files.
    def _decode_frame(b64: str, label: str) -> str:
        try:
            raw = base64.b64decode(b64)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid base64 in {label}")
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(raw)
        tmp.close()
        return tmp.name

    start_frame_path = _decode_frame(req.start_frame_b64, "start_frame_b64") if req.start_frame_b64 else None
    end_frame_path   = _decode_frame(req.end_frame_b64,   "end_frame_b64")   if req.end_frame_b64   else None

    job = Job(
        job_id=str(uuid.uuid4()),
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=req.width,
        height=req.height,
        num_frames=req.num_frames,
        fps=req.fps,
        seed=seed,
        start_frame_path=start_frame_path,
        end_frame_path=end_frame_path,
    )
    _register_job(job)
    await _generation_queue.put(job)

    return GenerateResponse(job_id=job.job_id, status=job.status.value)


@app.post("/generate-frames", response_model=GenerateResponse)
async def generate_frames(
    prompt: str = Form(...),
    negative_prompt: str = Form(default=""),
    width: int = Form(default=DEFAULT_WIDTH),
    height: int = Form(default=DEFAULT_HEIGHT),
    num_frames: int = Form(default=DEFAULT_NUM_FRAMES),
    fps: int = Form(default=DEFAULT_FPS),
    seed: int | None = Form(default=None),
    start_frame: UploadFile | None = File(default=None),
    end_frame: UploadFile | None = File(default=None),
) -> GenerateResponse:
    """Submit a generation job with optional start/end frame images (multipart form).

    Upload images as multipart files.  ``start_frame`` anchors the first frame
    of the video; ``end_frame`` anchors the last.  Both are optional — omit
    either for unconditioned generation at that end.

    Curl example::

        curl -X POST http://localhost:8000/generate-frames \\
          -F prompt="a fox riding a bicycle" \\
          -F width=1024 -F height=576 -F num_frames=97 \\
          -F start_frame=@start.png \\
          -F end_frame=@end.png
    """
    if width % 64 != 0 or height % 64 != 0:
        raise HTTPException(status_code=400, detail="Width and height must be multiples of 64.")
    if (num_frames - 1) % 8 != 0:
        raise HTTPException(status_code=400, detail="num_frames must satisfy (num_frames - 1) % 8 == 0.")

    def _save_upload(upload: UploadFile, label: str) -> str:
        data = asyncio.get_event_loop().run_until_complete(upload.read()) if False else None
        # We're in an async context — use await properly via helper.
        raise RuntimeError("use _save_upload_async")

    async def _save_upload_async(upload: UploadFile) -> str:
        data = await upload.read()
        tmp = tempfile.NamedTemporaryFile(suffix=Path(upload.filename or "frame.png").suffix or ".png", delete=False)
        tmp.write(data)
        tmp.close()
        return tmp.name

    start_frame_path = await _save_upload_async(start_frame) if start_frame else None
    end_frame_path   = await _save_upload_async(end_frame)   if end_frame   else None

    resolved_seed = seed if seed is not None else DEFAULT_SEED
    job = Job(
        job_id=str(uuid.uuid4()),
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        seed=resolved_seed,
        start_frame_path=start_frame_path,
        end_frame_path=end_frame_path,
    )
    _register_job(job)
    await _generation_queue.put(job)
    return GenerateResponse(job_id=job.job_id, status=job.status.value)


@app.get("/status/{job_id}", response_model=StatusResponse)
async def status(job_id: str) -> StatusResponse:
    """Poll job status."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    resp = StatusResponse(status=job.status.value)
    if job.status == JobStatus.COMPLETED:
        resp.url = f"/download/{job.job_id}"
    elif job.status == JobStatus.FAILED:
        resp.error = job.error
    return resp


@app.get("/download/{job_id}")
async def download(job_id: str) -> FileResponse:
    """Download the generated MP4 video."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=409, detail=f"Job is not completed (status={job.status.value})")
    if not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(status_code=410, detail="Output file no longer available")
    return FileResponse(
        path=job.output_path,
        media_type="video/mp4",
        filename=f"{job.job_id}.mp4",
    )


# ===========================================================================
# LTX-Desktop / LTX Console compatibility layer  (/v1/*)
# ===========================================================================
# These endpoints mirror the protocol used by api.ltx.video so that
# LTX-Desktop (running on macOS) can point to this server instead of the
# official LTX cloud.  Configure LTX-Desktop by setting:
#
#   LTX_API_BASE_URL=http://<this-server-ip>:8000
#
# in the environment before launching the LTX-Desktop backend (ltx2_server.py).
# Also set LTX_CONSOLE_API_KEY here to match the API key you enter in
# LTX-Desktop's settings so that the Bearer token is accepted.
# ===========================================================================

# ---- Pydantic models for /v1/* requests ----------------------------------

class _V1TextToVideoRequest(BaseModel):
    prompt: str
    model: str = "ltx-2-3-fast"
    resolution: str = "1024x576"
    duration: float = Field(default=5.0, ge=1.0)
    fps: float = Field(default=24.0, ge=1.0)
    generate_audio: bool = False
    camera_motion: str | None = None


class _V1ImageToVideoRequest(BaseModel):
    prompt: str
    image_uri: str
    model: str = "ltx-2-3-fast"
    resolution: str = "1024x576"
    duration: float = Field(default=5.0, ge=1.0)
    fps: float = Field(default=24.0, ge=1.0)
    generate_audio: bool = False
    camera_motion: str | None = None


class _V1AudioToVideoRequest(BaseModel):
    prompt: str
    audio_uri: str
    image_uri: str | None = None
    model: str = "ltx-2-3-pro"
    resolution: str = "1024x576"


# ---- Helpers --------------------------------------------------------------

def _v1_check_auth(request: Request) -> None:
    """Verify Bearer token if CONSOLE_API_KEY is configured."""
    if not CONSOLE_API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:].strip() != CONSOLE_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _v1_parse_resolution(resolution: str) -> tuple[int, int]:
    """Parse 'WxH' string to (width, height), e.g. '1920x1080' → (1920, 1080)."""
    try:
        w_str, h_str = resolution.lower().split("x")
        return int(w_str), int(h_str)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid resolution '{resolution}' — expected WxH, e.g. '1920x1080'")


def _v1_snap_to_t4(width: int, height: int) -> tuple[int, int]:
    """Scale + snap to the max safe resolution for the T4, preserving aspect ratio.

    Hard limit: 1024×576 for landscape, 576×1024 for portrait.
    Both dimensions are rounded to the nearest multiple of 64 (two-stage
    pipeline requirement).
    """
    # Choose portrait vs landscape limits
    if height > width:
        max_w, max_h = 576, 1024
    else:
        max_w, max_h = 1024, 576

    scale = min(max_w / width, max_h / height, 1.0)
    w = max(64, round(width * scale / 64) * 64)
    h = max(64, round(height * scale / 64) * 64)
    return w, h


def _v1_num_frames(duration_sec: float, fps: float) -> int:
    """Compute LTX-compatible num_frames: ((duration * fps) // 8) * 8 + 1, min 9."""
    n = int((duration_sec * fps) // 8) * 8 + 1
    return max(n, 9)


def _v1_resolve_upload(uri: str) -> bytes:
    """Return bytes for a previously-uploaded ltx-upload:// URI."""
    if not uri.startswith("ltx-upload://"):
        raise HTTPException(status_code=400, detail=f"Unsupported URI scheme: {uri!r}")
    upload_id = uri[len("ltx-upload://"):]
    data = _v1_uploads.get(upload_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Upload not found: {upload_id}")
    return data


async def _v1_generate(
    prompt: str,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    start_frame_path: str | None = None,
    end_frame_path: str | None = None,
) -> bytes:
    """Submit a generation job, wait for it, and return the raw MP4 bytes.

    This is the synchronous-from-the-caller's-perspective entry point used by
    all /v1/* generation endpoints.  Internally it still uses the async job
    queue so it does not block the event loop, and respects the single-worker
    constraint (only one GPU job runs at a time).
    """
    seed = int(time.time()) % 2_147_483_647
    job = Job(
        job_id=str(uuid.uuid4()),
        prompt=prompt,
        negative_prompt="",
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        seed=seed,
        start_frame_path=start_frame_path,
        end_frame_path=end_frame_path,
    )
    _register_job(job)

    done_event = asyncio.Event()
    _v1_done[job.job_id] = done_event
    await _generation_queue.put(job)

    logger.info(
        "[v1/%s] queued %dx%d %d frames %.1f fps",
        job.job_id[:8], width, height, num_frames, fps,
    )

    # Block this coroutine (not the thread) until the worker signals completion.
    await done_event.wait()

    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=500, detail=job.error or "Generation failed")
    if not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(status_code=500, detail="Output file missing after generation")

    return Path(job.output_path).read_bytes()


# ---- /v1/upload -----------------------------------------------------------

@app.post("/v1/upload")
async def v1_upload(request: Request) -> dict:
    """Initiate a file upload (mirrors api.ltx.video /v1/upload).

    Returns a pre-signed-style upload URL that points back to this server.
    The caller should PUT the file to upload_url, then use storage_uri as
    the image_uri / audio_uri in subsequent generation requests.
    """
    _v1_check_auth(request)
    upload_id = str(uuid.uuid4())
    base = str(request.base_url).rstrip("/")
    return {
        "upload_url": f"{base}/v1/upload-data/{upload_id}",
        "storage_uri": f"ltx-upload://{upload_id}",
        "required_headers": {},
    }


@app.put("/v1/upload-data/{upload_id}")
async def v1_upload_data(upload_id: str, request: Request) -> Response:
    """Receive the raw file bytes for a previously-initiated upload."""
    data = await request.body()
    _v1_uploads[upload_id] = data
    logger.info("[v1/upload] stored %d bytes for id=%s", len(data), upload_id[:8])
    return Response(status_code=200)


# ---- /v1/text-to-video ----------------------------------------------------

@app.post("/v1/text-to-video")
async def v1_text_to_video(req: _V1TextToVideoRequest, request: Request) -> Response:
    """Text-to-video generation (synchronous, returns video/mp4 bytes).

    Implements the same interface as api.ltx.video/v1/text-to-video so that
    LTX-Desktop configured with LTX_API_BASE_URL pointing here works without
    any frontend changes.
    """
    _v1_check_auth(request)

    # Parse and clamp resolution to what the T4 can handle
    raw_w, raw_h = _v1_parse_resolution(req.resolution)
    width, height = _v1_snap_to_t4(raw_w, raw_h)
    if (raw_w, raw_h) != (width, height):
        logger.info("[v1/t2v] resolution clamped %dx%d → %dx%d for T4", raw_w, raw_h, width, height)

    num_frames = _v1_num_frames(req.duration, req.fps)
    fps = int(req.fps)

    # Inject camera-motion suffix into the prompt
    cam = req.camera_motion or "none"
    prompt = req.prompt + _CAMERA_MOTION_PROMPTS.get(cam, "")

    video_bytes = await _v1_generate(
        prompt=prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
    )
    return Response(content=video_bytes, media_type="video/mp4")


# ---- /v1/image-to-video ---------------------------------------------------

@app.post("/v1/image-to-video")
async def v1_image_to_video(req: _V1ImageToVideoRequest, request: Request) -> Response:
    """Image-to-video generation.

    The image must first be uploaded via POST /v1/upload + PUT /v1/upload-data.
    Currently falls back to pure text-to-video because the Job queue does not
    yet carry image conditioning; the image_uri is accepted but ignored.
    Full i2v support requires adding image_path to the Job model and wiring
    it through ModelManager.generate().
    """
    _v1_check_auth(request)

    # Resolve upload — fail fast if URI is unknown, get raw bytes.
    image_bytes = _v1_resolve_upload(req.image_uri)

    # Save to a temp file so the image conditioner can load it.
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(image_bytes)
    tmp.close()
    start_frame_path = tmp.name

    raw_w, raw_h = _v1_parse_resolution(req.resolution)
    width, height = _v1_snap_to_t4(raw_w, raw_h)
    num_frames = _v1_num_frames(req.duration, req.fps)
    fps = int(req.fps)
    cam = req.camera_motion or "none"
    prompt = req.prompt + _CAMERA_MOTION_PROMPTS.get(cam, "")

    video_bytes = await _v1_generate(
        prompt=prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        start_frame_path=start_frame_path,
    )
    return Response(content=video_bytes, media_type="video/mp4")


# ---- /v1/audio-to-video ---------------------------------------------------

@app.post("/v1/audio-to-video")
async def v1_audio_to_video(req: _V1AudioToVideoRequest, request: Request) -> Response:
    """Audio-to-video generation — not yet implemented on this server.

    The A2V pipeline requires a separate two-stage setup with audio conditioning
    that has not been integrated into this server's Job model.  Returns 501.
    """
    _v1_check_auth(request)
    raise HTTPException(
        status_code=501,
        detail="audio-to-video is not yet supported on this server",
    )

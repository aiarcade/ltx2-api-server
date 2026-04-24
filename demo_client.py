#!/usr/bin/env python3
"""
LTX-2.3 API demo client.

Run from any machine that can reach the server.

Examples:
  python demo_client.py health
  python demo_client.py text "a rabbit runs across a meadow"
  python demo_client.py start-frame "a rabbit hops away" /path/to/start.png
  python demo_client.py end-frame   "a rabbit arrives"   /path/to/end.png
  python demo_client.py both-frames "a rabbit runs away" /path/to/start.png /path/to/end.png
  python demo_client.py status <job_id>

Environment variables:
  SERVER_URL   base URL of the server  (default: http://100.118.148.127:8000)
  POLL_SEC     seconds between status polls (default: 5)
"""

import argparse
import os
import sys
import time
import json
import mimetypes
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Please install requests:  pip install requests")

# ── Configuration ─────────────────────────────────────────────────────────────
SERVER_URL = os.environ.get("SERVER_URL", "http://100.118.148.127:8000").rstrip("/")
POLL_SEC   = int(os.environ.get("POLL_SEC", "5"))
OUTPUT_DIR = Path("./ltx_outputs")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "ltx-demo-client/1.0"
    return s


def _poll(session: requests.Session, job_id: str) -> str:
    """Poll /status/<job_id> until done or failed.  Returns final status."""
    url = f"{SERVER_URL}/status/{job_id}"
    print(f"  Polling job {job_id} …", flush=True)
    while True:
        try:
            r = session.get(url, timeout=15)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  Poll error: {e}")
            time.sleep(POLL_SEC)
            continue

        data = r.json()
        status = data.get("status", "unknown")
        progress = data.get("progress")
        step     = data.get("step")
        total    = data.get("total_steps")

        if progress is not None:
            bar = "#" * int(progress * 30)
            pct = int(progress * 100)
            print(f"  [{bar:<30}] {pct:3d}%  step {step}/{total}  ({status})", end="\r", flush=True)
        else:
            print(f"  Status: {status}", end="\r", flush=True)

        if status in ("done", "failed"):
            print()
            return status

        time.sleep(POLL_SEC)


def _download(session: requests.Session, job_id: str, output_path: Path) -> None:
    url = f"{SERVER_URL}/download/{job_id}"
    print(f"  Downloading → {output_path} …", flush=True)
    r = session.get(url, timeout=120, stream=True)
    r.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved {output_path}  ({size_mb:.1f} MB)")


def _submit(session: requests.Session, fields: dict, files: dict | None = None) -> str:
    """Submit to /generate-frames and return job_id."""
    if files:
        r = session.post(f"{SERVER_URL}/generate-frames", data=fields, files=files, timeout=30)
    else:
        r = session.post(f"{SERVER_URL}/generate", json=fields, timeout=30)
    r.raise_for_status()
    data = r.json()
    job_id = data["job_id"]
    print(f"  Submitted — job_id: {job_id}")
    return job_id


def _run_job(session: requests.Session, job_id: str, label: str) -> None:
    status = _poll(session, job_id)
    if status == "failed":
        r = session.get(f"{SERVER_URL}/status/{job_id}", timeout=10)
        err = r.json().get("error", "(no detail)")
        print(f"  Job FAILED: {err}")
        return
    out = OUTPUT_DIR / f"{label}_{job_id[:8]}.mp4"
    _download(session, job_id, out)


# ── Demo functions ─────────────────────────────────────────────────────────────

def demo_health(session: requests.Session) -> None:
    print(f"[health]  GET {SERVER_URL}/health")
    r = session.get(f"{SERVER_URL}/health", timeout=10)
    r.raise_for_status()
    data = r.json()
    print(f"  status      : {data.get('status')}")
    print(f"  model       : {data.get('model')}")
    print(f"  model_loaded: {data.get('model_loaded')}")
    print(f"  pending_jobs: {data.get('pending_jobs')}")


def demo_text(session: requests.Session, prompt: str) -> None:
    print(f"[text-to-video]  prompt: {prompt!r}")
    fields = {
        "prompt": prompt,
        "width": 768,
        "height": 512,
        "num_frames": 65,
        "fps": 24,
        "num_inference_steps": 30,
    }
    job_id = _submit(session, fields)
    _run_job(session, job_id, "text")


def demo_start_frame(session: requests.Session, prompt: str, image_path: str) -> None:
    print(f"[start-frame]  prompt: {prompt!r}  image: {image_path}")
    img = Path(image_path)
    if not img.exists():
        sys.exit(f"Image not found: {image_path}")
    mime = mimetypes.guess_type(str(img))[0] or "image/png"
    fields = {
        "prompt": prompt,
        "width": 768,
        "height": 512,
        "num_frames": 65,
        "fps": 24,
        "num_inference_steps": 30,
    }
    files = {"start_frame": (img.name, img.read_bytes(), mime)}
    job_id = _submit(session, fields, files)
    _run_job(session, job_id, "start_frame")


def demo_end_frame(session: requests.Session, prompt: str, image_path: str) -> None:
    print(f"[end-frame]  prompt: {prompt!r}  image: {image_path}")
    img = Path(image_path)
    if not img.exists():
        sys.exit(f"Image not found: {image_path}")
    mime = mimetypes.guess_type(str(img))[0] or "image/png"
    fields = {
        "prompt": prompt,
        "width": 768,
        "height": 512,
        "num_frames": 65,
        "fps": 24,
        "num_inference_steps": 30,
    }
    files = {"end_frame": (img.name, img.read_bytes(), mime)}
    job_id = _submit(session, fields, files)
    _run_job(session, job_id, "end_frame")


def demo_both_frames(session: requests.Session, prompt: str,
                     start_path: str, end_path: str) -> None:
    print(f"[both-frames]  prompt: {prompt!r}")
    print(f"  start: {start_path}")
    print(f"  end:   {end_path}")
    s_img = Path(start_path)
    e_img = Path(end_path)
    for p in (s_img, e_img):
        if not p.exists():
            sys.exit(f"Image not found: {p}")
    fields = {
        "prompt": prompt,
        "width": 768,
        "height": 512,
        "num_frames": 65,
        "fps": 24,
        "num_inference_steps": 30,
    }
    files = {
        "start_frame": (s_img.name, s_img.read_bytes(), mimetypes.guess_type(str(s_img))[0] or "image/png"),
        "end_frame":   (e_img.name, e_img.read_bytes(), mimetypes.guess_type(str(e_img))[0] or "image/png"),
    }
    job_id = _submit(session, fields, files)
    _run_job(session, job_id, "both_frames")


def demo_status(session: requests.Session, job_id: str) -> None:
    print(f"[status]  job_id: {job_id}")
    r = session.get(f"{SERVER_URL}/status/{job_id}", timeout=10)
    r.raise_for_status()
    print(json.dumps(r.json(), indent=2))


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LTX-2.3 demo client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health", help="Check server health")

    p = sub.add_parser("text", help="Text-to-video")
    p.add_argument("prompt")

    p = sub.add_parser("start-frame", help="Image-to-video with start frame")
    p.add_argument("prompt")
    p.add_argument("image")

    p = sub.add_parser("end-frame", help="Image-to-video with end frame")
    p.add_argument("prompt")
    p.add_argument("image")

    p = sub.add_parser("both-frames", help="Image-to-video with start + end frames")
    p.add_argument("prompt")
    p.add_argument("start_image")
    p.add_argument("end_image")

    p = sub.add_parser("status", help="Check job status (and download if done)")
    p.add_argument("job_id")

    args = parser.parse_args()
    session = _session()

    print(f"Server: {SERVER_URL}\n")

    if args.cmd == "health":
        demo_health(session)
    elif args.cmd == "text":
        demo_text(session, args.prompt)
    elif args.cmd == "start-frame":
        demo_start_frame(session, args.prompt, args.image)
    elif args.cmd == "end-frame":
        demo_end_frame(session, args.prompt, args.image)
    elif args.cmd == "both-frames":
        demo_both_frames(session, args.prompt, args.start_image, args.end_image)
    elif args.cmd == "status":
        demo_status(session, args.job_id)


if __name__ == "__main__":
    main()

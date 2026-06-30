#!/usr/bin/env python3
"""
NexusAI Video Agent — Cinematic AI Video Generator
Replicates "Multiverse of AI" style: FLUX image → MiniMax/Wan2.1 animation → FFmpeg post-process
REST API on port 8767
"""
import os
import uuid
import time
import json
import shutil
import requests
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, send_file
import replicate

app = Flask(__name__)
OUTPUT_DIR = Path("/data/.openclaw/workspace/videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")


# ─── CINEMATIC PROMPT ENHANCER ────────────────────────────────────────────────

CINEMATIC_SUFFIX = (
    ", cinematic, photorealistic, 8K, dramatic lighting, Hollywood VFX, "
    "hyperrealistic, film grain, IMAX quality, movie still, ultra detailed"
)

IMAGE_SUFFIX = (
    ", cinematic portrait, photorealistic, dramatic lighting, dark background, "
    "sharp focus, 8K, professional photography, movie poster style, "
    "hyperrealistic, high contrast, depth of field"
)


# ─── PIPELINE STEPS ───────────────────────────────────────────────────────────

def generate_image_flux(prompt: str, aspect: str = "9:16") -> bytes:
    """Step 1: Generate a cinematic image with FLUX-schnell via Replicate"""
    client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    enhanced = prompt + IMAGE_SUFFIX

    output = client.run(
        "black-forest-labs/flux-schnell",
        input={
            "prompt": enhanced,
            "aspect_ratio": aspect,
            "num_outputs": 1,
            "output_format": "jpg",
            "output_quality": 95,
        }
    )
    # output is a list of FileOutput objects
    url = str(output[0])
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def animate_image_minimax(image_bytes: bytes, prompt: str, image_path: str) -> str:
    """Step 2: Animate the image with MiniMax Video-01 image-to-video"""
    with open(image_path, 'wb') as f:
        f.write(image_bytes)

    client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    enhanced_prompt = prompt + CINEMATIC_SUFFIX

    with open(image_path, 'rb') as img_file:
        output = client.run(
            "minimax/video-01",
            input={
                "prompt": enhanced_prompt,
                "first_frame_image": img_file,
                "prompt_optimizer": True,
            }
        )

    video_url = str(output)
    r = requests.get(video_url, timeout=300)
    r.raise_for_status()
    return r.content


def generate_video_t2v(prompt: str) -> bytes:
    """Step 2 (alt): Direct text-to-video with Wan2.1 if no image needed"""
    client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    enhanced = prompt + CINEMATIC_SUFFIX

    output = client.run(
        "wavespeedai/wan-2.1-t2v-480p",
        input={
            "prompt": enhanced,
            "num_frames": 81,       # ~5 seconds
            "guidance_scale": 7.5,
            "num_inference_steps": 30,
            "fast_mode": "Balanced",
        }
    )
    video_url = str(output)
    r = requests.get(video_url, timeout=300)
    r.raise_for_status()
    return r.content


def add_overlays_ffmpeg(input_path: str, output_path: str, title: str = "", watermark: str = "@NexoraAgency"):
    """Step 3: Add title text + watermark with FFmpeg"""
    filters = []

    if title:
        # Semi-transparent black bar at top
        filters.append(
            "drawbox=x=0:y=0:w=iw:h=120:color=black@0.55:t=fill"
        )
        # Title text centered in bar
        safe_title = title.replace("'", "\\'")
        filters.append(
            f"drawtext=text='{safe_title}'"
            ":fontcolor=white:fontsize=44:x=(w-text_w)/2:y=38"
            ":shadowcolor=black@0.8:shadowx=2:shadowy=2"
        )

    if watermark:
        safe_wm = watermark.replace("'", "\\'")
        filters.append(
            f"drawtext=text='{safe_wm}'"
            ":fontcolor=white@0.65:fontsize=20:x=w-text_w-16:y=h-36"
            ":shadowcolor=black:shadowx=1:shadowy=1"
        )

    vf = ",".join(filters) if filters else "null"

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # FFmpeg might not have fonts — try without font file
        raise RuntimeError(f"FFmpeg error: {result.stderr[-500:]}")
    return output_path


# ─── API ENDPOINTS ────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    has_token = bool(REPLICATE_API_TOKEN)
    ffmpeg_ok = shutil.which('ffmpeg') is not None
    return jsonify({
        "status": "ok",
        "service": "nexusai-video-agent",
        "replicate_token": has_token,
        "ffmpeg": ffmpeg_ok,
    })


@app.route('/video/generate', methods=['POST'])
def generate_video():
    """
    Full pipeline: concept → FLUX image → MiniMax animation → FFmpeg overlay

    POST /video/generate
    {
        "prompt": "Superman in a red suit floating above Earth, Viltrumite armor",
        "title": "What If Superman Was Raised on Viltrum",
        "mode": "image2video"  // or "text2video"
    }
    """
    if not REPLICATE_API_TOKEN:
        return jsonify({"error": "REPLICATE_API_TOKEN not set in environment"}), 500

    data = request.json or {}
    prompt = data.get('prompt', '').strip()
    title = data.get('title', '')
    mode = data.get('mode', 'image2video')  # image2video | text2video
    watermark = data.get('watermark', '@NexoraAgency')

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    job_id = str(uuid.uuid4())[:8]
    img_path = str(OUTPUT_DIR / f"{job_id}_frame.jpg")
    raw_video = str(OUTPUT_DIR / f"{job_id}_raw.mp4")
    final_video = str(OUTPUT_DIR / f"{job_id}_final.mp4")

    try:
        if mode == 'image2video':
            # Step 1: Generate cinematic image with FLUX
            print(f"[{job_id}] Step 1: Generating image with FLUX...")
            image_bytes = generate_image_flux(prompt)

            # Step 2: Animate with MiniMax Video-01
            print(f"[{job_id}] Step 2: Animating with MiniMax Video-01...")
            video_bytes = animate_image_minimax(image_bytes, prompt, img_path)
        else:
            # Direct text-to-video with Wan2.1
            print(f"[{job_id}] Step 1: Generating video with Wan2.1...")
            video_bytes = generate_video_t2v(prompt)

        # Save raw video
        with open(raw_video, 'wb') as f:
            f.write(video_bytes)

        # Step 3: Add text overlays
        print(f"[{job_id}] Step 3: Adding overlays with FFmpeg...")
        try:
            add_overlays_ffmpeg(raw_video, final_video, title=title, watermark=watermark)
            os.remove(raw_video)
            if os.path.exists(img_path):
                os.remove(img_path)
            output_file = final_video
        except Exception as ffmpeg_err:
            print(f"[{job_id}] FFmpeg overlay failed ({ffmpeg_err}), returning raw video")
            output_file = raw_video

        fname = os.path.basename(output_file)
        print(f"[{job_id}] Done! File: {fname}")
        return jsonify({
            "job_id": job_id,
            "status": "done",
            "download": f"/video/download/{fname}",
        })

    except Exception as e:
        print(f"[{job_id}] ERROR: {e}")
        for f in [img_path, raw_video, final_video]:
            if os.path.exists(f):
                os.remove(f)
        return jsonify({"error": str(e)}), 500


@app.route('/video/download/<filename>', methods=['GET'])
def download_video(filename):
    path = OUTPUT_DIR / filename
    if not path.exists() or not path.suffix in ['.mp4', '.jpg']:
        return jsonify({"error": "not found"}), 404
    return send_file(str(path), mimetype='video/mp4', as_attachment=True)


@app.route('/video/list', methods=['GET'])
def list_videos():
    files = sorted(OUTPUT_DIR.glob('*_final.mp4'), key=lambda f: f.stat().st_mtime, reverse=True)
    return jsonify([
        {"file": f.name, "size_mb": round(f.stat().st_size / 1024 / 1024, 1)}
        for f in files[:20]
    ])


# ─── BATCH ENDPOINT (multiple clips for a "What If" episode) ─────────────────

@app.route('/video/episode', methods=['POST'])
def generate_episode():
    """
    Generate multiple clips and stitch them into one episode.

    POST /video/episode
    {
        "title": "What If Batman Fought The Predator",
        "scenes": [
            "Batman standing on a Gotham rooftop at night, rain, dramatic",
            "Predator uncloaking in the shadows behind Batman, infrared vision",
            "Batman activating sonar suit, electricity crackling around him"
        ],
        "watermark": "@NexoraAgency"
    }
    """
    if not REPLICATE_API_TOKEN:
        return jsonify({"error": "REPLICATE_API_TOKEN not set"}), 500

    data = request.json or {}
    title = data.get('title', '')
    scenes = data.get('scenes', [])
    watermark = data.get('watermark', '@NexoraAgency')

    if not scenes:
        return jsonify({"error": "scenes list required"}), 400
    if len(scenes) > 6:
        return jsonify({"error": "max 6 scenes per episode"}), 400

    episode_id = str(uuid.uuid4())[:8]
    clip_paths = []

    try:
        for i, scene_prompt in enumerate(scenes):
            print(f"[{episode_id}] Scene {i+1}/{len(scenes)}: {scene_prompt[:60]}...")
            img_path = str(OUTPUT_DIR / f"{episode_id}_s{i}_frame.jpg")
            raw_clip = str(OUTPUT_DIR / f"{episode_id}_s{i}_raw.mp4")

            image_bytes = generate_image_flux(scene_prompt)
            video_bytes = animate_image_minimax(image_bytes, scene_prompt, img_path)

            with open(raw_clip, 'wb') as f:
                f.write(video_bytes)
            clip_paths.append(raw_clip)

            if os.path.exists(img_path):
                os.remove(img_path)

        # Stitch clips together with FFmpeg
        print(f"[{episode_id}] Stitching {len(clip_paths)} clips...")
        concat_file = str(OUTPUT_DIR / f"{episode_id}_concat.txt")
        with open(concat_file, 'w') as f:
            for cp in clip_paths:
                f.write(f"file '{cp}'\n")

        stitched = str(OUTPUT_DIR / f"{episode_id}_stitched.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast", stitched
        ], check=True, capture_output=True)

        # Add title overlay to final episode
        final_episode = str(OUTPUT_DIR / f"{episode_id}_episode.mp4")
        try:
            add_overlays_ffmpeg(stitched, final_episode, title=title, watermark=watermark)
        except Exception:
            shutil.copy(stitched, final_episode)

        # Cleanup temp files
        for f in clip_paths + [concat_file, stitched]:
            if os.path.exists(f):
                os.remove(f)

        fname = os.path.basename(final_episode)
        print(f"[{episode_id}] Episode done! {fname}")
        return jsonify({
            "episode_id": episode_id,
            "status": "done",
            "scenes": len(scenes),
            "download": f"/video/download/{fname}",
        })

    except Exception as e:
        print(f"[{episode_id}] Episode ERROR: {e}")
        for f in clip_paths:
            if os.path.exists(f):
                os.remove(f)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("NexusAI Video Agent starting on port 8767")
    print(f"Replicate token: {'✓ set' if REPLICATE_API_TOKEN else '✗ MISSING'}")
    print(f"FFmpeg: {'✓' if shutil.which('ffmpeg') else '✗ not found'}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 50)
    app.run(host='0.0.0.0', port=8767, debug=False)

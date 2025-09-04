# All-in-One Transcriber

Zero-setup Whisper STT with optional speaker diarization. Works as a **GUI (Gradio)** or **CLI**. Ships FFmpeg automatically; auto-falls back to CPU if CUDA/cuDNN isn’t present.

## Features
- faster-whisper backend (no torch required for STT)
- FFmpeg auto-provision (via imageio-ffmpeg)
- CUDA→CPU auto-fallback
- Optional diarization (pyannote.audio + torch)
- Choose TXT / JSON / SRT / VTT outputs
- Save/load Hugging Face token locally (never committed)

## Quick start (from source)
```bash
pip install "faster-whisper>=1.0.0" "imageio-ffmpeg>=0.4.9" "gradio>=4.0.0"
# Optional for diarization:
# pip install "pyannote.audio>=3.1"
# pip install torch --index-url https://download.pytorch.org/whl/cpu

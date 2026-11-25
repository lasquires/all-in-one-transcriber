# ==== env + warning filters (keep first) ====
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # allow both OpenMP runtimes (Windows)
os.environ.setdefault("OMP_NUM_THREADS", "1")  # keeps threads sane
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # quieter HF hub on Windows

import warnings
warnings.filterwarnings("ignore", message=".*torchaudio._backend.*deprecated.*")
warnings.filterwarnings("ignore", message=".*AudioMetaData.*moved to `torchaudio.AudioMetaData`.*")

# ==== std imports ====
import json
import subprocess
import inspect
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from html import escape as html_escape

import gradio as gr
import ctranslate2
from faster_whisper import WhisperModel

# Optional diarization
try:
    from pyannote.audio import Pipeline as PyannotePipeline
    HAVE_PYANNOTE = True
except Exception:
    HAVE_PYANNOTE = False


# -------------------- constants --------------------
SPEAKER_COLORS = [
    "#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#d97706",
    "#0891b2", "#be185d", "#059669", "#b91c1c", "#4f46e5"
]
LANG_CHOICES = [
    "auto","en","es","fr","de","it","pt","nl","sv","no","da","fi","pl","cs","sk","sl",
    "hr","ro","bg","uk","ru","tr","ar","he","fa","hi","bn","ta","te","ml","ur","id","vi",
    "th","ko","ja","zh"
]
MODEL_CHOICES  = ["auto","tiny","base","small","medium","large-v2","large-v3","turbo","distil-large-v3","distil-large-v3.5"]
MODEL_IDS = {"turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
             "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2"}
DEVICE_CHOICES = ["auto","cuda","cpu"]
DTYPE_CHOICES  = ["auto","float16","float32","int8_float16","int8"]


# -------------------- gradio compat (for tooltips) --------------------
def _supports_info_arg(component_cls) -> bool:
    try:
        sig = inspect.signature(component_cls.__init__)
        return "info" in sig.parameters
    except Exception:
        return False

SUPPORTS_INFO = all([
    _supports_info_arg(gr.File),
    _supports_info_arg(gr.Dropdown),
    _supports_info_arg(gr.Radio),
    _supports_info_arg(gr.Checkbox),
    _supports_info_arg(gr.Textbox),
    _supports_info_arg(gr.Slider),
])

def maybe_info(kwargs: dict, text: str):
    """Attach hover info only if the installed Gradio supports it (keeps UI minimal on older versions)."""
    if SUPPORTS_INFO:
        kwargs["info"] = text


# -------------------- helpers --------------------
def torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def ffmpeg_path() -> str:
    import shutil
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def ffmpeg_installed() -> bool:
    try:
        r = subprocess.run([ffmpeg_path(), "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return r.returncode == 0
    except Exception:
        return False

def hhmmss_srt(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600); m = int((seconds % 3600) // 60); s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def hhmmss_vtt(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600); m = int((seconds % 3600) // 60); s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def overlap(a0,a1,b0,b1) -> float:
    return max(0.0, min(a1,b1) - max(a0,b0))

def prepare_audio_for_processing(src_path: str) -> str:
    """
    Convert compressed/container formats to 16 kHz mono WAV once, for both
    Whisper and pyannote. Returns the path to the (maybe converted) file.
    """
    p = Path(src_path)
    if p.suffix.lower() in {".wav", ".flac"}:
        return str(p)

    out_dir = Path(tempfile.gettempdir()) / "localstt_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{p.stem}_16k_mono.wav"

    # Reuse if up-to-date
    try:
        if out_path.exists() and out_path.stat().st_mtime > p.stat().st_mtime:
            return str(out_path)
    except Exception:
        pass

    cmd = [
        ffmpeg_path(), "-y", "-i", str(p),
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "wav", str(out_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
    return str(out_path)


# -------------------- token persistence --------------------
def _config_path() -> str:
    if os.name == "nt":
        base = os.getenv("APPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
        cfg_dir = os.path.join(base, "LocalSTT")
    else:
        cfg_dir = os.path.join(os.path.expanduser("~"), ".config", "local_stt")
    os.makedirs(cfg_dir, exist_ok=True)
    return os.path.join(cfg_dir, "config.json")

def load_saved_token() -> str:
    env = (os.getenv("HUGGINGFACE_TOKEN") or "").strip()
    if env: return env
    try:
        with open(_config_path(), "r", encoding="utf-8") as f:
            return (json.load(f).get("hf_token") or "").strip()
    except Exception:
        return ""

def save_token_to_disk(tok: str):
    with open(_config_path(), "w", encoding="utf-8") as f:
        json.dump({"hf_token": tok.strip()}, f, ensure_ascii=False, indent=2)

def clear_saved_token():
    try: os.remove(_config_path())
    except FileNotFoundError: pass


# -------------------- model caches --------------------
MODEL_CACHE: Dict[Tuple[str,str,str], WhisperModel] = {}
PYANNOTE_CACHE: Dict[str, object] = {}

def load_whisper(model_name: str, device: str, compute_type: str):
    key = (model_name, device, compute_type)
    if key in MODEL_CACHE: return MODEL_CACHE[key]
    mdl = WhisperModel(MODEL_IDS.get(model_name, model_name), device=device, compute_type=compute_type)
    MODEL_CACHE[key] = mdl
    return mdl

def load_pyannote(hf_token: Optional[str], device: str = "cpu"):
    if not HAVE_PYANNOTE:
        raise RuntimeError("pyannote.audio is not installed. Install it or disable diarization.")
    token = (hf_token or os.getenv("HUGGINGFACE_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("pyannote diarization requires a free Hugging Face token (scope: read).")
    if device in PYANNOTE_CACHE: return PYANNOTE_CACHE[device]

    # pyannote 4.x uses token= and a new result type; keep 3.x compatible.
    modern = "token" in inspect.signature(PyannotePipeline.from_pretrained).parameters
    model_id = ("pyannote/speaker-diarization-community-1" if modern
                else "pyannote/speaker-diarization-3.1")
    kwargs = {"token": token} if modern else {"use_auth_token": token}
    pipe = PyannotePipeline.from_pretrained(model_id, **kwargs)
    if pipe is None:
        raise RuntimeError(f"Could not load {model_id}; accept model access terms on Hugging Face.")

    # ↓ avoid cat() shape mismatch on odd-length decoded chunks
    try:
        if hasattr(pipe, "segmentation") and hasattr(pipe.segmentation, "inference"):
            pipe.segmentation.inference.batch_size = 1
        elif hasattr(pipe, "_segmentation") and hasattr(pipe._segmentation, "inference"):
            pipe._segmentation.inference.batch_size = 1
    except Exception:
        pass

    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is not available for diarization")
        pipe.to(torch.device("cuda"))
    PYANNOTE_CACHE[device] = pipe
    return pipe


# -------------------- core processing --------------------
def run_diarization_pyannote(audio_path: str, token: Optional[str], device: str = "cpu") -> List[Dict]:
    pipe = load_pyannote(token, device)
    # Decode using the existing ASR backend; avoid TorchCodec/FFmpeg DLL issues.
    import torch
    from faster_whisper.audio import decode_audio
    samples = decode_audio(audio_path, sampling_rate=16000)
    source = {"waveform": torch.from_numpy(samples).unsqueeze(0), "sample_rate": 16000}
    result = pipe(source)
    diar = getattr(result, "exclusive_speaker_diarization", None)
    if diar is None:
        diar = getattr(result, "speaker_diarization", result)
    out = []
    for seg, _, spk in diar.itertracks(yield_label=True):
        out.append({"start": float(seg.start), "end": float(seg.end), "speaker": str(spk)})
    # normalize labels
    remap, nxt = {}, 0
    for s in out:
        if s["speaker"] not in remap:
            remap[s["speaker"]] = f"SPK{nxt}"; nxt += 1
        s["speaker"] = remap[s["speaker"]]
    out.sort(key=lambda x: x["start"])
    return out

def merge_consecutive(segments: List[Dict], include_speakers: bool, max_gap: float = 0.35) -> List[Dict]:
    if not segments: return segments
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        last = merged[-1]
        same_speaker = (include_speakers and seg.get("speaker") == last.get("speaker")) or (not include_speakers)
        if same_speaker and (seg["start"] - last["end"] <= max_gap):
            last["end"] = seg["end"]
            last["text"] = (last["text"] + " " + seg["text"]).strip()
            if "words" in last and "words" in seg:
                last["words"].extend(seg["words"])
        else:
            merged.append(seg.copy())
    return merged

def assign_speakers(asr: List[Dict], diar: List[Dict], include_speakers: bool) -> List[Dict]:
    """If include_speakers=False, do NOT set any speaker labels at all."""
    if not include_speakers:
        return asr
    if not diar:
        for s in asr: s["speaker"] = "SPK0"
        return asr
    for s in asr:
        s0,s1 = s["start"], s["end"]
        best, ov_best = "SPK0", 0.0
        for d in diar:
            ov = overlap(s0,s1,d["start"],d["end"])
            if ov > ov_best: ov_best, best = ov, d["speaker"]
        s["speaker"] = best
    return asr

def write_srt(segs: List[Dict], include_speakers: bool) -> str:
    lines = []
    for i, s in enumerate(segs, 1):
        lines.append(str(i))
        lines.append(f"{hhmmss_srt(s['start'])} --> {hhmmss_srt(s['end'])}")
        if include_speakers and s.get("speaker"):
            lines.append(f"{s['speaker']}: {s['text']}")
        else:
            lines.append(s["text"])
        lines.append("")
    return "\n".join(lines)

def write_vtt(segs: List[Dict], include_speakers: bool) -> str:
    out = ["WEBVTT", ""]
    for s in segs:
        out.append(f"{hhmmss_vtt(s['start'])} --> {hhmmss_vtt(s['end'])}")
        if include_speakers and s.get("speaker"):
            out.append(f"{s['speaker']}: {s['text']}")
        else:
            out.append(s["text"])
        out.append("")
    return "\n".join(out)

def render_html(segs: List[Dict], include_speakers: bool) -> str:
    css = """
    <style>
      .wrap { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; line-height: 1.6; }
      .seg  { margin: 0.4rem 0; padding: 0.55rem 0.8rem; border-radius: 0.75rem; background: #f9fafb; }
      .spk  { display: inline-block; font-weight: 700; margin-right: 0.5rem; padding: 0.12rem 0.55rem; border-radius: 9999px; color: white; }
      .time { color: #6b7280; font-size: 0.82em; margin-left: 0.4rem; }
    </style>
    """
    html = [css, '<div class="wrap">']
    if include_speakers:
        # color map for speakers
        speakers = []
        for s in segs:
            if s.get("speaker") and s["speaker"] not in speakers:
                speakers.append(s["speaker"])
        color = {spk: SPEAKER_COLORS[i % len(SPEAKER_COLORS)] for i, spk in enumerate(speakers)}

    for s in segs:
        start = hhmmss_vtt(s["start"]); end = hhmmss_vtt(s["end"])
        txt = html_escape(s["text"])
        if include_speakers and s.get("speaker"):
            spk = s["speaker"]; col = color.get(spk, "#111827")
            html.append(
                f'<div class="seg"><span class="spk" style="background:{col}">{spk}</span>'
                f'<span class="time">{start} → {end}</span><div>{txt}</div></div>'
            )
        else:
            html.append(
                f'<div class="seg"><span class="time">{start} → {end}</span>'
                f'<div>{txt}</div></div>'
            )
    html.append("</div>")
    return "\n".join(html)

def resolve_device_and_compute(user_device: str, compute_type: str, log: List[str]) -> Tuple[str,str]:
    dev = "cpu"
    supported = set()
    if user_device != "cpu":
        try:
            if ctranslate2.get_cuda_device_count() > 0:
                supported = set(ctranslate2.get_supported_compute_types("cuda", 0))
                if supported:
                    dev = "cuda"
        except Exception as e:
            log.append(f"CUDA probe failed: {e}")
        if dev == "cuda":
            log.append("CTranslate2 detected a CUDA-compatible GPU.")
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                    "--format=csv,noheader"], capture_output=True,
                                   text=True, timeout=3, check=True)
                log.append("NVIDIA hardware (all adapters): " + r.stdout.strip())
            except (OSError, subprocess.SubprocessError):
                pass
        else:
            log.append("No usable CUDA device detected. CPU requires explicit approval.")
    if dev == "cuda":
        if compute_type == "auto" or compute_type not in supported:
            for candidate in ("int8_float16", "float16", "int8", "float32"):
                if candidate in supported:
                    compute_type = candidate
                    break
            else:
                compute_type = "auto"
    elif compute_type != "float32":
        compute_type = "int8"
    return dev, compute_type


CPU_NOTICE = (
    "**CPU processing requires your approval.** CPU transcription can take several times longer "
    "than GPU transcription, may heavily use your processor, increase heat and battery use, "
    "and consume substantial RAM for larger models. Auto mode uses the smaller Whisper model "
    "on CPU.\n\n"
    "**If you have an NVIDIA GPU:** check that it is enabled in Windows Device Manager, "
    "update your NVIDIA driver, run `nvidia-smi` in a terminal, and check that the CUDA/cuDNN "
    "runtime required by CTranslate2 is installed. Restart the app after changing drivers or "
    "runtime libraries. AMD/Intel GPUs are not accelerated by this CUDA build. "
    "Then select Auto or CUDA and click Recheck GPU.\n\n"
    "To proceed on CPU instead, check the acknowledgment below and click Transcribe."
)


def hardware_status(device_choice="auto", enable_diar=False):
    logs = []
    dev, precision = resolve_device_and_compute(device_choice, "auto", logs)
    if dev == "cpu":
        heading = "### ⚠️ CPU selected" if device_choice == "cpu" else "### ⚠️ GPU acceleration unavailable"
        return heading + "\n\n" + CPU_NOTICE + "\n\n" + "  \n".join(logs)
    if enable_diar and not torch_cuda_available():
        return ("### ⚠️ GPU available for transcription, but not speaker identification\n\n"
                + CPU_NOTICE + "\n\nInstall a CUDA-enabled matching PyTorch/TorchAudio pair "
                "to accelerate speaker identification, or turn speaker identification off.")
    return "### ✅ CUDA GPU detected\n\n" + "  \n".join(logs) + f"\n\nRecommended precision: {precision}. Actual inference will still be checked."


def refresh_hardware(device_choice, enable_diar):
    # Changing settings or explicitly rechecking resets approval.
    return hardware_status(device_choice, enable_diar), False


def read_asr(model, path, language, vad_filter, word_timestamps, beam_size):
    seg_iter, info = model.transcribe(
        path, language=None if language == "auto" else language,
        vad_filter=vad_filter, word_timestamps=word_timestamps,
        beam_size=int(beam_size),
    )
    segments = []
    for seg in seg_iter:
        entry = {"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()}
        if word_timestamps and seg.words:
            entry["words"] = [{"start": float(w.start), "end": float(w.end), "word": w.word}
                              for w in seg.words if w]
        segments.append(entry)
    return segments, info


def explain(err: Exception) -> str:
    msg = str(err)
    lower = msg.lower()
    if ("nonetype" in lower and "eval" in lower) or "speaker-diarization" in lower:
        return ("pyannote pipeline did not fully load (usually missing model access, a partial/corrupt HF cache, "
                "or a version mismatch). Accept model terms on Hugging Face, pin pyannote.audio & torch/torchaudio, "
                "and clear the cached pyannote models.")
    if ("cuinit" in lower) or ("cuda" in lower and "not" in lower):
        return "CUDA init failed. Check NVIDIA driver/CUDA, or set Device=cpu."
    if "ffmpeg" in lower:
        return "FFmpeg is required. Install it and ensure it's on PATH."
    if "out of memory" in lower:
        return "GPU OOM. Try a smaller model or lower precision (int8_float16/int8)."
    if "token" in lower and "pyannote" in lower:
        return "pyannote needs a Hugging Face token with read permission (same account as the model access)."
    return msg or err.__class__.__name__


# -------------------- callbacks --------------------
def transcribe(
    audio_files,
    model_name,
    language,
    enable_diar,
    hf_token,
    remember_token,
    device_choice,
    compute_type_choice,
    vad_filter,
    word_timestamps,
    beam_size,
    allow_cpu=False,
):
    logs: List[str] = []
    hardware_message = ""

    def blocked(message):
        try: gr.Warning("CPU processing blocked until you explicitly approve it.")
        except Exception: pass
        return ("", None, "", gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                "\n".join(logs), message + "\n\n" + CPU_NOTICE, False)

    if not ffmpeg_installed():
        logs.append("FFmpeg not found on PATH. Please install FFmpeg (and restart the terminal).")
        try: gr.Error("FFmpeg missing")
        except: pass
        # keep downloads hidden
        return ("", None, "", None, None, None, None, "\n".join(logs), gr.update(), False)

    # Normalize to a list of paths (gradio gives list[dict] when file_count='multiple')
    if not audio_files:
        logs.append("Please upload/select at least one audio file.")
        try: gr.Error("No audio file selected")
        except: pass
        return ("", None, "", None, None, None, None, "\n".join(logs), hardware_message or gr.update(), False)
    file_items = audio_files if isinstance(audio_files, list) else [audio_files]

    token_used = (hf_token or "").strip()
    if enable_diar and not token_used:
        token_used = load_saved_token()
        if not token_used:
            logs.append("Diarization is ON but no token provided. Add a free Hugging Face token (read scope).")
            try: gr.Warning("Hugging Face token missing for diarization")
            except: pass

    if remember_token and token_used:
        try:
            save_token_to_disk(token_used)
            os.environ["HUGGINGFACE_TOKEN"] = token_used
            logs.append("Hugging Face token saved to user config and set for this session.")
        except Exception as e:
            logs.append(f"Could not save token: {e}")

    device, compute_type = resolve_device_and_compute(device_choice, compute_type_choice, logs)
    hardware_message = hardware_status(device_choice, enable_diar)
    if device == "cpu" and not allow_cpu:
        return blocked("### ⚠️ GPU acceleration unavailable or CPU selected")
    diar_device = device
    if enable_diar and device == "cuda" and not torch_cuda_available():
        if not allow_cpu:
            return blocked("### ⚠️ Speaker identification requires CPU because PyTorch CUDA is unavailable")
        diar_device = "cpu"
        logs.append("CPU speaker identification explicitly approved.")
    if device == "cpu":
        logs.append("CPU processing explicitly approved.")
    automatic_model = model_name == "auto"
    if automatic_model:
        model_name = "turbo" if device == "cuda" else "small"
    if model_name.startswith("distil-") and language not in ("auto", "en"):
        logs.append("Distilled Whisper is English-only. Switching to multilingual turbo.")
        model_name = "turbo"
    elif model_name.startswith("distil-"):
        language = "en"

    # Load once; reuse for all files
    try:
        try:
            model = load_whisper(model_name, device, compute_type)
        except (RuntimeError, OSError, ValueError) as e:
            if device != "cuda":
                raise
            logs.append(f"GPU model load failed ({e}).")
            if not allow_cpu:
                return blocked("### ⚠️ GPU model initialization failed")
            logs.append("CPU retry explicitly approved.")
            hardware_message = "### ⚠️ GPU initialization failed — running on CPU with your approval.\n\n" + CPU_NOTICE
            device, compute_type, diar_device = "cpu", "int8", "cpu"
            if automatic_model:
                model_name = "small"
            model = load_whisper(model_name, device, compute_type)
        logs.append(f"Loaded Whisper model '{model_name}' on {device} ({compute_type}).")
    except Exception as e:
        logs.append(f"Failed to load Whisper: {explain(e)}")
        try: gr.Error("Model load failed")
        except: pass
        return ("", None, "", None, None, None, None, "\n".join(logs), hardware_message or gr.update(), False)

    total_duration = 0.0
    detected_langs = set()
    html_sections: List[str] = []
    srt_paths: List[str] = []
    vtt_paths: List[str] = []
    txt_paths: List[str] = []
    json_paths: List[str] = []

    for item in file_items:
        in_path = item.get("name") if isinstance(item, dict) else item
        original_path = in_path

        # Pre-convert for robust processing
        try:
            in_path = prepare_audio_for_processing(in_path)
        except Exception as e:
            logs.append(f"[{original_path}] FFmpeg pre-conversion failed: {e} (using original input).")
            try: gr.Warning("FFmpeg pre-conversion failed; using original input instead.")
            except: pass
            in_path = original_path

        # --- Transcribe this file ---
        try:
            try:
                asr_segments, info = read_asr(model, in_path, language, vad_filter, word_timestamps, beam_size)
            except (RuntimeError, OSError) as e:
                if device != "cuda":
                    raise
                logs.append(f"GPU inference failed ({e}).")
                MODEL_CACHE.pop((model_name, device, compute_type), None)
                if not allow_cpu:
                    hardware_message = "### ⚠️ GPU inference failed — batch stopped. Previously completed files are retained.\n\n" + CPU_NOTICE
                    try: gr.Warning("GPU inference failed. Approve CPU processing or recheck your GPU.")
                    except Exception: pass
                    break
                hardware_message = "### ⚠️ GPU inference failed — running on CPU with your approval.\n\n" + CPU_NOTICE
                logs.append("CPU retry explicitly approved.")
                device, compute_type, diar_device = "cpu", "int8", "cpu"
                if automatic_model:
                    model_name = "small"
                model = load_whisper(model_name, device, compute_type)
                asr_segments, info = read_asr(model, in_path, language, vad_filter, word_timestamps, beam_size)
            total_duration += float(info.duration)
            if info.language: detected_langs.add(info.language)
            logs.append(f"[{original_path}] Transcription complete. Duration: {round(info.duration,2)} s. Detected language: {info.language}.")
        except Exception as e:
            logs.append(f"[{original_path}] Transcription failed: {explain(e)}")
            try: gr.Error("Transcription failed")
            except: pass
            # Skip to next file
            continue

        # --- Diarization (optional) ---
        diar_segments: List[Dict] = []
        if enable_diar:
            try:
                try:
                    diar_segments = run_diarization_pyannote(in_path, token_used, diar_device)
                except (RuntimeError, OSError):
                    if diar_device != "cuda":
                        raise
                    PYANNOTE_CACHE.pop("cuda", None)
                    if not allow_cpu:
                        hardware_message = "### ⚠️ GPU speaker identification failed — CPU retry requires approval. Transcript saved without speaker labels.\n\n" + CPU_NOTICE
                        raise
                    hardware_message = "### ⚠️ Speaker identification is using CPU with your approval.\n\n" + CPU_NOTICE
                    logs.append("CPU diarization retry explicitly approved.")
                    diar_device = "cpu"
                    diar_segments = run_diarization_pyannote(in_path, token_used, diar_device)
                logs.append(f"[{original_path}] Diarization produced {len(diar_segments)} segments.")
            except Exception as e:
                logs.append(f"[{original_path}] Diarization failed: {explain(e)}")
                try: gr.Warning("Diarization failed; transcript will not be speaker-labeled.")
                except: pass

        # --- Merge & write outputs for this file ---
        has_speakers = bool(enable_diar and diar_segments)
        asr_segments = assign_speakers(asr_segments, diar_segments, include_speakers=has_speakers)
        merged = merge_consecutive(asr_segments, include_speakers=has_speakers)

        base = os.path.splitext(os.path.basename(original_path))[0]
        out_dir = os.path.join(os.path.dirname(original_path), f"{base}_outputs")
        try:
            os.makedirs(out_dir, exist_ok=True)
            srt_path  = os.path.join(out_dir, f"{base}.srt")
            vtt_path  = os.path.join(out_dir, f"{base}.vtt")
            txt_path  = os.path.join(out_dir, f"{base}.txt")
            json_path = os.path.join(out_dir, f"{base}.json")

            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(write_srt(merged, include_speakers=has_speakers))
            with open(vtt_path, "w", encoding="utf-8") as f:
                f.write(write_vtt(merged, include_speakers=has_speakers))
            with open(txt_path, "w", encoding="utf-8") as f:
                for s in merged:
                    if has_speakers and s.get("speaker"):
                        f.write(f"{s['speaker']}: {s['text']}\n")
                    else:
                        f.write(f"{s['text']}\n")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"language": info.language,
                           "duration": info.duration, "segments": merged}, f, ensure_ascii=False, indent=2)

            srt_paths.append(srt_path); vtt_paths.append(vtt_path)
            txt_paths.append(txt_path); json_paths.append(json_path)

            # Per-file HTML section
            section_title = f"<h3>{html_escape(base)}</h3>"
            html_sections.append(section_title + render_html(merged, include_speakers=has_speakers))
            logs.append(f"[{original_path}] Saved outputs in: {out_dir}")
        except Exception as e:
            logs.append(f"[{original_path}] Failed to save outputs: {explain(e)}")

    # If nothing succeeded:
    if not (srt_paths or vtt_paths or txt_paths or json_paths or html_sections):
        return ("", None, "", None, None, None, None, "\n".join(logs), hardware_message or gr.update(), False)

    # Compose and reveal
    lang_display = (list(detected_langs)[0] if len(detected_langs) == 1 else "multiple")
    html_preview = "\n<hr/>\n".join(html_sections)

    return (
        lang_display,
        round(total_duration, 2),
        html_preview,
        gr.update(value=srt_paths,  visible=True),
        gr.update(value=vtt_paths,  visible=True),
        gr.update(value=txt_paths,  visible=True),
        gr.update(value=json_paths, visible=True),
        "\n".join(logs),
        hardware_message,
        False,  # CPU permission is per run, never persisted.
    )


def persist_token(hf_token: str, remember: bool):
    tok = (hf_token or "").strip()
    if not tok:
        try: gr.Warning("No token entered")
        except: pass
        return load_saved_token(), "No token entered."
    if not remember:
        os.environ["HUGGINGFACE_TOKEN"] = tok
        return tok, "Token set for this session only (not saved)."
    try:
        save_token_to_disk(tok); os.environ["HUGGINGFACE_TOKEN"] = tok
        return tok, f"Token saved to: {_config_path()}"
    except Exception as e:
        try: gr.Error("Could not save token")
        except: pass
        return tok, f"Could not save token: {e}"

def clear_token_click():
    clear_saved_token()
    try: del os.environ["HUGGINGFACE_TOKEN"]
    except KeyError: pass
    try: gr.Info("Cleared saved token")
    except: pass
    return "", "Saved token cleared."

def toggle_token_ui(enabled: bool):
    vis = gr.update(visible=bool(enabled))
    return vis, vis, vis, vis

def make_audio_input():
    # Build kwargs once (adds tooltip on newer Gradio that supports `info`)
    kwargs = {
        "file_types": [".wav",".mp3",".m4a",".flac",".ogg",".wma",".mp4",".mkv"],
        "label": "Audio files",
    }
    try:
        import inspect
        if "info" in inspect.signature(gr.File.__init__).parameters:
            kwargs["info"] = "Audio/video accepted; audio is extracted from video."
    except Exception:
        pass

    # Newer Gradio: gr.File supports file_count="multiple"
    try:
        import inspect
        if "file_count" in inspect.signature(gr.File.__init__).parameters:
            return gr.File(file_count="multiple", **kwargs)
    except Exception:
        pass

    # Older Gradio: use gr.Files (multi-uploader component)
    if hasattr(gr, "Files"):
        return gr.Files(**kwargs)

    # Last resort: single-file input with a hint
    return gr.File(label="Audio file (single — upgrade Gradio for multiple)")

# -------------------- UI --------------------
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("## 🎙️ Local Speech-to-Text + Diarization")
    if not HAVE_PYANNOTE:
        gr.Markdown("Speaker labels require the optional pyannote environment; transcription still works.")

    with gr.Row():
        with gr.Column(scale=1):
            audio = make_audio_input()


            mk = dict(choices=MODEL_CHOICES, value="auto", label="ASR model")
            maybe_info(mk, "Auto: turbo on CUDA, small on CPU. Distil models are English-only; large-v3 is the full multilingual model.")
            model_name = gr.Dropdown(**mk)

            # keep Language simple, default auto
            lk = dict(choices=LANG_CHOICES, value="auto", label="Language")
            maybe_info(lk, "Keep ‘auto’ unless you want to force a specific language.")
            language = gr.Dropdown(**lk)

            # diarization toggle + token (hidden until on)
            enable_diar = gr.Checkbox(value=False, label="Enable diarization (speaker labels)", interactive=HAVE_PYANNOTE)
            with gr.Row():
                tk = dict(label="Hugging Face token (for diarization)", type="password", value=load_saved_token(), placeholder="hf_...")
                maybe_info(tk, "Needed only if diarization is enabled. Scope: read.")
                hf_token = gr.Textbox(**tk)
                remember_token = gr.Checkbox(value=True, label="Remember token")
            with gr.Row():
                save_tok_btn  = gr.Button("Save/Use Token", variant="secondary")
                clear_tok_btn = gr.Button("Clear Saved Token", variant="secondary")
            # hide token controls until enabled
            hf_token.visible = False; remember_token.visible = False; save_tok_btn.visible = False; clear_tok_btn.visible = False
            enable_diar.change(toggle_token_ui, inputs=[enable_diar], outputs=[hf_token, remember_token, save_tok_btn, clear_tok_btn])

            with gr.Accordion("Advanced", open=False):
                dk = dict(choices=DEVICE_CHOICES, value="auto", label="Device")
                maybe_info(dk, "auto: CUDA if available, else CPU.")
                device = gr.Radio(**dk)

                pk = dict(choices=DTYPE_CHOICES, value="auto", label="Precision")
                maybe_info(pk, "Auto chooses supported GPU precision or CPU int8.")
                compute_type = gr.Radio(**pk)

                vad_filter = gr.Checkbox(value=True,  label="Voice activity detection")
                word_ts    = gr.Checkbox(value=False, label="Word timestamps")
                bk = dict(minimum=1, maximum=8, value=5, step=1, label="Beam size")
                maybe_info(bk, "Higher may improve accuracy slightly (slower).")
                beam = gr.Slider(**bk)

            hardware_md = gr.Markdown("Checking GPU availability…")
            allow_cpu = gr.Checkbox(value=False, label="I understand CPU processing may be much slower and use more RAM/power. Allow CPU for this run only.")
            recheck_gpu = gr.Button("Recheck GPU", variant="secondary")
            run_btn = gr.Button("Transcribe", variant="primary")

        with gr.Column(scale=1):
            lang_out = gr.Textbox(label="Detected language", interactive=False)
            dur_out  = gr.Number(label="Audio duration (s)", interactive=False, precision=2)
            html_out = gr.HTML(label="Transcript")
            with gr.Accordion("Downloads", open=False):
                srt_out  = gr.File(label="SRT",  file_count="multiple", visible=False)
                vtt_out  = gr.File(label="VTT",  file_count="multiple", visible=False)
                txt_out  = gr.File(label="TXT",  file_count="multiple", visible=False)
                json_out = gr.File(label="JSON", file_count="multiple", visible=False)

            with gr.Accordion("Status & logs", open=False):
                status_md = gr.Markdown("")

    demo.load(refresh_hardware, inputs=[device, enable_diar], outputs=[hardware_md, allow_cpu])
    recheck_gpu.click(refresh_hardware, inputs=[device, enable_diar], outputs=[hardware_md, allow_cpu])
    device.change(refresh_hardware, inputs=[device, enable_diar], outputs=[hardware_md, allow_cpu])
    enable_diar.change(refresh_hardware, inputs=[device, enable_diar], outputs=[hardware_md, allow_cpu])
    run_btn.click(
        transcribe,
        inputs=[audio, model_name, language, enable_diar, hf_token, remember_token, device, compute_type, vad_filter, word_ts, beam, allow_cpu],
        outputs=[lang_out, dur_out, html_out, srt_out, vtt_out, txt_out, json_out, status_md, hardware_md, allow_cpu]
    )
    save_tok_btn.click(persist_token, inputs=[hf_token, remember_token], outputs=[hf_token, status_md])
    clear_tok_btn.click(clear_token_click, inputs=[], outputs=[hf_token, status_md])

if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        print("UI initialized; FFmpeg:", ffmpeg_installed())
        if not ffmpeg_installed():
            raise SystemExit(1)
    else:
        demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)

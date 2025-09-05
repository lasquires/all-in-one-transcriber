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
MODEL_CHOICES  = ["tiny","base","small","medium","large-v2","large-v3","distil-large-v3"]
DEVICE_CHOICES = ["auto","cuda","cpu"]
DTYPE_CHOICES  = ["float16","float32","int8_float16","int8"]


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

def ffmpeg_installed() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
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
        "ffmpeg", "-y", "-i", str(p),
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
    mdl = WhisperModel(model_name, device=device, compute_type=compute_type)
    MODEL_CACHE[key] = mdl
    return mdl

def load_pyannote(hf_token: Optional[str]):
    if not HAVE_PYANNOTE:
        raise RuntimeError("pyannote.audio is not installed. Install it or disable diarization.")
    token = (hf_token or os.getenv("HUGGINGFACE_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("pyannote diarization requires a free Hugging Face token (scope: read).")
    if "pipeline" in PYANNOTE_CACHE: return PYANNOTE_CACHE["pipeline"]

    pipe = PyannotePipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)

    # ↓ avoid cat() shape mismatch on odd-length decoded chunks
    try:
        if hasattr(pipe, "segmentation") and hasattr(pipe.segmentation, "inference"):
            pipe.segmentation.inference.batch_size = 1
        elif hasattr(pipe, "_segmentation") and hasattr(pipe._segmentation, "inference"):
            pipe._segmentation.inference.batch_size = 1
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available(): pipe.to(torch.device("cuda"))
    except Exception:
        pass
    PYANNOTE_CACHE["pipeline"] = pipe
    return pipe


# -------------------- core processing --------------------
def run_diarization_pyannote(audio_path: str, token: Optional[str]) -> List[Dict]:
    pipe = load_pyannote(token)
    diar = pipe(audio_path)
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
    dev = user_device
    if user_device == "auto":
        if torch_cuda_available():
            dev = "cuda"; log.append("Device auto-detected: CUDA GPU.")
        else:
            dev = "cpu";  log.append("Device auto-detected: CPU (no CUDA visible).")
    if dev == "cuda" and not torch_cuda_available():
        log.append("Requested CUDA, but CUDA not available. Falling back to CPU."); dev = "cpu"
    if dev == "cpu" and compute_type in ("float16","int8_float16"):
        log.append("Adjusted precision to 'int8' for CPU (float16 not supported on CPU)."); compute_type = "int8"
    return dev, compute_type

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
):
    logs: List[str] = []

    if not ffmpeg_installed():
        logs.append("FFmpeg not found on PATH. Please install FFmpeg (and restart the terminal).")
        try: gr.Error("FFmpeg missing")
        except: pass
        # keep downloads hidden
        return ("", None, "", None, None, None, None, "\n".join(logs))

    # Normalize to a list of paths (gradio gives list[dict] when file_count='multiple')
    if not audio_files:
        logs.append("Please upload/select at least one audio file.")
        try: gr.Error("No audio file selected")
        except: pass
        return ("", None, "", None, None, None, None, "\n".join(logs))
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

    # Load once; reuse for all files
    try:
        model = load_whisper(model_name, device, compute_type)
        logs.append(f"Loaded Whisper model '{model_name}' on {device} ({compute_type}).")
    except Exception as e:
        logs.append(f"Failed to load Whisper: {explain(e)}")
        try: gr.Error("Model load failed")
        except: pass
        return ("", None, "", None, None, None, None, "\n".join(logs))

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
            lang_arg = None if language == "auto" else language
            seg_iter, info = model.transcribe(
                in_path,
                language=lang_arg,
                vad_filter=vad_filter,
                word_timestamps=word_timestamps,
                beam_size=int(beam_size),
            )
            asr_segments = []
            for seg in seg_iter:
                e = {"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()}
                if word_timestamps and seg.words:
                    e["words"] = [{"start": float(w.start), "end": float(w.end), "word": w.word} for w in seg.words if w]
                asr_segments.append(e)
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
                diar_segments = run_diarization_pyannote(in_path, token_used)
                logs.append(f"[{original_path}] Diarization produced {len(diar_segments)} segments.")
            except Exception as e:
                logs.append(f"[{original_path}] Diarization failed: {explain(e)}")
                try: gr.Warning("Diarization failed; transcript will not be speaker-labeled.")
                except: pass

        # --- Merge & write outputs for this file ---
        asr_segments = assign_speakers(asr_segments, diar_segments, include_speakers=enable_diar)
        merged = merge_consecutive(asr_segments, include_speakers=enable_diar)

        base = os.path.splitext(os.path.basename(original_path))[0]
        out_dir = os.path.join(os.path.dirname(original_path), f"{base}_outputs")
        try:
            os.makedirs(out_dir, exist_ok=True)
            srt_path  = os.path.join(out_dir, f"{base}.srt")
            vtt_path  = os.path.join(out_dir, f"{base}.vtt")
            txt_path  = os.path.join(out_dir, f"{base}.txt")
            json_path = os.path.join(out_dir, f"{base}.json")

            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(write_srt(merged, include_speakers=enable_diar))
            with open(vtt_path, "w", encoding="utf-8") as f:
                f.write(write_vtt(merged, include_speakers=enable_diar))
            with open(txt_path, "w", encoding="utf-8") as f:
                for s in merged:
                    if enable_diar and s.get("speaker"):
                        f.write(f"{s['speaker']}: {s['text']}\n")
                    else:
                        f.write(f"{s['text']}\n")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"language": list(detected_langs)[-1] if detected_langs else None,
                           "duration": info.duration, "segments": merged}, f, ensure_ascii=False, indent=2)

            srt_paths.append(srt_path); vtt_paths.append(vtt_path)
            txt_paths.append(txt_path); json_paths.append(json_path)

            # Per-file HTML section
            section_title = f"<h3>{html_escape(base)}</h3>"
            html_sections.append(section_title + render_html(merged, include_speakers=enable_diar))
            logs.append(f"[{original_path}] Saved outputs in: {out_dir}")
        except Exception as e:
            logs.append(f"[{original_path}] Failed to save outputs: {explain(e)}")

    # If nothing succeeded:
    if not (srt_paths or vtt_paths or txt_paths or json_paths or html_sections):
        return ("", None, "", None, None, None, None, "\n".join(logs))

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


# -------------------- UI --------------------
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("## 🎙️ Local CUDA Speech-to-Text + Diarization")

    with gr.Row():
        with gr.Column(scale=1):
            fk = dict(file_count="multiple", file_types=[".wav",".mp3",".m4a",".flac",".ogg",".wma",".mp4",".mkv"], label="Audio file")
            maybe_info(fk, "Audio/video accepted; audio is extracted from video.")
            audio = gr.File(**fk)

            mk = dict(choices=MODEL_CHOICES, value="small", label="ASR model")
            maybe_info(mk, "Accuracy vs speed trade-off. ‘small’ is fast; ‘large-v3’ best accuracy (needs VRAM).")
            model_name = gr.Dropdown(**mk)

            # keep Language simple, default auto
            lk = dict(choices=LANG_CHOICES, value="auto", label="Language")
            maybe_info(lk, "Keep ‘auto’ unless you want to force a specific language.")
            language = gr.Dropdown(**lk)

            # diarization toggle + token (hidden until on)
            enable_diar = gr.Checkbox(value=False, label="Enable diarization (speaker labels)")
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

                pk = dict(choices=DTYPE_CHOICES, value="float16", label="Precision")
                maybe_info(pk, "GPU: float16 recommended. CPU: int8/float32.")
                compute_type = gr.Radio(**pk)

                vad_filter = gr.Checkbox(value=True,  label="Voice activity detection")
                word_ts    = gr.Checkbox(value=False, label="Word timestamps")
                bk = dict(minimum=1, maximum=8, value=5, step=1, label="Beam size")
                maybe_info(bk, "Higher may improve accuracy slightly (slower).")
                beam = gr.Slider(**bk)

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

    run_btn.click(
        transcribe,
        inputs=[audio, model_name, language, enable_diar, hf_token, remember_token, device, compute_type, vad_filter, word_ts, beam],
        outputs=[lang_out, dur_out, html_out, srt_out, vtt_out, txt_out, json_out, status_md]
    )
    save_tok_btn.click(persist_token, inputs=[hf_token, remember_token], outputs=[hf_token, status_md])
    clear_tok_btn.click(clear_token_click, inputs=[], outputs=[hf_token, status_md])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)

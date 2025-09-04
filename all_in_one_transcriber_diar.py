#!/usr/bin/env python3
# all_in_one_transcriber_diar.py
# -------------------------------------------------------------------
# Zero-setup Whisper STT (faster-whisper) with:
#  • FFmpeg auto-provision (imageio-ffmpeg)
#  • CUDA -> CPU auto-fallback (handles cuDNN/CUDA missing)
#  • Optional speaker diarization (pyannote.audio + HF token)
#  • GUI via Gradio or plain CLI
#  • Per-file downloads (TXT / JSON / SRT / VTT), no zip
#  • Persist/restore HF token locally
# -------------------------------------------------------------------

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

# ---------- Robust defaults / warnings ----------
# Avoid Windows OpenMP duplicate runtime crash (libiomp5md.dll)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Quiet HF symlink warning on Windows (nice-to-have)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

APP_NAME = "all_in_one_transcriber"
APP_DIRNAME = "all_in_one_transcriber"
MODEL_CACHE_DIR = Path.home() / ".cache" / APP_DIRNAME / "models"

def _config_dir() -> Path:
    if platform.system().lower().startswith("win"):
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / APP_DIRNAME
    elif platform.system().lower().startswith("darwin"):
        return Path.home() / "Library" / "Application Support" / APP_DIRNAME
    else:
        # Linux/Unix
        return Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / APP_DIRNAME

def _config_path() -> Path:
    return _config_dir() / "config.json"

def load_config() -> dict:
    try:
        p = _config_path()
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def save_config(cfg: dict) -> None:
    try:
        cdir = _config_dir()
        cdir.mkdir(parents=True, exist_ok=True)
        _config_path().write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"Warning: failed to save config: {e}", file=sys.stderr)

def get_saved_token() -> Optional[str]:
    cfg = load_config()
    tok = cfg.get("hf_token")
    if tok:
        return tok
    # env fallbacks
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

def set_saved_token(token: Optional[str]) -> None:
    cfg = load_config()
    if token:
        cfg["hf_token"] = token.strip()
        os.environ["HF_TOKEN"] = cfg["hf_token"]  # also set for this process
    else:
        cfg.pop("hf_token", None)
        os.environ.pop("HF_TOKEN", None)
    save_config(cfg)

# ---------- Auto-provision FFmpeg ----------
FFMPEG_EXE = "ffmpeg"
try:
    import imageio_ffmpeg  # pip install imageio-ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["PATH"] = f"{Path(FFMPEG_EXE).parent}{os.pathsep}{os.environ.get('PATH','')}"
except Exception:
    pass

def ffmpeg_available() -> bool:
    try:
        return (Path(FFMPEG_EXE).exists() or shutil.which("ffmpeg") is not None)
    except Exception:
        return False

# ---------- Core STT (no torch required) ----------
try:
    from faster_whisper import WhisperModel
except Exception as e:
    print(
        "Missing dependency: faster-whisper.\n"
        "Install:\n  pip install 'faster-whisper>=1.0.0' 'imageio-ffmpeg>=0.4.9' 'gradio>=4.0.0'\n",
        file=sys.stderr
    )
    raise

# ---------- Optional diarization (torch + pyannote) ----------
_HAS_PYANNOTE = False
try:
    # Requires: pip install "pyannote.audio>=3.1" torch
    from pyannote.audio import Pipeline
    from pyannote.core import Segment as PyaSegment
    _HAS_PYANNOTE = True
except Exception:
    _HAS_PYANNOTE = False

# ---------- Optional UI ----------
try:
    import gradio as gr
    _GRADIO_OK = True
except Exception:
    _GRADIO_OK = False

SUPPORTED_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".wma", ".mp4", ".mkv", ".mov", ".webm"}
DEFAULT_MODEL = "small"  # tiny, base, small, medium, large-v3, distil-*

# ---------- Helpers ----------
def find_media_inputs(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        out: List[Path] = []
        for ext in SUPPORTED_EXTS:
            out.extend(path.rglob(f"*{ext}"))
        return sorted({p.resolve() for p in out})
    return []

def load_model(model_name: str, device: str, compute_type: Optional[str]) -> WhisperModel:
    """
    device: 'auto' | 'cpu' | 'cuda'
    compute_type: None -> auto; else 'float32','float16','int8','int8_float16'
    Automatically falls back to CPU if CUDA/cuDNN is unavailable.
    """
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _make(dev: str) -> WhisperModel:
        return WhisperModel(
            model_name,
            device=dev,
            compute_type=compute_type or ("float16" if dev == "cuda" else "int8"),
            download_root=str(MODEL_CACHE_DIR),
        )

    # Respect explicit CPU
    if device == "cpu":
        os.environ["CTRANSLATE2_FORCE_CPU"] = "1"
        return _make("cpu")

    # Try CUDA (or auto) first
    os.environ.pop("CTRANSLATE2_FORCE_CPU", None)
    try:
        dev = "cuda" if device == "cuda" else "auto"
        return _make(dev)
    except Exception as e:
        msg = str(e).lower()
        # Graceful fallback if CUDA/cuDNN not set up
        if any(k in msg for k in ("cudnn", "cublas", "cuda", "cudart", "cudnn_ops64")):
            print("⚠️  CUDA/cuDNN not available; falling back to CPU.")
            os.environ["CTRANSLATE2_FORCE_CPU"] = "1"
            return _make("cpu")
        raise

def _fmt_srt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")

def _canon_speaker(label, label_map):
    if label is None:
        return None
    if label not in label_map:
        label_map[label] = f"SPK_{len(label_map)+1:02d}"
    return label_map[label]

def write_text_srt_vtt(
    infile: Path,
    outdir: Path,
    segments,
    speakers: Optional[List[Optional[str]]],
    emit_txt: bool,
    emit_srt: bool,
    emit_vtt: bool,
) -> List[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    base = infile.stem
    written: List[Path] = []

    if emit_txt:
        txt_out = outdir / f"{base}.txt"
    if emit_srt:
        srt_out = outdir / f"{base}.srt"
    if emit_vtt:
        vtt_out = outdir / f"{base}.vtt"

    if emit_txt:
        all_lines: List[str] = []
    if emit_srt:
        srt_lines: List[str] = []
        srt_idx = 1
    if emit_vtt:
        vtt_lines: List[str] = ["WEBVTT\n"]

    for i, seg in enumerate(segments):
        text = seg.text.strip()
        spk = speakers[i] if (speakers and i < len(speakers)) else None
        prefix = f"[{spk}] " if spk else ""
        if emit_txt:
            all_lines.append(prefix + text)
        if emit_srt or emit_vtt:
            start, end = _fmt_srt_ts(seg.start), _fmt_srt_ts(seg.end)
            if emit_srt:
                srt_lines += [str(srt_idx), f"{start} --> {end}", prefix + text, ""]
            if emit_vtt:
                vtt_lines += [f"{start.replace(',', '.')} --> {end.replace(',', '.')}", prefix + text, ""]
            if emit_srt:
                srt_idx += 1

    if emit_txt:
        txt_out.write_text("\n".join(all_lines), encoding="utf-8")
        written.append(txt_out)
    if emit_srt:
        srt_out.write_text("\n".join(srt_lines), encoding="utf-8")
        written.append(srt_out)
    if emit_vtt:
        vtt_out.write_text("\n".join(vtt_lines), encoding="utf-8")
        written.append(vtt_out)

    return written

def write_json(
    infile: Path,
    outdir: Path,
    segments,
    speakers: Optional[List[Optional[str]]],
    model_name: str,
    language: Optional[str],
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    base = infile.stem
    jpath = outdir / f"{base}.json"
    data = {
        "file": str(infile),
        "model": model_name,
        "language_hint": language,
        "segments": [
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
                "speaker": (speakers[i] if (speakers and i < len(speakers)) else None),
            }
            for i, seg in enumerate(segments)
        ],
    }
    jpath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return jpath

# ---------- Core ops ----------
def transcribe_one(
    model: WhisperModel,
    model_name: str,
    infile: Path,
    outdir: Path,
    language: Optional[str],
    vad: bool,
    word_timestamps: bool,
    beam_size: int,
    best_of: int,
    do_diar: bool,
    hf_token: Optional[str],
    speaker_count: Optional[int],
    emit_txt: bool,
    emit_json: bool,
    emit_srt: bool,
    emit_vtt: bool,
) -> List[Path]:
    print(f"→ Transcribing: {infile.name}")
    seg_iter, info = model.transcribe(
        str(infile),
        vad_filter=vad,
        language=language,
        beam_size=beam_size,
        best_of=best_of,
        word_timestamps=word_timestamps
    )
    segments = list(seg_iter)

    # Optional diarization
    speakers: Optional[List[Optional[str]]] = None
    if do_diar:
        if not _HAS_PYANNOTE:
            raise RuntimeError("Diarization requested but pyannote.audio (and torch) not installed.")
        token = hf_token or get_saved_token()
        if not token:
            raise RuntimeError("Diarization requested but no HF token provided. Use --hf-token, save it in UI, or set HF_TOKEN.")
        print("   • Running speaker diarization (pyannote)…")
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        diar = pipe(str(infile), num_speakers=(speaker_count if speaker_count and speaker_count > 0 else None))

        label_map = {}
        speakers = []
        from pyannote.core import Segment as PyaSegment
        for seg in segments:
            try:
                label = diar.argmax(PyaSegment(seg.start, seg.end))
            except Exception:
                label = None
            speakers.append(_canon_speaker(label, label_map))

    written: List[Path] = []
    # text/srt/vtt
    written.extend(
        write_text_srt_vtt(
            infile, outdir, segments, speakers,
            emit_txt=emit_txt, emit_srt=emit_srt, emit_vtt=emit_vtt
        )
    )
    # json
    if emit_json:
        jpath = write_json(infile, outdir, segments, speakers, model_name, language)
        written.append(jpath)

    print("   ✓ Wrote:", ", ".join(p.name for p in written) if written else "(no files)")
    return written

# =============================== CLI =========================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="all_in_one_transcriber_diar",
        formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent(f"""\
            All-in-one transcription (Whisper + auto FFmpeg) with OPTIONAL speaker diarization.

            Examples:
              python %(prog)s --ui
              python %(prog)s meeting.mp3 --model {DEFAULT_MODEL}
              python %(prog)s ./folder --diarize --hf-token <TOKEN> --speaker-count 2 --emit txt json
        """),
    )
    p.add_argument("input", nargs="?", help="Audio/video file or folder. Omit and use --ui for the GUI.")
    p.add_argument("-o", "--outdir", default="transcripts", help="Output folder (default: transcripts)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Whisper model: tiny/base/small/medium/large-v3/…")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Where to run (default: auto)")
    p.add_argument("--compute-type", choices=["float32","float16","int8","int8_float16"], default=None,
                   help="CTranslate2 compute type (default: auto based on device)")
    p.add_argument("--language", default=None, help="Language hint (e.g., en, es). Omit for auto-detect.")
    p.add_argument("--vad", action="store_true", help="VAD filter (reduce silence/noise)")
    p.add_argument("--word-timestamps", action="store_true", help="Emit word-level timings (slower)")
    p.add_argument("--beam-size", type=int, default=5, help="Beam search size (default 5)")
    p.add_argument("--best-of", type=int, default=5, help="Number of candidates to sample (default 5)")
    # diarization
    p.add_argument("--diarize", action="store_true", help="Enable speaker diarization (requires pyannote+torch)")
    p.add_argument("--hf-token", default=None, help="Hugging Face access token (or saved token / HF_TOKEN env var)")
    p.add_argument("--speaker-count", type=int, default=None, help="Optional fixed number of speakers (int)")
    # outputs
    p.add_argument("--emit", nargs="+", choices=["txt","json","srt","vtt"], default=["txt","srt","vtt"],
                   help="Which outputs to write (default: txt srt vtt)")
    # UI
    p.add_argument("--ui", action="store_true", help="Launch the Gradio web UI")
    return p

def run_cli(args: argparse.Namespace) -> int:
    if not args.input:
        print("No input provided. Use --ui to launch the web interface, or pass a file/folder.", file=sys.stderr)
        return 1

    inpath = Path(args.input).expanduser().resolve()
    files = find_media_inputs(inpath)
    if not files:
        print(f"No supported media found at: {inpath}", file=sys.stderr)
        print(f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTS))}", file=sys.stderr)
        return 1

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {outdir}")
    if not ffmpeg_available():
        print("Note: FFmpeg could not be auto-provisioned; attempting decode anyway.", file=sys.stderr)

    try:
        model = load_model(args.model, args.device, args.compute_type)
    except Exception as e:
        print("Failed to load faster-whisper model. Details:", file=sys.stderr)
        print(e, file=sys.stderr)
        return 2

    emit_txt = "txt" in args.emit
    emit_json = "json" in args.emit
    emit_srt = "srt" in args.emit
    emit_vtt = "vtt" in args.emit

    errors = 0
    for f in files:
        try:
            transcribe_one(
                model=model,
                model_name=args.model,
                infile=f,
                outdir=outdir,
                language=args.language,
                vad=args.vad,
                word_timestamps=args.word_timestamps,
                beam_size=args.beam_size,
                best_of=args.best_of,
                do_diar=args.diarize,
                hf_token=args.hf_token,
                speaker_count=args.speaker_count,
                emit_txt=emit_txt,
                emit_json=emit_json,
                emit_srt=emit_srt,
                emit_vtt=emit_vtt,
            )
        except Exception as e:
            errors += 1
            print(f"✗ Error transcribing {f.name}: {e}", file=sys.stderr)

    if errors:
        print(f"Completed with {errors} error(s).", file=sys.stderr)
        return 3
    print("All done. ✔")
    return 0

# =============================== UI ==========================================
def _ensure_ui():
    if not _GRADIO_OK:
        print("Gradio is not installed. Install it with:\n  pip install gradio", file=sys.stderr)
        sys.exit(4)

def build_ui():
    _ensure_ui()
    with gr.Blocks(title="All-in-One Transcriber + Diarization") as demo:
        gr.Markdown("# 🎙️ Transcriber + Diarization\nWhisper STT with optional pyannote speakers. FFmpeg auto-included.")

        with gr.Row():
            with gr.Column(scale=2):
                files = gr.File(label="Upload audio/video (multiple OK)", file_count="multiple", type="filepath")
                outdir = gr.Textbox(value="transcripts", label="Output folder")
                model = gr.Dropdown(
                    choices=["tiny","base","small","medium","large-v2","large-v3","distil-small","distil-medium","distil-large-v2"],
                    value=DEFAULT_MODEL, label="Model"
                )
                # Default to CPU to avoid CUDA errors by default
                device = gr.Radio(["auto","cpu","cuda"], value="cpu", label="Device")
                compute = gr.Dropdown(choices=[None,"float32","float16","int8","int8_float16"], value=None, label="Compute type")
                language = gr.Textbox(value="", label="Language hint (e.g., en, es). Leave empty for auto.")
                vad = gr.Checkbox(value=False, label="VAD filter")
                word_ts = gr.Checkbox(value=False, label="Word timestamps")
                beam = gr.Slider(1, 10, value=5, step=1, label="Beam size")
                best_of = gr.Slider(1, 10, value=5, step=1, label="Best of")

                gr.Markdown("### 📄 Outputs to generate")
                out_txt = gr.Checkbox(value=True, label="TXT")
                out_json = gr.Checkbox(value=True, label="JSON")
                out_srt = gr.Checkbox(value=False, label="SRT")
                out_vtt = gr.Checkbox(value=False, label="VTT")

                gr.Markdown("### 🔊 Speaker Diarization (optional)")
                diarize = gr.Checkbox(value=False, label="Enable diarization (pyannote + torch required)")
                hf_token = gr.Textbox(value="", label="Hugging Face token (saved locally)")
                with gr.Row():
                    save_tok = gr.Button("💾 Save token")
                    clear_tok = gr.Button("🗑️ Clear token")
                tok_status = gr.Markdown("")

                go = gr.Button("Transcribe", variant="primary")

            with gr.Column(scale=1):
                status = gr.Textbox(label="Status / Log", lines=18)
                file_list = gr.Files(label="Download outputs")

        def do_ui_transcribe(
            file_list_in, outdir_s, model_s, device_s, compute_s, language_s, vad_b, word_ts_b, beam_i, best_i,
            out_txt_b, out_json_b, out_srt_b, out_vtt_b,
            diar_b, hf_tok_s
        ):
            if not file_list_in:
                return "Please upload at least one file.", []
            outdir_p = Path(outdir_s).expanduser().resolve()
            outdir_p.mkdir(parents=True, exist_ok=True)
            logs = [f"Output directory: {outdir_p}"]

            if not ffmpeg_available():
                logs.append("Note: FFmpeg could not be auto-provisioned; attempting decode anyway.")

            try:
                mdl = load_model(model_s, device_s, None if compute_s in (None, "None", "") else str(compute_s))
            except Exception as e:
                return f"Failed to load model: {e}", []

            if diar_b:
                # Use explicitly provided token or saved/env token
                token = (hf_tok_s or get_saved_token())
                if not _HAS_PYANNOTE:
                    return "Diarization requested but pyannote.audio (and torch) not installed.", []
                if not token:
                    return "Please provide or save a Hugging Face token for diarization.", []
                # Set it for the current run so pyannote sees it even if not passed
                os.environ["HF_TOKEN"] = token

            written: List[Path] = []
            for fp in file_list_in:
                p = Path(fp)
                try:
                    logs.append(f"→ Transcribing: {p.name}")
                    outs = transcribe_one(
                        model=mdl,
                        model_name=model_s,
                        infile=p,
                        outdir=outdir_p,
                        language=(language_s or None),
                        vad=bool(vad_b),
                        word_timestamps=bool(word_ts_b),
                        beam_size=int(beam_i),
                        best_of=int(best_i),
                        do_diar=bool(diar_b),
                        hf_token=(hf_tok_s or None),
                        speaker_count=None,  # UI keeps this simple; add a Number if you want
                        emit_txt=bool(out_txt_b),
                        emit_json=bool(out_json_b),
                        emit_srt=bool(out_srt_b),
                        emit_vtt=bool(out_vtt_b),
                    )
                    written.extend(outs)
                except Exception as e:
                    logs.append(f"✗ Error on {p.name}: {e}")

            if not written:
                return "\n".join(logs + ["No outputs were written."]), []

            # Return list of files directly; gr.Files will render each as an individual download
            logs.append(f"✓ Ready: {len(written)} file(s)")
            return "\n".join(logs), [str(p) for p in written]

        def _save_token_ui(token_in: str):
            if not token_in or not token_in.strip():
                return "⚠️ No token provided."
            set_saved_token(token_in.strip())
            return "✅ Token saved."

        def _clear_token_ui():
            set_saved_token(None)
            return "🗑️ Token cleared."

        def _load_token_on_start():
            return get_saved_token() or ""

        # Wire up events
        go.click(
            do_ui_transcribe,
            inputs=[files, outdir, model, device, compute, language, vad, word_ts, beam, best_of,
                    out_txt, out_json, out_srt, out_vtt,
                    diarize, hf_token],
            outputs=[status, file_list]
        )

        save_tok.click(_save_token_ui, inputs=[hf_token], outputs=[tok_status])
        clear_tok.click(_clear_token_ui, outputs=[tok_status])

        # Prefill token from saved config/env when UI loads
        demo.load(_load_token_on_start, inputs=None, outputs=[hf_token])

    return demo

# ============================== entrypoint ===================================
def main(argv: Optional[List[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    if args.ui:
        if not _GRADIO_OK:
            print("Gradio is required for the UI. Install it with: pip install gradio", file=sys.stderr)
            return 4
        demo = build_ui()
        demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
        return 0

    # -------- CLI path --------
    inpath = Path(args.input).expanduser().resolve() if args.input else None
    if not inpath:
        print("No input provided. Use --ui or pass a file/folder.", file=sys.stderr)
        return 1

    files = find_media_inputs(inpath)
    if not files:
        print(f"No supported media found at: {inpath}", file=sys.stderr)
        print(f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTS))}", file=sys.stderr)
        return 1

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {outdir}")
    if not ffmpeg_available():
        print("Note: FFmpeg could not be auto-provisioned; attempting decode anyway.", file=sys.stderr)

    try:
        model = load_model(args.model, args.device, args.compute_type)
    except Exception as e:
        print("Failed to load faster-whisper model. Details:", file=sys.stderr)
        print(e, file=sys.stderr)
        return 2

    emit_txt = "txt" in args.emit
    emit_json = "json" in args.emit
    emit_srt = "srt" in args.emit
    emit_vtt = "vtt" in args.emit

    errors = 0
    for f in files:
        try:
            transcribe_one(
                model=model,
                model_name=args.model,
                infile=f,
                outdir=outdir,
                language=args.language,
                vad=args.vad,
                word_timestamps=args.word_timestamps,
                beam_size=args.beam_size,
                best_of=args.best_of,
                do_diar=args.diarize,
                hf_token=args.hf_token,
                speaker_count=args.speaker_count,
                emit_txt=emit_txt,
                emit_json=emit_json,
                emit_srt=emit_srt,
                emit_vtt=emit_vtt,
            )
        except Exception as e:
            errors += 1
            print(f"✗ Error transcribing {f.name}: {e}", file=sys.stderr)

    if errors:
        print(f"Completed with {errors} error(s).", file=sys.stderr)
        return 3
    print("All done. ✔")
    return 0

if __name__ == "__main__":
    sys.exit(main())

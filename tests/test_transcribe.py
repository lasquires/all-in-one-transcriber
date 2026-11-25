"""Offline regression checks for model routing and output compatibility."""
import unittest
from unittest.mock import patch, MagicMock
import transcribe as app


class DeviceTests(unittest.TestCase):
    @patch.object(app.ctranslate2, "get_cuda_device_count", return_value=0)
    def test_cpu_fallback(self, count):
        self.assertEqual(app.resolve_device_and_compute("auto", "float16", []), ("cpu", "int8"))

    @patch.object(app.ctranslate2, "get_cuda_device_count", side_effect=RuntimeError("missing CUDA"))
    def test_broken_driver(self, count):
        self.assertEqual(app.resolve_device_and_compute("cuda", "auto", []), ("cpu", "int8"))

    @patch.object(app.subprocess, "run", side_effect=FileNotFoundError)
    @patch.object(app.ctranslate2, "get_supported_compute_types", return_value={"float16", "int8_float16"})
    @patch.object(app.ctranslate2, "get_cuda_device_count", return_value=1)
    def test_gpu_auto_without_smi(self, count, types, run):
        self.assertEqual(app.resolve_device_and_compute("auto", "auto", []), ("cuda", "int8_float16"))

    @patch.object(app.ctranslate2, "get_cuda_device_count")
    def test_manual_cpu(self, count):
        self.assertEqual(app.resolve_device_and_compute("cpu", "float32", []), ("cpu", "float32"))
        count.assert_not_called()


class ModelTests(unittest.TestCase):
    def setUp(self):
        app.MODEL_CACHE.clear()

    def tearDown(self):
        app.MODEL_CACHE.clear()

    @patch.object(app, "WhisperModel")
    def test_new_models_and_cache(self, model):
        for name in ("turbo", "distil-large-v3.5"):
            app.load_whisper(name, "cpu", "int8")
            model.assert_called_with(app.MODEL_IDS[name], device="cpu", compute_type="int8")
            app.load_whisper(name, "cpu", "int8")
        self.assertEqual(model.call_count, 2)

    def test_exports(self):
        segments = [{"start": 0.0, "end": 1.5, "text": "hello", "speaker": "SPK0"}]
        self.assertIn("00:00:01,500", app.write_srt(segments, True))
        self.assertIn("SPK0: hello", app.write_srt(segments, True))
        self.assertIn("WEBVTT", app.write_vtt(segments, False))
        self.assertNotIn("SPK0", app.write_vtt(segments, False))

    def test_asr_generator(self):
        model = MagicMock()
        seg = MagicMock(start=0, end=1, text=" hello ", words=[])
        info = MagicMock(duration=1, language="en")
        model.transcribe.return_value = (iter([seg]), info)
        result, got_info = app.read_asr(model, "sample.wav", "auto", True, False, 5)
        self.assertEqual(result, [{"start": 0.0, "end": 1.0, "text": "hello"}])
        self.assertIs(got_info, info)



class ConsentTests(unittest.TestCase):
    def invoke(self, consent=False, device="auto", model="auto"):
        return app.transcribe(["test.wav"], model, "en", False, "", False,
                              device, "auto", True, False, 5, consent)

    @patch.object(app, "ffmpeg_installed", return_value=True)
    @patch.object(app, "resolve_device_and_compute", return_value=("cpu", "int8"))
    @patch.object(app, "load_whisper")
    def test_no_gpu_blocks_before_model_load(self, load, resolve, ffmpeg):
        result = self.invoke()
        load.assert_not_called()
        self.assertEqual(len(result), 10)
        self.assertIn("CPU processing requires your approval", result[8])
        self.assertFalse(result[9])

    @patch.object(app, "ffmpeg_installed", return_value=True)
    @patch.object(app, "resolve_device_and_compute", return_value=("cpu", "int8"))
    @patch.object(app, "load_whisper", side_effect=RuntimeError("test stop"))
    def test_approved_cpu_uses_small(self, load, resolve, ffmpeg):
        result = self.invoke(True)
        load.assert_called_once_with("small", "cpu", "int8")
        self.assertFalse(result[9])

    @patch.object(app, "ffmpeg_installed", return_value=True)
    @patch.object(app, "resolve_device_and_compute", return_value=("cuda", "int8_float16"))
    @patch.object(app, "load_whisper", side_effect=RuntimeError("missing GPU DLL"))
    def test_gpu_load_failure_does_not_silently_retry(self, load, resolve, ffmpeg):
        result = self.invoke()
        load.assert_called_once_with("turbo", "cuda", "int8_float16")
        self.assertIn("GPU model initialization failed", result[8])
        self.assertFalse(result[9])

    @patch.object(app, "ffmpeg_installed", return_value=True)
    @patch.object(app, "resolve_device_and_compute", return_value=("cuda", "int8_float16"))
    @patch.object(app, "load_whisper", side_effect=[RuntimeError("GPU failed"), RuntimeError("test stop")])
    def test_approved_gpu_failure_retries_small_on_cpu(self, load, resolve, ffmpeg):
        result = self.invoke(True)
        self.assertEqual(load.call_args_list[-1].args, ("small", "cpu", "int8"))
        self.assertFalse(result[9])

    @patch.object(app, "ffmpeg_installed", return_value=True)
    @patch.object(app, "resolve_device_and_compute", return_value=("cuda", "int8_float16"))
    @patch.object(app, "load_whisper")
    @patch.object(app, "read_asr", side_effect=RuntimeError("GPU inference failed"))
    @patch.object(app, "prepare_audio_for_processing", return_value="test.wav")
    def test_gpu_inference_failure_does_not_retry_without_consent(self, prep, read, load, resolve, ffmpeg):
        result = self.invoke()
        load.assert_called_once()
        read.assert_called_once()
        self.assertIn("GPU inference failed", result[8])
        self.assertFalse(result[9])

    @patch.object(app, "resolve_device_and_compute", return_value=("cpu", "int8"))
    def test_recheck_resets_permission(self, resolve):
        message, consent = app.refresh_hardware("auto", False)
        self.assertIn("GPU acceleration unavailable", message)
        self.assertFalse(consent)

if __name__ == "__main__":
    unittest.main()

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


if __name__ == "__main__":
    unittest.main()

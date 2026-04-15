"""
Tests for server.py — covers all testable logic paths.
SDK-dependent paths are tested via mocking; no live API key required.
"""

import io
import os
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch, PropertyMock


class TestSuppressOutput(unittest.TestCase):
    """_suppress_output redirects stdout and stderr, restores on exit."""

    def setUp(self):
        # Reset module-level singletons before each test
        import server
        server._rf = None
        server._ws = None

    def test_suppresses_stdout(self):
        import server
        original_stdout = sys.stdout
        with server._suppress_output():
            print("this should be suppressed")
            self.assertIsNot(sys.stdout, original_stdout)
        self.assertIs(sys.stdout, original_stdout)

    def test_suppresses_stderr(self):
        import server
        original_stderr = sys.stderr
        with server._suppress_output():
            self.assertIsNot(sys.stderr, original_stderr)
        self.assertIs(sys.stderr, original_stderr)

    def test_restores_on_exception(self):
        import server
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            with server._suppress_output():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        self.assertIs(sys.stdout, original_stdout)
        self.assertIs(sys.stderr, original_stderr)


class TestGetRf(unittest.TestCase):
    """get_rf() lazy init and auth validation."""

    def setUp(self):
        import server
        server._rf = None
        server._ws = None

    def test_raises_when_no_api_key(self):
        import server
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ROBOFLOW_API_KEY", None)
            with self.assertRaises(ValueError) as ctx:
                server.get_rf()
            self.assertIn("ROBOFLOW_API_KEY", str(ctx.exception))

    def test_initializes_once(self):
        import server
        mock_rf_instance = MagicMock()
        mock_roboflow = MagicMock()
        mock_roboflow.Roboflow.return_value = mock_rf_instance

        with patch.dict(os.environ, {"ROBOFLOW_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"roboflow": mock_roboflow}):
                rf1 = server.get_rf()
                rf2 = server.get_rf()

        self.assertIs(rf1, rf2)
        mock_roboflow.Roboflow.assert_called_once_with(api_key="test-key")

    def test_thread_safe_init(self):
        """Concurrent calls must initialize exactly once."""
        import server
        mock_rf_instance = MagicMock()
        mock_roboflow = MagicMock()
        mock_roboflow.Roboflow.return_value = mock_rf_instance
        results = []

        def call_get_rf():
            results.append(server.get_rf())

        with patch.dict(os.environ, {"ROBOFLOW_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"roboflow": mock_roboflow}):
                threads = [threading.Thread(target=call_get_rf) for _ in range(10)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

        self.assertEqual(len(results), 10)
        self.assertTrue(all(r is mock_rf_instance for r in results))
        mock_roboflow.Roboflow.assert_called_once()


class TestGetWs(unittest.TestCase):
    """get_ws() caching and custom workspace_url behaviour."""

    def setUp(self):
        import server
        server._rf = None
        server._ws = None

    def _mock_rf(self, server):
        mock_ws = MagicMock()
        mock_rf = MagicMock()
        mock_rf.workspace.return_value = mock_ws
        server._rf = mock_rf
        return mock_rf, mock_ws

    def test_caches_default_workspace(self):
        import server
        mock_rf, mock_ws = self._mock_rf(server)
        ws1 = server.get_ws()
        ws2 = server.get_ws()
        self.assertIs(ws1, ws2)
        mock_rf.workspace.assert_called_once()

    def test_custom_workspace_url_bypasses_cache(self):
        import server
        mock_rf, _ = self._mock_rf(server)
        server.get_ws("custom-workspace")
        server.get_ws("custom-workspace")
        self.assertEqual(mock_rf.workspace.call_count, 2)


class TestUploadImage(unittest.TestCase):
    """upload_image split validation."""

    def setUp(self):
        import server
        server._rf = None
        server._ws = None

    def test_invalid_split_raises(self):
        import server
        with self.assertRaises(ValueError) as ctx:
            server.upload_image("my-project", "/some/image.jpg", split="foo")
        self.assertIn("foo", str(ctx.exception))
        self.assertIn("train", str(ctx.exception))

    def test_valid_splits_pass_validation(self):
        import server
        mock_proj = MagicMock()
        mock_proj.upload.return_value = "ok"
        mock_ws = MagicMock()
        mock_ws.project.return_value = mock_proj

        with patch("server.get_ws", return_value=mock_ws):
            for split in ("train", "valid", "test"):
                with server._suppress_output():
                    result = server.upload_image("proj", "/img.jpg", split=split)
                self.assertTrue(result["success"])


class TestCreateVersion(unittest.TestCase):
    """create_version mutable default arg fix."""

    def setUp(self):
        import server
        server._rf = None
        server._ws = None

    def test_none_defaults_do_not_share_state(self):
        import server
        import inspect
        sig = inspect.signature(server.create_version)
        self.assertIsNone(sig.parameters["preprocessing"].default)
        self.assertIsNone(sig.parameters["augmentation"].default)

    def test_empty_call_passes_empty_settings(self):
        import server
        mock_proj = MagicMock()
        mock_proj.generate_version.return_value = MagicMock()
        mock_ws = MagicMock()
        mock_ws.project.return_value = mock_proj

        with patch("server.get_ws", return_value=mock_ws):
            with server._suppress_output():
                server.create_version("my-project")

        mock_proj.generate_version.assert_called_once_with({})

    def test_preprocessing_passed_through(self):
        import server
        mock_proj = MagicMock()
        mock_ws = MagicMock()
        mock_ws.project.return_value = mock_proj

        pp = {"auto-orient": True}
        with patch("server.get_ws", return_value=mock_ws):
            with server._suppress_output():
                server.create_version("my-project", preprocessing=pp)

        call_args = mock_proj.generate_version.call_args[0][0]
        self.assertEqual(call_args["preprocessing"], pp)
        self.assertNotIn("augmentation", call_args)


class TestRunInference(unittest.TestCase):
    """run_inference null model guard."""

    def setUp(self):
        import server
        server._rf = None
        server._ws = None

    def test_raises_when_model_is_none(self):
        import server
        mock_ver = MagicMock()
        mock_ver.model = None
        mock_proj = MagicMock()
        mock_proj.version.return_value = mock_ver
        mock_ws = MagicMock()
        mock_ws.project.return_value = mock_proj

        with patch("server.get_ws", return_value=mock_ws):
            with self.assertRaises(ValueError) as ctx:
                with server._suppress_output():
                    server.run_inference("my-project", 1, "/img.jpg")
        self.assertIn("No trained model", str(ctx.exception))
        self.assertIn("1", str(ctx.exception))

    def test_calls_predict_when_model_exists(self):
        import server
        mock_predictions = MagicMock()
        mock_predictions.json.return_value = {"predictions": []}
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_predictions
        mock_ver = MagicMock()
        mock_ver.model = mock_model
        mock_proj = MagicMock()
        mock_proj.version.return_value = mock_ver
        mock_ws = MagicMock()
        mock_ws.project.return_value = mock_proj

        with patch("server.get_ws", return_value=mock_ws):
            with server._suppress_output():
                result = server.run_inference("my-project", 2, "/img.jpg", confidence=70)

        mock_model.predict.assert_called_once_with("/img.jpg", confidence=70, overlap=50)
        self.assertEqual(result, {"predictions": []})


class TestSearchUniverse(unittest.TestCase):
    """search_universe auth consistency."""

    def setUp(self):
        import server
        server._rf = None
        server._ws = None

    def test_raises_when_no_api_key(self):
        import server
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ROBOFLOW_API_KEY", None)
            with self.assertRaises(ValueError):
                server.search_universe("hard hat")

    def test_returns_formatted_results(self):
        import server
        server._rf = MagicMock()  # skip auth
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"name": "Hard Hat", "url": "workspace/hard-hat", "type": "object-detection",
                 "images": 500, "classes": ["hard-hat"]}
            ]
        }
        with patch.dict(os.environ, {"ROBOFLOW_API_KEY": "test-key"}):
            with patch("server.requests.get", return_value=mock_response):
                results = server.search_universe("hard hat")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Hard Hat")
        self.assertIn("universe_url", results[0])
        self.assertIn("universe.roboflow.com", results[0]["universe_url"])


class TestGetModelMetrics(unittest.TestCase):
    """get_model_metrics auth consistency and response parsing."""

    def setUp(self):
        import server
        server._rf = None
        server._ws = None

    def test_raises_when_no_api_key(self):
        import server
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ROBOFLOW_API_KEY", None)
            with self.assertRaises(ValueError):
                server.get_model_metrics("my-project", 1)

    def test_parses_metrics(self):
        import server
        server._rf = MagicMock()
        mock_ws = MagicMock()
        mock_ws.url = "my-workspace"
        server._ws = mock_ws
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "version": {
                "model": {"map": 0.87, "precision": 0.91, "recall": 0.84},
                "training": {"status": "trained"}
            }
        }
        with patch.dict(os.environ, {"ROBOFLOW_API_KEY": "test-key"}):
            with patch("server.requests.get", return_value=mock_response):
                result = server.get_model_metrics("my-project", 3)

        self.assertEqual(result["mAP"], 0.87)
        self.assertEqual(result["precision"], 0.91)
        self.assertEqual(result["recall"], 0.84)
        self.assertEqual(result["training_status"], "trained")
        self.assertEqual(result["version"], 3)


class TestModelFormatParam(unittest.TestCase):
    """model_format parameter rename — no 'format' builtin shadowing."""

    def test_download_dataset_uses_model_format(self):
        import server
        import inspect
        sig = inspect.signature(server.download_dataset)
        self.assertIn("model_format", sig.parameters)
        self.assertNotIn("format", sig.parameters)

    def test_download_universe_dataset_uses_model_format(self):
        import server
        import inspect
        sig = inspect.signature(server.download_universe_dataset)
        self.assertIn("model_format", sig.parameters)
        self.assertNotIn("format", sig.parameters)


if __name__ == "__main__":
    unittest.main(verbosity=2)

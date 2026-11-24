"""Unit tests for gpu-memory-guard.

Tests use mocked GPU data so they run on any machine,
including CI environments without a physical GPU.
"""

import json
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, "..")
from gpu_guard import (
    GPUInfo,
    check_vram,
    can_load_model,
    format_human_output,
    format_json_output,
    get_gpu_info,
)


@pytest.fixture
def single_gpu():
    return [
        GPUInfo(
            device_id=0,
            name="NVIDIA GeForce RTX 4090",
            total_memory_gb=24.0,
            used_memory_gb=8.0,
            available_memory_gb=16.0,
            utilization_percent=35.0,
        )
    ]


@pytest.fixture
def dual_gpu():
    return [
        GPUInfo(
            device_id=0,
            name="NVIDIA GeForce RTX 4090",
            total_memory_gb=24.0,
            used_memory_gb=10.0,
            available_memory_gb=14.0,
            utilization_percent=40.0,
        ),
        GPUInfo(
            device_id=1,
            name="NVIDIA GeForce RTX 4080",
            total_memory_gb=16.0,
            used_memory_gb=4.0,
            available_memory_gb=12.0,
            utilization_percent=20.0,
        ),
    ]


@pytest.fixture
def rtx_5090():
    return [
        GPUInfo(
            device_id=0,
            name="NVIDIA GeForce RTX 5090",
            total_memory_gb=32.0,
            used_memory_gb=2.0,
            available_memory_gb=30.0,
            utilization_percent=5.0,
        )
    ]


class TestGPUInfo:
    def test_fields(self, single_gpu):
        gpu = single_gpu[0]
        assert gpu.device_id == 0
        assert gpu.name == "NVIDIA GeForce RTX 4090"
        assert gpu.total_memory_gb == 24.0
        assert gpu.available_memory_gb == 16.0

    def test_memory_math(self, single_gpu):
        gpu = single_gpu[0]
        expected = gpu.total_memory_gb - gpu.used_memory_gb
        assert abs(expected - gpu.available_memory_gb) < 0.01


class TestCheckVram:
    @patch("gpu_guard.get_gpu_info")
    def test_model_fits(self, mock_info, single_gpu):
        mock_info.return_value = single_gpu
        fits, msg = check_vram(model_size_gb=12.0, buffer_gb=1.0)
        assert fits is True

    @patch("gpu_guard.get_gpu_info")
    def test_model_too_large(self, mock_info, single_gpu):
        mock_info.return_value = single_gpu
        fits, _ = check_vram(model_size_gb=20.0, buffer_gb=1.0)
        assert fits is False

    @patch("gpu_guard.get_gpu_info")
    def test_exact_fit(self, mock_info, single_gpu):
        mock_info.return_value = single_gpu
        fits, _ = check_vram(model_size_gb=15.5, buffer_gb=0.5)
        assert fits is True

    @patch("gpu_guard.get_gpu_info")
    def test_no_gpu_detected(self, mock_info):
        mock_info.return_value = None
        fits, msg = check_vram(model_size_gb=8.0)
        assert fits is False
        assert "Unable to detect" in msg

    @patch("gpu_guard.get_gpu_info")
    def test_empty_gpu_list(self, mock_info):
        mock_info.return_value = []
        fits, msg = check_vram(model_size_gb=8.0)
        assert fits is False

    @patch("gpu_guard.get_gpu_info")
    def test_multi_gpu_aggregation(self, mock_info, dual_gpu):
        mock_info.return_value = dual_gpu
        fits, msg = check_vram(model_size_gb=20.0, buffer_gb=1.0)
        assert fits is True


class TestCanLoadModel:
    @patch("gpu_guard.get_gpu_info")
    def test_returns_bool(self, mock_info, single_gpu):
        mock_info.return_value = single_gpu
        result = can_load_model(8.0)
        assert isinstance(result, bool)
        assert result is True

    @patch("gpu_guard.get_gpu_info")
    def test_too_large(self, mock_info, single_gpu):
        mock_info.return_value = single_gpu
        assert can_load_model(20.0) is False


class TestFormatHumanOutput:
    def test_contains_gpu_name(self, single_gpu):
        out = format_human_output(single_gpu)
        assert "RTX 4090" in out

    def test_model_fits_message(self, single_gpu):
        out = format_human_output(single_gpu, model_size_gb=8.0)
        assert "Estimated to fit" in out

    def test_model_too_large_message(self, single_gpu):
        out = format_human_output(single_gpu, model_size_gb=20.0)
        assert "Estimated NOT to fit" in out


class TestFormatJsonOutput:
    def test_valid_json(self, single_gpu):
        raw = format_json_output(single_gpu)
        data = json.loads(raw)
        assert "gpus" in data
        assert "total_available_gb" in data

    def test_model_check_fields(self, single_gpu):
        raw = format_json_output(single_gpu, model_size_gb=10.0, buffer_gb=1.0)
        data = json.loads(raw)
        assert data["can_fit"] is True
        assert data["total_required_gb"] == 11.0

    def test_gpu_count(self, dual_gpu):
        raw = format_json_output(dual_gpu)
        data = json.loads(raw)
        assert len(data["gpus"]) == 2


class TestEdgeCases:
    @patch("gpu_guard.get_gpu_info")
    def test_zero_model_size(self, mock_info, single_gpu):
        mock_info.return_value = single_gpu
        fits, _ = check_vram(model_size_gb=0.0)
        assert fits is True

    @patch("gpu_guard.get_gpu_info")
    def test_large_buffer(self, mock_info, single_gpu):
        mock_info.return_value = single_gpu
        fits, _ = check_vram(model_size_gb=10.0, buffer_gb=10.0)
        assert fits is False

    @patch("gpu_guard.get_gpu_info")
    def test_rtx_5090_large_model(self, mock_info, rtx_5090):
        mock_info.return_value = rtx_5090
        fits, _ = check_vram(model_size_gb=27.0, buffer_gb=2.0)
        assert fits is True


class TestMultiGpuNote:
    def test_note_for_multiple_gpus(self, dual_gpu):
        assert "needs the space on one card" in format_human_output(dual_gpu, model_size_gb=8.0)

    def test_no_note_for_single_gpu(self, single_gpu):
        assert "needs the space on one card" not in format_human_output(single_gpu, model_size_gb=8.0)


class TestPynvmlName:
    def _fake(self, name):
        import types
        return types.SimpleNamespace(
            nvmlInit=lambda: None, nvmlShutdown=lambda: None, nvmlDeviceGetCount=lambda: 1,
            nvmlDeviceGetHandleByIndex=lambda i: i, nvmlDeviceGetName=lambda h: name,
            nvmlDeviceGetMemoryInfo=lambda h: types.SimpleNamespace(total=32 * 1024**3, used=2 * 1024**3, free=30 * 1024**3),
            nvmlDeviceGetUtilizationRates=lambda h: types.SimpleNamespace(gpu=5),
        )

    def test_str_name(self):
        from gpu_guard import get_pynvml_info
        with patch.dict(sys.modules, {"pynvml": self._fake("NVIDIA Test GPU")}):
            info = get_pynvml_info()
        assert info is not None and info[0].name == "NVIDIA Test GPU"

    def test_bytes_name(self):
        from gpu_guard import get_pynvml_info
        with patch.dict(sys.modules, {"pynvml": self._fake(b"NVIDIA Test GPU")}):
            info = get_pynvml_info()
        assert info is not None and info[0].name == "NVIDIA Test GPU"


class TestExitCodes:
    def _run(self, argv, gpus):
        from gpu_guard import main
        with patch("gpu_guard.get_gpu_info", return_value=gpus), patch.object(sys, "argv", ["gpu-guard"] + argv):
            with pytest.raises(SystemExit) as e:
                main()
        return e.value.code

    def test_fits(self, single_gpu):
        assert self._run(["--model-size", "8", "--quiet"], single_gpu) == 0

    def test_does_not_fit(self, single_gpu):
        assert self._run(["--model-size", "20", "--quiet"], single_gpu) == 1

    def test_no_gpu(self):
        assert self._run(["--model-size", "8", "--quiet"], None) == 2

    def test_status_only(self, single_gpu):
        assert self._run([], single_gpu) == 0

    def test_non_utf8_stdout_does_not_change_exit_code(self, single_gpu):
        import io
        from gpu_guard import main
        out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        with patch("gpu_guard.get_gpu_info", return_value=single_gpu),                 patch.object(sys, "argv", ["gpu-guard", "--model-size", "8"]), patch.object(sys, "stdout", out):
            with pytest.raises(SystemExit) as e:
                main()
        assert e.value.code == 0


class TestPerGpu:
    """Two cards with 6 and 10 GB free: 16 GB in total, 10 GB on the best card."""

    def _gpus(self):
        return [GPUInfo(0, "A", 24.0, 18.0, 6.0, 0.0), GPUInfo(1, "B", 24.0, 14.0, 10.0, 0.0)]

    def _main(self, argv):
        from gpu_guard import main
        with patch("gpu_guard.get_gpu_info", return_value=self._gpus()), \
                patch.object(sys, "argv", ["gpu-guard"] + argv):
            with pytest.raises(SystemExit) as e:
                main()
        return e.value.code

    def test_sum_fits_but_no_single_card_does(self):
        with patch("gpu_guard.get_gpu_info", return_value=self._gpus()):
            assert can_load_model(12.0, 0.5) is True
            assert can_load_model(12.0, 0.5, per_gpu=True) is False

    def test_fits_on_the_card_with_most_free_memory(self):
        with patch("gpu_guard.get_gpu_info", return_value=self._gpus()):
            fits, message = check_vram(9.5, 0.5, per_gpu=True)
        assert fits is True and "GPU 1" in message

    def test_exit_codes(self):
        assert self._main(["--model-size", "12", "--quiet"]) == 0
        assert self._main(["--model-size", "12", "--quiet", "--per-gpu"]) == 1
        assert self._main(["--model-size", "9", "--quiet", "--per-gpu"]) == 0

    def test_human_and_json_agree(self, capsys):
        assert self._main(["--model-size", "12", "--per-gpu"]) == 1
        out = capsys.readouterr().out
        assert "Estimated NOT to fit (need 2.50GB more)" in out and "this is a sum" not in out
        assert self._main(["--model-size", "12", "--per-gpu", "--json"]) == 1
        data = json.loads(capsys.readouterr().out)
        assert data["can_fit"] is False and data["best_gpu_id"] == 1

    def test_json_without_the_flag_is_unchanged(self):
        data = json.loads(format_json_output(self._gpus(), 12.0))
        assert data["can_fit"] is True and "per_gpu" not in data

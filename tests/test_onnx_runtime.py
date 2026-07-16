"""ONNX provider selection — pure logic, no onnxruntime import."""
from cruze.perception.onnx_runtime import select_providers


def test_cpu_only():
    assert select_providers("cpu", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]


def test_cuda_available():
    got = select_providers("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert got == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_cuda_unavailable_falls_back_to_cpu():
    assert select_providers("cuda", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]


def test_tensorrt_prefers_trt_then_cuda():
    avail = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    assert select_providers("tensorrt", avail)[0] == "TensorrtExecutionProvider"


def test_unknown_provider_is_cpu():
    assert select_providers("bogus", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]

"""Thin wrapper over onnxruntime.InferenceSession with config-driven execution
providers and graceful CPU fallback. onnxruntime is an optional dependency —
imported lazily so this module (and select_providers) stay import-clean for tests."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_PROVIDER_MAP: dict[str, list[str]] = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
}


def select_providers(provider: str, available: list[str]) -> list[str]:
    """Ordered EP list for the requested provider, filtered to what's installed,
    always ending in CPU. Unknown provider → CPU. Logs a warning on fallback."""
    key = provider.lower()
    if key not in _PROVIDER_MAP:
        logger.warning(
            "ONNX provider '%s' not recognized (typo in config?); falling back to CPU", provider
        )
    wanted = _PROVIDER_MAP.get(key, ["CPUExecutionProvider"])
    chosen = [p for p in wanted if p in available]
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    if chosen[0] != wanted[0]:
        logger.warning("ONNX provider '%s' unavailable; falling back to %s", provider, chosen[0])
    return chosen


class OnnxSession:
    def __init__(self, model_path: str, provider: str = "cpu") -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "onnxruntime required for ONNX model backends. "
                "Install with: pip install 'cruze[onnx]'"
            ) from exc
        providers = select_providers(provider, ort.get_available_providers())
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]
        logger.info("OnnxSession %s providers=%s in=%s out=%s",
                    model_path, providers, self._input_names, self._output_names)

    @property
    def input_names(self) -> list[str]:
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        return self._output_names

    def run(self, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        return self._session.run(self._output_names, feeds)

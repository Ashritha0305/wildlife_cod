"""CPU-first monocular depth estimation with an optional ZoeDepth backend."""

from typing import Any, Optional

import numpy as np


class DepthEstimator:
    """Estimate a per-pixel depth map from an OpenCV BGR image.

    The Transformers backend loads ``Intel/zoedepth-nyu-kitti`` lazily. The
    ``fallback`` backend is intended for offline tests and development only;
    it produces a normalized relative-depth proxy, not metric distance.
    """

    DEFAULT_MODEL = "Intel/zoedepth-nyu-kitti"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        backend: str = "auto",
        allow_fallback: bool = True,
    ) -> None:
        if backend not in {"auto", "zoedepth", "fallback"}:
            raise ValueError("backend must be 'auto', 'zoedepth', or 'fallback'")
        if device not in {"cpu", "cuda", "cuda:0"}:
            raise ValueError("device must be 'cpu', 'cuda', or 'cuda:0'")
        if backend == "fallback" and not allow_fallback:
            raise ValueError("allow_fallback must be True when backend='fallback'")
        self.model_name = model_name
        self.device = device
        self.backend = backend
        self.allow_fallback = allow_fallback
        self._processor: Optional[Any] = None
        self._model: Optional[Any] = None
        self._resolved_backend: Optional[str] = None

    def _validate_frame(self, frame: Any) -> tuple[int, int]:
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"frame must be a numpy.ndarray, got {type(frame).__name__}")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape (height, width, 3)")
        height, width = frame.shape[:2]
        if height < 2 or width < 2 or frame.size == 0:
            raise ValueError("frame must be a non-empty image at least 2x2 pixels")
        if not np.issubdtype(frame.dtype, np.number):
            raise TypeError("frame must contain numeric pixel values")
        if not np.isfinite(frame).all():
            raise ValueError("frame must contain only finite pixel values")
        return height, width

    def _load_zoedepth(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation
        except ImportError as error:
            raise RuntimeError(
                "ZoeDepth requires compatible 'torch' and 'transformers' packages"
            ) from error

        requested_device = "cuda" if self.device.startswith("cuda") else "cpu"
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = ZoeDepthForDepthEstimation.from_pretrained(self.model_name)
        self._model.to(requested_device)
        self._model.eval()
        self._resolved_backend = "zoedepth"

    @staticmethod
    def _resize_nearest(depth: np.ndarray, height: int, width: int) -> np.ndarray:
        row_indices = np.minimum((np.arange(height) * depth.shape[0] // height), depth.shape[0] - 1)
        column_indices = np.minimum((np.arange(width) * depth.shape[1] // width), depth.shape[1] - 1)
        return depth[row_indices[:, None], column_indices[None, :]]

    def _fallback_depth(self, frame: np.ndarray) -> np.ndarray:
        gray = frame.astype(np.float32).mean(axis=2)
        minimum = float(gray.min())
        maximum = float(gray.max())
        if maximum == minimum:
            return np.ones(gray.shape, dtype=np.float32)
        return ((gray - minimum) / (maximum - minimum)).astype(np.float32)

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """Return a float32 depth map with the same height and width as frame."""
        height, width = self._validate_frame(frame)
        if self.backend == "fallback":
            self._resolved_backend = "fallback"
            return self._fallback_depth(frame)

        if self._resolved_backend is None:
            try:
                self._load_zoedepth()
            except (ImportError, OSError, RuntimeError):
                if not self.allow_fallback or self.backend == "zoedepth":
                    raise
                self._resolved_backend = "fallback"

        if self._resolved_backend == "fallback":
            return self._fallback_depth(frame)

        import torch

        rgb_frame = frame[:, :, ::-1]
        inputs = self._processor(images=rgb_frame, return_tensors="pt")
        inputs = {name: value.to(self._model.device) for name, value in inputs.items()}
        with torch.no_grad():
            predicted_depth = self._model(**inputs).predicted_depth
        depth = predicted_depth.squeeze().detach().cpu().numpy().astype(np.float32)
        if depth.shape != (height, width):
            depth = self._resize_nearest(depth, height, width)
        return depth

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.estimate(frame)
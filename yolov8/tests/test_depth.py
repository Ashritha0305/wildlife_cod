import numpy as np
import pytest

from yolov8.src.depth import DepthEstimator


def test_initialization_is_lazy():
    estimator = DepthEstimator(backend="fallback")

    assert estimator.device == "cpu"
    assert estimator.model_name == DepthEstimator.DEFAULT_MODEL
    assert estimator._model is None
    assert estimator._processor is None


def test_valid_image_returns_float32_depth_map():
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    frame[:, 16:, :] = 255
    depth = DepthEstimator(backend="fallback").estimate(frame)

    assert isinstance(depth, np.ndarray)
    assert depth.shape == (24, 32)
    assert depth.dtype == np.float32
    assert np.isfinite(depth).all()


def test_invalid_inputs_are_rejected():
    estimator = DepthEstimator(backend="fallback")

    with pytest.raises(TypeError):
        estimator.estimate(None)
    with pytest.raises(ValueError):
        estimator.estimate(np.zeros((24, 32), dtype=np.uint8))
    with pytest.raises(ValueError):
        estimator.estimate(np.zeros((24, 32, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        estimator.estimate(np.full((24, 32, 3), np.nan, dtype=np.float32))


def test_zoedepth_backend_does_not_load_during_initialization():
    estimator = DepthEstimator(backend="zoedepth")

    assert estimator._model is None
    assert estimator._processor is None
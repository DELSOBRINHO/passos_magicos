from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))

from risk_calibration import apply_probability_calibration, get_probability_calibration


def test_default_calibration_matches_business_anchors():
    meta = {'probability_calibration': {'temperature': 3.5, 'bias': 3.8}}

    strong = apply_probability_calibration(2.65341603257061e-07, meta)
    intermediate = apply_probability_calibration(0.9942251004550335, meta)
    critical = apply_probability_calibration(0.9999999999965472, meta)

    assert 0.0 <= strong <= 0.05
    assert 0.50 <= intermediate <= 0.65
    assert 0.90 <= critical <= 1.0
    assert strong < intermediate < critical


def test_calibration_is_monotonic_for_arrays():
    meta = {'probability_calibration': {'temperature': 3.5, 'bias': 3.8}}
    raw = np.array([0.001, 0.05, 0.30, 0.80, 0.99])

    calibrated = apply_probability_calibration(raw, meta)

    assert calibrated.shape == raw.shape
    assert np.all(np.diff(calibrated) > 0)
    assert np.all((0.0 <= calibrated) & (calibrated <= 1.0))


def test_meta_defaults_are_available():
    calibration = get_probability_calibration({})

    assert calibration['method'] == 'logit_temperature'
    assert calibration['temperature'] > 0
    assert 0 < calibration['clip_epsilon'] < 0.01


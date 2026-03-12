import numpy as np

DEFAULT_PROBABILITY_CALIBRATION = {
    'method': 'logit_temperature',
    'clip_epsilon': 1e-6,
    'temperature': 3.5,
    'bias': 3.8,
}


def get_probability_calibration(meta_obj=None):
    meta_obj = meta_obj or {}
    calibration = dict(DEFAULT_PROBABILITY_CALIBRATION)
    calibration.update(meta_obj.get('probability_calibration', {}))
    return calibration


def apply_probability_calibration(probability, meta_obj=None):
    calibration = get_probability_calibration(meta_obj)
    values = np.asarray(probability, dtype=float)

    if calibration.get('method') != 'logit_temperature':
        calibrated = np.clip(values, 0.0, 1.0)
    else:
        epsilon = max(float(calibration.get('clip_epsilon', 1e-6)), 1e-9)
        temperature = max(float(calibration.get('temperature', 3.5)), epsilon)
        bias = float(calibration.get('bias', 3.8))

        clipped = np.clip(values, epsilon, 1.0 - epsilon)
        logits = np.log(clipped / (1.0 - clipped))
        calibrated = 1.0 / (1.0 + np.exp(-((logits - bias) / temperature)))
        calibrated = np.clip(calibrated, 0.0, 1.0)

    if np.isscalar(probability):
        return float(calibrated)
    return calibrated


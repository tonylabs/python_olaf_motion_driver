"""ONNX policy runner + RunningStandardScaler reproduction.

skrl's ``RunningStandardScaler`` normalises obs as ``(x − μ) / √(σ² + ε)``.
At deploy time we freeze the final running stats (mean, var) into JSON next
to the ONNX and re-apply them here.  The exported ONNX contains only the
policy MLP forward pass — deterministic mean actions, no sampling.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Suppress GPU device discovery warning on systems without a GPU (e.g. Raspberry Pi)
ort.set_default_logger_severity(3)


class DeployedPolicy:
    def __init__(self, policy_dir: Path | str, providers: tuple[str, ...] = ("CPUExecutionProvider",)):
        policy_dir = Path(policy_dir)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(policy_dir / "policy.onnx"),
            sess_options=opts,
            providers=list(providers),
        )
        self._input_name = self._session.get_inputs()[0].name
        with open(policy_dir / "preprocessor.json") as f:
            pp = json.load(f)
        self._mean = np.asarray(pp["mean"], dtype=np.float32)
        self._var  = np.asarray(pp["var"], dtype=np.float32)
        self._eps  = float(pp.get("epsilon", 1.0e-8))
        self._clip = float(pp.get("clip", 5.0))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (obs - self._mean) / np.sqrt(self._var + self._eps)
        x = np.clip(x, -self._clip, self._clip).astype(np.float32)
        if x.ndim == 1:
            x = x[None, :]
        out = self._session.run(None, {self._input_name: x})[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)

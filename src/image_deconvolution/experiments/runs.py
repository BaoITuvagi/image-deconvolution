"""Run numerical processing independently of Qt and retain input identities."""

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from image_deconvolution.core.algorithms import blur, inverse, otf, rl, tikhonov
from image_deconvolution.experiments.inspection import quality, trace_rl
from image_deconvolution.io.images import ImageData, prepare_psf


@dataclass(frozen=True)
class RunConfig:
    iterations: int = 20
    epsilon: float = 1e-6
    method: str = "RL"
    threshold: float = 0.01
    regularization: float = 0.1
    data_range: float = 1.0
    inspect: bool = False

    def validate(self):
        if type(self.iterations) is not int or not 1 <= self.iterations <= 1000:
            raise ValueError("RL iterations must be an integer from 1 to 1000.")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("Epsilon must be finite and positive.")
        if self.method not in {"RL", "IF", "TRIF"}:
            raise ValueError("Choose RL, thresholded IF, or zeroth-order TRIF.")
        for name, value in [("IF threshold", self.threshold), ("TRIF lambda", self.regularization), ("Intensity range L", self.data_range)]:
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if type(self.inspect) is not bool:
            raise ValueError("Capture RL steps must be a boolean.")


@dataclass
class RunRecord:
    observation: ImageData
    psf: ImageData
    config: RunConfig
    output: np.ndarray
    seconds: float
    ground_truth: ImageData | None = None
    quality: dict | None = None
    metric_note: str = ""
    steps: dict = field(default_factory=dict)
    curve: list = field(default_factory=list)
    degradation: dict | None = None


def restore(observation, psf, config, ground_truth=None, degradation=None):
    config.validate()
    if config.method == "RL" and np.any(observation.pixels < 0):
        raise ValueError("Richardson-Lucy requires nonnegative observed intensities.")
    h = prepare_psf(psf.pixels, observation.pixels.shape)
    start = perf_counter()
    H = otf(h)
    otf_seconds = perf_counter() - start
    steps, curve = {}, []
    start = perf_counter()
    if config.method == "RL" and config.inspect:
        output, steps, curve, seconds = trace_rl(observation.pixels, H, config.iterations,
                                                config.epsilon, ground_truth, config.data_range)
        seconds += otf_seconds
    else:
        if config.method == "RL":
            output = rl(observation.pixels, H, config.iterations, config.epsilon)
        elif config.method == "IF":
            output = inverse(observation.pixels, H, config.threshold)
        else:
            output = tikhonov(observation.pixels, H, config.regularization)
        seconds = perf_counter() - start + otf_seconds
    if not np.isfinite(output).all():
        raise ValueError("Restoration produced nonfinite values. Check inputs and parameters.")
    output.setflags(write=False)
    values, note = quality(ground_truth, output, config.data_range)
    return RunRecord(observation, ImageData(h, psf.source, psf.policy, psf.file_sha256),
                     config, output, seconds, ground_truth, values, note, steps, curve, degradation)


def degrade(source, psf, photons, seed, data_range):
    """Generate Poisson observations explicitly; no restoration call regenerates noise."""
    if not np.isfinite(photons) or photons <= 0:
        raise ValueError("Photon level must be finite and positive.")
    if type(seed) is not int or not 0 <= seed <= 2**32-1:
        raise ValueError("Seed must be an integer between 0 and 2^32-1.")
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("Intensity range L must be finite and positive.")
    if source.pixels.min() < 0 or source.pixels.max() > data_range:
        raise ValueError("Source intensities must lie inside the declared [0, L] range.")
    h = prepare_psf(psf.pixels, source.pixels.shape)
    prediction = blur(source.pixels, otf(h))
    # Circular FFT may introduce tiny negative roundoff at theoretical zeros.
    rate = np.maximum(prediction / data_range, 0) * photons
    y = np.random.default_rng(seed).poisson(rate) / photons * data_range
    configuration = {"model": "Poisson", "photons": photons, "seed": seed,
                     "data_range": data_range, "boundaries": "circular",
                     "source": source.identity(), "psf": ImageData(h, psf.source).identity()}
    return ImageData(y, f"Simulated from {source.source}; seed={seed}"), configuration

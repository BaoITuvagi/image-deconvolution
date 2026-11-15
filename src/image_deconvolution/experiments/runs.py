"""Run numerical processing independently of Qt and retain input identities."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from image_deconvolution.core.algorithms import otf, rl
from image_deconvolution.io.images import ImageData, prepare_psf


@dataclass(frozen=True)
class RunConfig:
    iterations: int = 20
    epsilon: float = 1e-6

    def validate(self):
        if type(self.iterations) is not int or not 1 <= self.iterations <= 1000:
            raise ValueError("RL iterations must be an integer from 1 to 1000.")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("Epsilon must be finite and positive.")


@dataclass
class RunRecord:
    observation: ImageData
    psf: ImageData
    config: RunConfig
    output: np.ndarray
    seconds: float


def restore(observation, psf, config):
    config.validate()
    if np.any(observation.pixels < 0):
        raise ValueError("Richardson-Lucy requires nonnegative observed intensities.")
    h = prepare_psf(psf.pixels, observation.pixels.shape)
    start = perf_counter()
    output = rl(observation.pixels, otf(h), config.iterations, config.epsilon)
    seconds = perf_counter() - start
    if not np.isfinite(output).all():
        raise ValueError("Restoration produced nonfinite values. Check inputs and parameters.")
    output.setflags(write=False)
    return RunRecord(observation, ImageData(h, psf.source, psf.policy, psf.file_sha256),
                     config, output, seconds)

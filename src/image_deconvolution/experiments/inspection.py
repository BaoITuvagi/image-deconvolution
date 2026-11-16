"""Bounded RL observation preserving the original multiplicative update."""

from time import perf_counter

import numpy as np

from image_deconvolution.core.algorithms import blur, metrics


def quality(truth, estimate, data_range):
    if truth is None:
        return None, "No ground truth: restoration quality is unavailable."
    if truth.pixels.shape != estimate.shape:
        return None, "Ground truth dimensions do not match the observation."
    if min(estimate.shape) <= 10:
        return None, "SSIM requires both image dimensions to exceed 10."
    if truth.pixels.min() < 0 or truth.pixels.max() > data_range:
        return None, "Ground truth lies outside the declared intensity range [0, L]."
    values = metrics(truth.pixels / data_range, estimate / data_range)
    values["mse"] *= data_range**2
    return values, "Full-image metrics; PSNR range L is fixed by the run."


def trace_rl(
    y, H, iterations, epsilon, truth=None, data_range=1, memory_limit=64 * 1024 * 1024
):
    """Keep at most eight five-image checkpoints within a 64 MiB budget.

    Algorithm time excludes metric calculation and checkpoint copies. Curves use
    up to 32 selected iteration counts, not an optimizer of benchmark parameters.
    """
    capacity = min(8, memory_limit // (5 * y.nbytes))
    if capacity < 1:
        raise ValueError(
            "RL intermediate capture exceeds 64 MiB. Disable Capture RL steps for this image."
        )
    checkpoints = set(np.linspace(1, iterations, min(capacity, iterations), dtype=int))
    measurements = set(np.linspace(1, iterations, min(32, iterations), dtype=int))
    x = y.copy()
    snapshots, curve = {}, []
    seconds = 0.0
    for iteration in range(1, iterations + 1):
        start = perf_counter()
        prediction = blur(x, H)
        ratio = y / np.maximum(prediction, epsilon)
        correction = blur(ratio, np.conj(H))
        updated = x * correction
        seconds += perf_counter() - start
        if iteration in checkpoints:
            snapshots[int(iteration)] = {
                "Estimate x(k)": x.copy(),
                "Prediction Hx(k)": prediction.copy(),
                "Ratio y / max(Hx, epsilon)": ratio.copy(),
                "Correction H* ratio": correction.copy(),
                "Updated x(k+1)": updated.copy(),
            }
            for a in snapshots[int(iteration)].values():
                a.setflags(write=False)
        x = updated
        if iteration in measurements:
            values, _ = quality(truth, x, data_range)
            if values is not None:
                curve.append({"iteration": int(iteration), **values})
    return x, snapshots, curve, seconds

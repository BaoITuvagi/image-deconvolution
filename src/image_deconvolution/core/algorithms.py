"""Circular 2D deconvolution adapted from the group 07 research code.

See docs/attribution.md for provenance. Use float64 arrays and full-image,
centered, normalized PSFs. Metrics assume normalized intensities with L = 1.
This numerical API assumes valid inputs; UI input validation is a later stage.
"""

import numpy as np
from scipy.fft import irfft2, rfft2
from scipy.ndimage import gaussian_filter


def psf_gaussian(shape, sigma):
    """Return a normalized Gaussian PSF centered at shape // 2."""
    y, x = np.indices(shape)
    y = y - shape[0] // 2
    x = x - shape[1] // 2
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h /= h.sum()
    return h


def otf(h):
    """Transform a full-image centered PSF into its real-FFT transfer function."""
    return rfft2(np.fft.ifftshift(h))


def blur(x, H):
    """Apply circular convolution without clipping or intensity rescaling."""
    return irfft2(rfft2(x) * H, s=x.shape)


def inverse(y, H, tau=0.01):
    """Apply inverse filtering only where the OTF magnitude is at least tau."""
    Y = rfft2(y)
    X = np.zeros_like(Y)
    valid = np.abs(H) >= tau
    X[valid] = Y[valid] / H[valid]
    return irfft2(X, s=y.shape)


def tikhonov(y, H, lam=0.001):
    """Apply zeroth-order Tikhonov with L = I, not a Laplacian regularizer."""
    return irfft2(np.conj(H) * rfft2(y) / (np.abs(H) ** 2 + lam), s=y.shape)


def rl(y, H, iterations=20, epsilon=1e-6, checkpoints=None, cache=True):
    """Run RL initialized from y, optionally returning copies at checkpoints.

    cache=False retains the research's deliberately redundant OTF-transform
    baseline. It does not represent an optimized alternative implementation.
    """
    x = y.copy()
    hist = {}
    checkpoints = set(checkpoints or [])
    for k in range(1, iterations + 1):
        if not cache:
            H = rfft2(irfft2(H, s=y.shape))
        prediction = blur(x, H)
        ratio = y / np.maximum(prediction, epsilon)
        x *= blur(ratio, np.conj(H))
        if k in checkpoints:
            hist[k] = x.copy()
    return (x, hist) if checkpoints else x


def metrics(ref, x):
    """Measure full arrays with PSNR L = 1 and research SSIM conventions.

    SSIM uses an 11 x 11 Gaussian window, sigma 1.5, population covariance,
    and the valid interior. Arrays must match and both dimensions exceed 10.
    """
    mse = float(np.mean((ref - x) ** 2))
    psnr = float(10 * np.log10(1 / mse)) if mse else float("inf")

    def g(a):
        return gaussian_filter(a, 1.5, truncate=3.5, mode="reflect")

    a, b = g(ref), g(x)
    v1 = g(ref * ref) - a * a
    v2 = g(x * x) - b * b
    cov = g(ref * x) - a * b
    m = ((2 * a * b + 0.01**2) * (2 * cov + 0.03**2)) / (
        (a * a + b * b + 0.01**2) * (v1 + v2 + 0.03**2)
    )
    return {"mse": mse, "psnr": psnr, "ssim": float(m[5:-5, 5:-5].mean())}

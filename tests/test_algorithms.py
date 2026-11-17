"""Numerical regression checks adapted from the group 07 baseline tests.

Run: python -B -m unittest discover -s tests -v
Provenance and preserved conventions: docs/attribution.md.
"""

import subprocess
import sys
import unittest

import numpy as np
from scipy.fft import irfft2, rfft2
from scipy.signal import convolve2d

from image_deconvolution.core.algorithms import (
    blur,
    inverse,
    metrics,
    otf,
    psf_gaussian,
    rl,
    tikhonov,
)


class NumericalTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.x = self.rng.random((32, 32))
        self.kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
        self.h = np.zeros_like(self.x)
        self.h[15:18, 15:18] = self.kernel
        self.H = otf(self.h)
        self.identity = np.ones((32, 17))

    def test_circular_convolution_matches_spatial_reference(self):
        expected = convolve2d(self.x, self.kernel, mode="same", boundary="wrap")
        np.testing.assert_allclose(blur(self.x, self.H), expected, rtol=0, atol=1e-12)

    def test_convolution_adjoint(self):
        u, v = self.rng.random((2, 32, 32))
        self.assertLess(
            abs(np.sum(blur(u, self.H) * v) - np.sum(u * blur(v, self.H.conj()))),
            1e-10,
        )

    def test_identical_image_metrics(self):
        result = metrics(self.x, self.x)
        self.assertEqual(result, {"mse": 0, "psnr": float("inf"), "ssim": 1})

    def test_psnr_uses_fixed_unit_range(self):
        result = metrics(np.zeros_like(self.x), np.full_like(self.x, 0.1))
        self.assertAlmostEqual(result["mse"], 0.01)
        self.assertAlmostEqual(result["psnr"], 20)

    def test_rl_identity(self):
        np.testing.assert_allclose(
            rl(self.x, self.identity, 1), self.x, rtol=0, atol=1e-12
        )

    def test_tikhonov_identity_regularizer(self):
        np.testing.assert_allclose(
            tikhonov(self.x, self.identity, 0.1), self.x / 1.1, rtol=0, atol=1e-12
        )

    def test_inverse_threshold_includes_boundary_and_zeros_rejected_bins(self):
        H = np.full_like(self.identity, 0.25)
        np.testing.assert_allclose(
            inverse(self.x, H, 0.25), self.x * 4, rtol=0, atol=1e-12
        )
        np.testing.assert_array_equal(inverse(self.x, H, 0.26), np.zeros_like(self.x))
        np.testing.assert_array_equal(inverse(self.x, H * 0), np.zeros_like(self.x))

    def test_odd_rectangular_psf_centering_and_output_shape(self):
        x = self.rng.random((31, 35))
        h = np.zeros_like(x)
        h[15, 17] = 1
        np.testing.assert_allclose(blur(x, otf(h)), x, rtol=0, atol=1e-12)
        gaussian = psf_gaussian(x.shape, 2)
        self.assertAlmostEqual(float(gaussian.sum()), 1)
        self.assertEqual(np.unravel_index(gaussian.argmax(), gaussian.shape), (15, 17))

    def test_rl_checkpoints_preserve_input_and_estimates(self):
        original = self.x.copy()
        result, history = rl(self.x, self.H, 5, checkpoints=[1, 3, 5])
        np.testing.assert_array_equal(self.x, original)
        for iteration, estimate in history.items():
            np.testing.assert_allclose(
                estimate, rl(self.x, self.H, iteration), rtol=0, atol=1e-12
            )
            self.assertFalse(np.shares_memory(estimate, result))
        np.testing.assert_array_equal(result, rl(self.x, self.H, 5))

    def test_rl_matches_previous_loop_including_uncached_spectrum_and_checkpoints(self):
        for shape in [(32, 32), (31, 35)]:
            y = self.rng.random(shape)
            h = np.zeros(shape)
            row, col = (n // 2 for n in shape)
            h[row, col], h[row - 1, col + 1] = 0.7, 0.3
            H = otf(h)  # Asymmetric PSF exercises a complex, nontrivial adjoint.
            original = H.copy()
            for cache in [True, False]:
                with self.subTest(shape=shape, cache=cache):
                    expected, spectrum, checkpoints = y.copy(), H.copy(), {}
                    for k in range(1, 6):
                        if not cache:
                            spectrum = rfft2(irfft2(spectrum, s=shape))
                        ratio = y / np.maximum(blur(expected, spectrum), 1e-6)
                        expected *= blur(ratio, np.conj(spectrum))
                        if k in [1, 3, 5]:
                            checkpoints[k] = expected.copy()
                    result, history = rl(y, H, 5, checkpoints=[1, 3, 5], cache=cache)
                    np.testing.assert_array_equal(result, expected)
                    self.assertEqual(set(history), set(checkpoints))
                    for k, estimate in checkpoints.items():
                        np.testing.assert_array_equal(history[k], estimate)
                        self.assertFalse(np.shares_memory(history[k], result))
                    np.testing.assert_array_equal(H, original)

    def test_core_import_does_not_load_qt(self):
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; import image_deconvolution.core.algorithms; "
                    "assert not any(n.startswith(('PySide6', 'shiboken6')) for n in sys.modules)"
                ),
            ],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()

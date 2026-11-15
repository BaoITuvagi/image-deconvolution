"""Scientific invariants for controlled comparisons and bounded RL inspection."""

from dataclasses import replace
import unittest

import numpy as np

from image_deconvolution.core.algorithms import blur, inverse, otf, rl, tikhonov
from image_deconvolution.experiments.inspection import quality, trace_rl
from image_deconvolution.experiments.runs import RunConfig, degrade, restore
from image_deconvolution.io.images import ImageData, prepare_psf, read_image, sample_path


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.observation = read_image(sample_path("cells_s2_p200_r0_input.tif"))
        self.psf = read_image(sample_path("psf_s2.tif"))
        self.truth = read_image(sample_path("cells_gt.tif"))

    def test_comparison_uses_identical_input_and_expected_algorithms(self):
        original = self.observation.pixels.copy()
        H = otf(prepare_psf(self.psf.pixels, original.shape))
        functions = {"IF": lambda: inverse(original, H, .01),
                     "TRIF": lambda: tikhonov(original, H, .1),
                     "RL": lambda: rl(original, H, 5)}
        for method, expected in functions.items():
            record = restore(self.observation, self.psf, RunConfig(5, method=method), self.truth)
            np.testing.assert_array_equal(record.output, expected())
            self.assertIsNotNone(record.quality)
        np.testing.assert_array_equal(self.observation.pixels, original)

    def test_degradation_is_explicit_and_seeded(self):
        a, settings = degrade(self.truth, self.psf, 200, 7, 1)
        b, _ = degrade(self.truth, self.psf, 200, 7, 1)
        c, _ = degrade(self.truth, self.psf, 200, 8, 1)
        np.testing.assert_array_equal(a.pixels, b.pixels)
        self.assertFalse(np.array_equal(a.pixels, c.pixels))
        self.assertEqual(settings["seed"], 7)
        for iterations in [2, 5]:
            restore(a, self.psf, RunConfig(iterations))
        np.testing.assert_array_equal(a.pixels, b.pixels)

    def test_metric_availability_and_intensity_convention(self):
        for truth in [None, ImageData(np.zeros((11, 12)), "wrong shape"),
                      ImageData(self.truth.pixels*255, "wrong range")]:
            values, note = quality(truth, self.observation.pixels, 1)
            self.assertIsNone(values)
            self.assertTrue(note)
        unit, _ = quality(self.truth, self.observation.pixels, 1)
        scaled, _ = quality(ImageData(self.truth.pixels*255, "scaled"), self.observation.pixels*255, 255)
        self.assertAlmostEqual(unit["psnr"], scaled["psnr"])
        self.assertAlmostEqual(unit["ssim"], scaled["ssim"])
        self.assertAlmostEqual(unit["mse"]*255**2, scaled["mse"])

    def test_trace_matches_baseline_and_intermediate_equations(self):
        y = self.observation.pixels
        H = otf(prepare_psf(self.psf.pixels, y.shape))
        result, steps, curve, seconds = trace_rl(y, H, 20, 1e-6, self.truth)
        np.testing.assert_array_equal(result, rl(y, H, 20))
        self.assertLessEqual(sum(a.nbytes for s in steps.values() for a in s.values()), 64*1024*1024)
        self.assertLessEqual(len(steps), 8)
        self.assertGreater(seconds, 0)
        for iteration, step in steps.items():
            arrays = list(step.values())
            np.testing.assert_array_equal(arrays[0], y if iteration == 1 else rl(y, H, iteration-1))
            np.testing.assert_array_equal(arrays[1], blur(arrays[0], H))
            np.testing.assert_array_equal(arrays[2], y / np.maximum(arrays[1], 1e-6))
            np.testing.assert_array_equal(arrays[3], blur(arrays[2], H.conj()))
            np.testing.assert_array_equal(arrays[4], arrays[0]*arrays[3])
        self.assertEqual(curve[-1]["iteration"], 20)

    def test_bars_late_iterations_can_lose_quality(self):
        y = read_image(sample_path("bars_s2_p200_r0_input.tif"))
        truth = read_image(sample_path("bars_gt.tif"))
        record = restore(y, self.psf, RunConfig(100, inspect=True), truth)
        self.assertLess(record.curve[-1]["psnr"], max(row["psnr"] for row in record.curve))

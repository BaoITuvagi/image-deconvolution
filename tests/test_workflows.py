"""Maintained image-I/O, run-snapshot, and desktop workflow regression tests."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from image_deconvolution.core.algorithms import otf, rl
from image_deconvolution.experiments.runs import RunConfig, restore
from image_deconvolution.io.images import ImageData, prepare_psf, read_image, sample_path, save_tiff
from image_deconvolution.ui.main_window import MainWindow


class ImageIOTests(unittest.TestCase):
    def test_explicit_integer_policy_and_float_export(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "integer.png"
            a = np.arange(256, dtype=np.uint8).reshape(16, 16)
            Image.fromarray(a).save(path)
            np.testing.assert_array_equal(read_image(path).pixels, a)
            np.testing.assert_array_equal(read_image(path, "normalize_integer").pixels, a / 255)
            output = Path(directory) / "output.tif"
            values = np.linspace(-0.25, 1.5, 256).reshape(16, 16)
            save_tiff(output, values)
            np.testing.assert_array_equal(read_image(output).pixels, values.astype(np.float32))
            with self.assertRaises(FileExistsError):
                save_tiff(output, values)

    def test_reject_color_stack_nonfinite_and_invalid_psf(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            color = Path(directory) / "color.png"
            Image.new("RGB", (16, 16)).save(color)
            with self.assertRaisesRegex(ValueError, "grayscale"):
                read_image(color)
            stack = Path(directory) / "stack.tif"
            Image.new("L", (16, 16)).save(stack, save_all=True, append_images=[Image.new("L", (16, 16))])
            with self.assertRaisesRegex(ValueError, "Stacks"):
                read_image(stack)
        for a in [np.full((16, 16), np.nan), np.zeros((2, 2, 2))]:
            with self.assertRaises(ValueError):
                ImageData(a, "invalid")
        for h in [np.zeros((3, 3)), -np.ones((3, 3)), np.ones((33, 33))]:
            with self.assertRaises(ValueError):
                prepare_psf(h, (32, 32))

    def test_center_padding_and_rl_snapshot(self):
        y = read_image(sample_path("cells_s2_p200_r0_input.tif"))
        psf = read_image(sample_path("psf_s2.tif"))
        record = restore(y, psf, RunConfig(5))
        expected = rl(y.pixels, otf(prepare_psf(psf.pixels, y.pixels.shape)), 5)
        np.testing.assert_array_equal(record.output, expected)
        self.assertFalse(record.observation.pixels.flags.writeable)
        self.assertFalse(record.output.flags.writeable)
        padded = prepare_psf(np.eye(3), (16, 18))
        self.assertAlmostEqual(padded.sum(), 1)
        self.assertEqual(padded[8, 9], 1 / 3)
        for config in [RunConfig(0), RunConfig(2, 0), RunConfig(2, float("nan"))]:
            with self.assertRaises(ValueError):
                restore(y, psf, config)


class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        warning_patch = patch("PySide6.QtWidgets.QMessageBox.warning")
        self.warning = warning_patch.start()
        self.addCleanup(warning_patch.stop)
        self.window = MainWindow()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        if self.window.worker:
            self.wait_for_worker()
        self.window.close()
        self.app.processEvents()

    def wait_for_worker(self):
        deadline = time.monotonic() + 20
        while self.window.worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(10)
        self.assertIsNone(self.window.worker, "Worker failed to finish")

    def test_load_run_stale_save_and_errors(self):
        with patch("PySide6.QtWidgets.QMessageBox.warning") as warning:
            QTest.mouseClick(self.window.run_button, Qt.MouseButton.LeftButton)
            self.assertTrue(warning.called)
        QTest.mouseClick(self.window.sample_button, Qt.MouseButton.LeftButton)
        self.assertEqual(self.window.observation.pixels.shape, (256, 256))
        QTest.mouseClick(self.window.run_button, Qt.MouseButton.LeftButton)
        self.assertFalse(self.window.controls.isEnabled())
        self.wait_for_worker()
        self.assertTrue(self.window.controls.isEnabled())
        self.assertIsNotNone(self.window.record)
        snapshot = self.window.record.output.copy()
        self.window.iterations.setValue(10)
        self.assertIn("earlier configuration", self.window.result_info.text())
        np.testing.assert_array_equal(self.window.record.output, snapshot)
        self.window.low.setValue(-1)
        self.window.high.setValue(2)
        self.window.refresh_views()
        np.testing.assert_array_equal(self.window.record.output, snapshot)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = str(Path(directory) / "restoration.tif")
            with patch("PySide6.QtWidgets.QFileDialog.getSaveFileName", return_value=(path, "")):
                self.window.save_output()
            self.assertFalse(self.warning.called, str(self.warning.call_args))
            np.testing.assert_array_equal(read_image(path).pixels, snapshot.astype(np.float32))

    def test_worker_failure_and_close_guard(self):
        def fail():
            time.sleep(0.05)
            raise ValueError("Expected worker failure")
        with patch("PySide6.QtWidgets.QMessageBox.warning") as warning:
            self.window.start_operation(fail, self.window.accept_record)
            self.assertFalse(self.window.close())
            self.wait_for_worker()
            self.assertTrue(warning.called)
        self.assertTrue(self.window.controls.isEnabled())

    def test_comparison_history_inspector_and_fixed_observation(self):
        self.window.load_sample()
        original = self.window.observation.pixels.copy()
        self.window.compare_methods()
        self.wait_for_worker()
        self.assertEqual([r.config.method for r in self.window.history], ["IF", "TRIF", "RL"])
        self.assertEqual(self.window.table.rowCount(), 3)
        self.assertIsNotNone(self.window.record.quality)
        self.assertGreater(self.window.inspector.iteration.count(), 0)
        self.window.other_run.setCurrentIndex(1)
        self.assertIsNotNone(self.window.other_view.array)
        self.window.method.setCurrentText("TRIF")
        self.window.sigma.setValue(3)
        np.testing.assert_array_equal(self.window.observation.pixels, original)
        self.window.select_history(0)
        self.assertEqual(self.window.record.config.method, "IF")
        self.window.clear_history()
        self.assertEqual(len(self.window.history), 0)

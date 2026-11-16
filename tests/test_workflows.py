"""Maintained image-I/O, run-snapshot, and desktop workflow regression tests."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gc
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtCore import QCoreApplication, QEvent, Qt, QThread
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
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        del self.window
        gc.collect()

    def wait_for_worker(self):
        deadline = time.monotonic() + 20
        while self.window.worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(10)
        self.assertIsNone(self.window.worker, "Worker failed to finish")

    def test_load_run_stale_save_and_errors(self):
        self.window.control_pages.setCurrentIndex(1)
        with patch("PySide6.QtWidgets.QMessageBox.warning") as warning:
            QTest.mouseClick(self.window.run_button, Qt.MouseButton.LeftButton)
            self.assertTrue(warning.called)
        self.window.control_pages.setCurrentIndex(0)
        QTest.mouseClick(self.window.sample_button, Qt.MouseButton.LeftButton)
        self.assertEqual(self.window.observation.pixels.shape, (256, 256))
        QTest.mouseClick(self.window.run_button, Qt.MouseButton.LeftButton)
        self.assertFalse(self.window.controls.isEnabled())
        self.wait_for_worker()
        self.assertTrue(self.window.controls.isEnabled())
        received_on = []
        self.window.start_operation(lambda: "finished", lambda _: received_on.append(QThread.currentThread()))
        self.wait_for_worker()
        self.assertEqual(received_on, [self.app.thread()])
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
        self.window.input_view._auto_fit = False
        self.window.input_view.scale(2, 2)
        self.window.input_view._notify()
        for view in self.window.views:
            self.assertEqual(view.transform().m11(), self.window.input_view.transform().m11())
            self.assertFalse(view._auto_fit)
        self.window.fit_views()
        self.assertTrue(all(v._auto_fit for v in self.window.views))
        self.window.method.setCurrentText("TRIF")
        self.window.sigma.setValue(3)
        np.testing.assert_array_equal(self.window.observation.pixels, original)
        self.window.select_history(0)
        self.assertEqual(self.window.record.config.method, "IF")
        self.window.clear_history()
        self.assertEqual(len(self.window.history), 0)
        self.assertIsNone(self.window.inspector.record)
        self.window.method.setCurrentText("IF")
        self.window.load_bars()
        self.assertEqual(self.window.method.currentText(), "RL")

    def test_reference_and_export_reopen_workflow(self):
        self.window.verify_references()
        self.wait_for_worker()
        self.assertEqual(len(self.window.reference_panel.rows), 6)
        self.window.load_research("fibers")
        self.window.run_restoration()
        self.wait_for_worker()
        output = self.window.record.output.copy()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            folder = str(Path(directory) / "saved experiment")
            with patch("PySide6.QtWidgets.QFileDialog.getSaveFileName", return_value=(folder, "")):
                self.window.export_experiment()
            self.wait_for_worker()
            self.assertIn("Experiment saved:", self.window.statusBar().currentMessage())
            with patch("PySide6.QtWidgets.QFileDialog.getOpenFileName", return_value=(str(Path(folder) / "experiment.json"), "")):
                self.window.open_experiment()
            self.window.run_restoration()
            self.wait_for_worker()
            np.testing.assert_allclose(self.window.record.output, output, rtol=0, atol=1e-12)
        self.assertFalse(self.warning.called, str(self.warning.call_args))

    def test_inspector_plot_renders_without_overriding_qt_metrics(self):
        self.window.load_sample()
        self.window.run_restoration()
        self.wait_for_worker()
        self.window.tabs.setCurrentIndex(1)
        self.app.processEvents()
        plot = self.window.inspector.chart_view
        self.assertTrue(callable(plot.metric))
        self.assertFalse(plot.grab().isNull())
        view = self.window.inspector.view
        mapped = view.mapFromScene(view.sceneRect()).boundingRect()
        self.assertLessEqual(mapped.width(), view.viewport().width()+2)
        self.assertLessEqual(mapped.height(), view.viewport().height()+2)
        self.window.inspector.metric_choice.setCurrentText("SSIM")
        self.app.processEvents()
        self.assertFalse(plot.grab().isNull())

    def test_keyboard_selection_export_and_current_inputs(self):
        self.window.load_sample()
        self.window.compare_methods()
        self.wait_for_worker()
        self.assertEqual(self.window.table.currentRow(), 2)
        self.window.table.setFocus()
        QTest.keyClick(self.window.table, Qt.Key.Key_Up)
        self.assertIs(self.window.record, self.window.history[1])
        self.assertEqual(self.window.record.config.method, "TRIF")
        np.testing.assert_array_equal(self.window.output_view.array, self.window.history[1].output)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = str(Path(directory) / "keyboard-selected.tif")
            with patch("PySide6.QtWidgets.QFileDialog.getSaveFileName", return_value=(path, "")):
                self.window.save_output()
            np.testing.assert_array_equal(read_image(path).pixels, self.window.history[1].output.astype(np.float32))
        self.window.load_bars()
        self.assertIsNone(self.window.record)
        self.assertEqual(len(self.window.history), 3)
        np.testing.assert_array_equal(self.window.input_view.array, self.window.observation.pixels)
        np.testing.assert_array_equal(self.window.truth_view.array, self.window.ground_truth.pixels)
        self.assertIsNone(self.window.output_view.array)
        self.assertFalse(self.window.save_button.isEnabled())
        self.assertFalse(self.window.export_button.isEnabled())
        self.window.select_history(0)
        self.window.select_history(1)
        self.assertIn("earlier configuration", self.window.result_info.text())
        self.window.current_button.click()
        self.assertIsNone(self.window.record)
        np.testing.assert_array_equal(self.window.input_view.array, self.window.observation.pixels)
        self.assertEqual(len(self.window.history), 3)

    def test_truth_edits_refresh_without_discarding_history(self):
        self.window.set_observation(read_image(sample_path("cells_s2_p200_r0_input.tif")))
        truth_path = str(sample_path("cells_gt.tif"))
        with patch("PySide6.QtWidgets.QFileDialog.getOpenFileName", return_value=(truth_path, "")):
            self.window.open_truth()
        np.testing.assert_array_equal(self.window.truth_view.array, self.window.ground_truth.pixels)
        self.window.run_restoration()
        self.wait_for_worker()
        self.window.clear_truth()
        self.assertIsNone(self.window.record)
        self.assertIsNone(self.window.truth_view.array)
        self.assertIsNotNone(self.window.history[0].ground_truth)
        with patch("PySide6.QtWidgets.QFileDialog.getOpenFileName", return_value=(truth_path, "")):
            self.window.open_source()
        np.testing.assert_array_equal(self.window.truth_view.array, self.window.source.pixels)

    def test_inspector_quality_fit_and_reference_keyboard_selection(self):
        self.window.load_sample()
        self.window.run_restoration()
        self.wait_for_worker()
        panel = self.window.inspector
        for index in range(panel.phase.count()):
            panel.phase.setCurrentIndex(index)
            self.assertIn("Quality of updated x(k+1), iteration 1", panel.range_info.text())
        self.window.tabs.setCurrentIndex(1)
        self.app.processEvents()
        panel.view._auto_fit = False
        panel.view.scale(3, 3)
        zoomed = panel.view.transform().m11()
        self.window.fit_views()
        self.assertTrue(panel.view._auto_fit)
        self.assertLess(panel.view.transform().m11(), zoomed)
        self.window.verify_references()
        self.wait_for_worker()
        self.window.tabs.setCurrentIndex(2)
        table = self.window.reference_panel.table
        table.setFocus()
        QTest.keyClick(table, Qt.Key.Key_Down)
        self.assertEqual(table.currentRow(), 1)
        self.assertIn("cells / TRIF", self.window.reference_panel.detail.text())
        np.testing.assert_array_equal(self.window.reference_panel.python_view.array,
                                      self.window.reference_panel.rows[1]["record"].output)

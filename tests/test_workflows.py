"""Maintained image-I/O, run-snapshot, and desktop workflow regression tests."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gc
import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt, QThread
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from image_deconvolution.core.algorithms import otf, rl
from image_deconvolution.experiments.export import export_bundle
from image_deconvolution.experiments.inspection import quality
from image_deconvolution.experiments.runs import RunConfig, restore
from image_deconvolution.io.images import (
    ImageData,
    prepare_psf,
    read_image,
    sample_path,
    save_tiff,
)
from image_deconvolution.ui.main_window import MainWindow


class ImageIOTests(unittest.TestCase):
    def test_explicit_integer_policy_and_float_export(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "integer.png"
            a = np.arange(256, dtype=np.uint8).reshape(16, 16)
            Image.fromarray(a).save(path)
            np.testing.assert_array_equal(read_image(path).pixels, a)
            np.testing.assert_array_equal(
                read_image(path, "normalize_integer").pixels, a / 255
            )
            output = Path(directory) / "output.tif"
            values = np.linspace(-0.25, 1.5, 256).reshape(16, 16)
            save_tiff(output, values)
            np.testing.assert_array_equal(
                read_image(output).pixels, values.astype(np.float32)
            )
            with self.assertRaises(FileExistsError):
                save_tiff(output, values)

    def test_reject_color_stack_nonfinite_and_invalid_psf(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            color = Path(directory) / "color.png"
            Image.new("RGB", (16, 16)).save(color)
            with self.assertRaisesRegex(ValueError, "grayscale"):
                read_image(color)
            stack = Path(directory) / "stack.tif"
            Image.new("L", (16, 16)).save(
                stack, save_all=True, append_images=[Image.new("L", (16, 16))]
            )
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
        if self.window.worker or self.window.baseline.pending:
            self.wait_for_worker()
        self.window.close()
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        del self.window
        gc.collect()

    def wait_for_worker(self):
        deadline = time.monotonic() + 20
        while (
            self.window.worker is not None or self.window.baseline.pending
        ) and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.002)  # Yield the GIL so the Python worker can make progress.
        self.assertIsNone(self.window.worker, "Worker failed to finish")
        self.assertFalse(
            self.window.baseline.pending, "Baseline measurement failed to finish"
        )

    def test_baseline_on_load_current_range_and_unavailable_inputs(self):
        w = self.window
        self.assertEqual(w.baseline.text(), "Degraded input: N/A")
        for name in ("cells", "bars", "fibers"):
            w.load_research(name)
            self.wait_for_worker()
            expected, _ = quality(w.ground_truth, w.observation.pixels, 1)
            self.assertIn(f"MSE {expected['mse']:.5g}", w.baseline.text())
            self.assertIn(f"PSNR {expected['psnr']:.3f} dB", w.baseline.text())
            self.assertIn(f"SSIM {expected['ssim']:.5f}", w.baseline.text())
            self.assertEqual(w.history, [])
            self.assertIsNone(w.record)
            self.assertFalse(w.results_visible)
            self.app.processEvents()
            self.assertGreater(
                w.input_view.height(), 0.7 * w.image_layout.geometry().height()
            )
        observation, truth = w.observation, w.ground_truth
        w.data_range.setValue(2)
        self.wait_for_worker()
        expected, _ = quality(truth, observation.pixels, 2)
        self.assertIn(f"PSNR {expected['psnr']:.3f} dB", w.baseline.text())
        self.assertIn("L=2", w.baseline.toolTip())
        self.assertIs(w.observation, observation)
        w.clear_truth()
        self.assertEqual(w.baseline.text(), "Degraded input: N/A")
        self.assertEqual(w.baseline.toolTip(), "")
        for ground_truth, data_range in [
            (ImageData(np.zeros((12, 13)), "mismatched"), 1),
            (truth, 0.1),
        ]:
            w.ground_truth = ground_truth
            w.data_range.setValue(data_range)
            w.refresh_views()
            self.wait_for_worker()
            self.assertEqual(w.baseline.text(), "Degraded input: N/A")
            self.assertEqual(w.baseline.toolTip(), "")
        w.set_observation(ImageData(np.ones((10, 10)), "small"))
        w.ground_truth = ImageData(np.ones((10, 10)), "small truth")
        w.data_range.setValue(1)
        w.refresh_views()
        self.wait_for_worker()
        self.assertEqual(w.baseline.text(), "Degraded input: N/A")
        with patch.object(
            w, "choose_image", return_value=str(sample_path("cells_gt.tif"))
        ):
            w.open_source()
        self.assertIsNone(w.observation)
        self.assertEqual(w.baseline.text(), "Degraded input: N/A")
        w.simulate()
        self.wait_for_worker()
        self.assertIn("MSE ", w.baseline.text())
        self.assertEqual(w.history, [])

    def test_baseline_coalesces_background_requests_and_rejects_stale_values(self):
        w = self.window
        gate = threading.Event()
        calls, threads = [], []

        def measure(truth, pixels, data_range):
            calls.append(data_range)
            threads.append(QThread.currentThread())
            if len(calls) == 1:
                gate.wait(5)
            return quality(truth, pixels, data_range)

        with patch(
            "image_deconvolution.ui.baseline_summary.quality", side_effect=measure
        ):
            w.load_sample()
            self.app.processEvents()
            self.assertIsNotNone(w.baseline.worker)
            self.assertFalse(w.close())
            w.data_range.setValue(2)
            w.load_bars()
            w.data_range.setValue(3)
            self.assertEqual(w.baseline.text(), "Degraded input: N/A")
            self.assertTrue(w.controls.isEnabled())
            gate.set()
            self.wait_for_worker()
        expected, _ = quality(w.ground_truth, w.observation.pixels, 3)
        self.assertEqual(calls, [1, 3])
        self.assertTrue(all(thread is not self.app.thread() for thread in threads))
        self.assertIn(f"PSNR {expected['psnr']:.3f} dB", w.baseline.text())
        self.assertEqual(w.history, [])
        # A pending current-input measurement cannot overwrite a historical baseline.
        w.run_restoration()
        self.wait_for_worker()
        historical = w.record
        gate.clear()
        calls.clear()
        with patch(
            "image_deconvolution.ui.baseline_summary.quality", side_effect=measure
        ):
            w.load_sample()
            self.app.processEvents()
            w.select_history(0)
            selected = w.baseline.text()
            gate.set()
            self.wait_for_worker()
            self.assertEqual(w.baseline.text(), selected)
            self.assertIn("L=3", w.baseline.toolTip())
            self.assertIs(w.record, historical)

    def test_baseline_failure_recovers_without_extra_status_prose(self):
        w = self.window
        with patch(
            "image_deconvolution.ui.baseline_summary.quality",
            side_effect=ValueError("Unavailable measurement"),
        ):
            w.load_sample()
            self.wait_for_worker()
        self.assertEqual(w.baseline.text(), "Degraded input: N/A")
        self.assertEqual(w.baseline.toolTip(), "")
        self.assertTrue(w.run_button.isEnabled())
        self.assertFalse(self.warning.called)
        w.data_range.setValue(2)
        self.wait_for_worker()
        self.assertIn("MSE ", w.baseline.text())

    def test_inline_result_titles_and_compact_equal_image_viewports(self):
        w = self.window
        w.load_sample()
        self.wait_for_worker()
        w.compare_methods()
        self.wait_for_worker()
        for points in (9, 12, 16):
            font = w.font()
            font.setPointSize(points)
            w.setFont(font)
            for width in (1100, 1700):
                w.resize(width, 1000)
                self.app.processEvents()
                for i, box in enumerate(w.image_boxes[2:]):
                    run = w.slot_records[i]
                    self.assertEqual(
                        box.full_title,
                        f"{w.run_ids[id(run)]}: {run.config.method}: {w.run_parameters(run.config)}",
                    )
                    self.assertEqual(box.toolTip(), "")
                    self.assertLessEqual(
                        box.fontMetrics().horizontalAdvance(box.title()),
                        box.width() - 20,
                    )
                    self.assertEqual(box.layout().count(), 1)
                    self.assertFalse(hasattr(box, "parameters"))
                heights = [view.viewport().height() for view in w.views[2:]]
                self.assertLessEqual(max(heights) - min(heights), 1)
                self.assertEqual(
                    w.baseline.parentWidget(), w.current_button.parentWidget()
                )
                self.assertTrue(w.table.isVisible())
        w.setFont(self.app.font())
        w.resize(1700, 1000)
        for _ in range(5):
            self.app.processEvents()
        text_width = w.baseline.fontMetrics().horizontalAdvance(w.baseline.text())
        w.resize(max(w.width(), w.width() - w.baseline.width() + text_width + 32), 1000)
        for _ in range(5):
            self.app.processEvents()
        self.assertGreaterEqual(w.baseline.width(), text_width)
        self.assertLessEqual(
            w.baseline.heightForWidth(w.baseline.width()),
            2 * w.baseline.fontMetrics().height(),
        )
        self.assertLessEqual(
            w.baseline.height(),
            max(w.current_button.height(), 2 * w.baseline.fontMetrics().height()),
        )
        self.assertGreater(min(view.viewport().height() for view in w.views[2:]), 250)
        w.fit_views()
        self.assertTrue(all(view._auto_fit for view in w.views))

    def test_single_runs_replace_selected_slot_without_auto_filling(self):
        w = self.window
        w.load_sample()
        self.assertTrue(all(not box.isVisible() for box in w.image_boxes[2:]))
        self.assertIsNotNone(w.input_view.array)
        self.assertIsNotNone(w.truth_view.array)
        for count in range(1, 5):
            w.run_restoration()
            self.wait_for_worker()
            self.assertEqual(len(w.history), count)
            self.assertEqual(w.active_slot, 0)
            self.assertIs(w.slot_records[0], w.history[-1])
            self.assertEqual(w.slot_records[1:], [None, None])
            self.assertTrue(all(box.isVisible() for box in w.image_boxes[2:]))
            self.assertEqual(
                [box.property("active") for box in w.image_boxes[2:]],
                [True, False, False],
            )
            np.testing.assert_array_equal(w.record.output, w.history[0].output)

        left = w.slot_records[0]
        QTest.mouseClick(w.other_view.viewport(), Qt.MouseButton.LeftButton)
        self.assertEqual(w.active_slot, 1)
        self.assertIsNone(w.record)
        self.assertIsNone(w.inspector.record)
        self.assertFalse(w.save_button.isEnabled())
        self.assertFalse(w.export_button.isEnabled())
        self.assertEqual(w.table.currentRow(), -1)
        self.assertIsNotNone(w.input_view.array)
        w.iterations.setValue(7)
        w.run_restoration()
        # The run belongs to the slot selected at dispatch, even during inspection.
        w.select_slot(2)
        self.wait_for_worker()
        self.assertEqual(w.active_slot, 1)
        self.assertIs(w.slot_records[0], left)
        self.assertEqual(w.record.config.iterations, 7)
        self.assertIsNone(w.slot_records[2])
        QTest.keyClick(w.image_boxes[4], Qt.Key.Key_Space)
        self.assertEqual(w.active_slot, 2)
        w.select_history(0)
        self.assertIs(w.slot_records[2], w.history[0])
        self.assertIs(w.slot_records[0], left)
        self.assertEqual(w.slot_records[1].config.iterations, 7)
        self.assertEqual(len(w.history), 5)
        self.assertEqual(w.iterations.value(), 7)
        QTest.mouseClick(w.output_view.viewport(), Qt.MouseButton.LeftButton)
        self.assertIs(w.record, left)
        self.assertEqual(w.table.currentRow(), 3)

    def test_history_slot_context_keeps_current_inputs_and_metrics_separate(self):
        w = self.window
        w.load_sample()
        w.compare_methods()
        self.wait_for_worker()
        cells = list(w.history)
        w.load_bars()
        current_observation, current_truth = w.observation, w.ground_truth
        w.data_range.setValue(2)
        w.run_restoration()
        self.wait_for_worker()
        bars = w.record
        w.select_slot(1)
        w.select_history(0)
        self.assertIs(w.record, cells[0])
        self.assertIsNone(w.slot_records[0])
        self.assertIsNone(w.slot_records[2])
        self.assertIs(w.observation, current_observation)
        self.assertIs(w.ground_truth, current_truth)
        self.assertEqual(w.data_range.value(), 2)
        np.testing.assert_array_equal(w.input_view.array, cells[0].observation.pixels)
        np.testing.assert_array_equal(w.truth_view.array, cells[0].ground_truth.pixels)
        self.assertIn(
            f"PSNR {cells[0].input_quality['psnr']:.3f} dB", w.baseline.text()
        )
        self.assertIn("L=1", w.table.item(0, 2).text())
        self.assertIn("L=2", w.table.item(3, 2).text())
        w.select_slot(2)
        self.assertIsNone(w.record)
        self.assertTrue(w.current_button.isEnabled())
        self.assertEqual(
            w.baseline.text(),
            f"Degraded input: MSE {cells[0].input_quality['mse']:.5g}; PSNR {cells[0].input_quality['psnr']:.3f} dB; SSIM {cells[0].input_quality['ssim']:.5f}",
        )
        w.select_history(3)
        self.assertIs(w.record, bars)
        self.assertIsNone(w.slot_records[1])
        w.current_button.click()
        self.wait_for_worker()
        self.assertFalse(w.results_visible)
        self.assertIn(f"PSNR {bars.input_quality['psnr']:.3f} dB", w.baseline.text())
        self.assertEqual(len(w.history), 4)
        np.testing.assert_array_equal(w.input_view.array, current_observation.pixels)

    def test_slot_eviction_stable_identifiers_and_failed_run_retention(self):
        w = self.window
        w.load_sample()
        w.run_restoration()
        self.wait_for_worker()
        first = w.record
        w.select_slot(2)
        w.select_history(0)
        w.select_slot(0)
        for count in range(8):
            w.accept_record(replace(first, seconds=count + 1), 0)
        self.assertEqual(len(w.history), 8)
        self.assertIsNone(w.slot_records[2])
        self.assertNotIn(id(first), w.run_ids)
        self.assertEqual(w.table.item(0, 0).text(), "2: RL")
        self.assertEqual(w.table.item(7, 0).text(), "9: RL")
        before = list(w.slot_records)
        w.select_slot(1)

        def fail():
            raise ValueError("Slot failure regression")

        w.start_operation(fail, lambda record: w.accept_record(record, 1))
        self.wait_for_worker()
        self.assertTrue(all(a is b for a, b in zip(before, w.slot_records)))
        self.assertEqual(len(w.history), 8)
        self.assertIsNone(w.record)
        w.clear_history()
        self.assertEqual(w.slot_records, [None, None, None])
        self.assertFalse(w.results_visible)
        self.assertIsNone(w.view_record)

    def test_history_headers_fill_width_and_keep_layout_without_bold(self):
        w = self.window
        w.load_sample()
        for points in (9, 12, 16):
            font = w.font()
            font.setPointSize(points)
            w.setFont(font)
            for width in (1100, 1800):
                w.resize(width, 1000)
                self.app.processEvents()
                table = w.table
                table.fit_columns()
                columns = [table.columnWidth(i) for i in range(table.columnCount())]
                metrics = table.horizontalHeader().fontMetrics()
                self.assertGreaterEqual(sum(columns), table.viewport().width())
                for i, heading in enumerate(table.HEADINGS):
                    self.assertGreaterEqual(
                        columns[i], metrics.horizontalAdvance(heading) + 24
                    )
                self.assertFalse(table.horizontalHeader().font().bold())
                self.assertEqual(table.horizontalHeader().font().pointSize(), points)
                self.assertFalse(table.horizontalHeader().highlightSections())
                # Row contents must never trigger a different column-sizing policy.
                w.run_restoration()
                self.wait_for_worker()
                self.app.processEvents()
                self.assertEqual(
                    columns, [table.columnWidth(i) for i in range(table.columnCount())]
                )
                self.assertIn("L=1", table.item(0, 2).text())
                w.clear_history()
                self.app.processEvents()
        for removed in ("other_run", "third_run", "result_info", "context_info"):
            self.assertFalse(hasattr(w, removed))

    def test_load_run_stale_save_and_errors(self):
        self.window.control_pages.setCurrentIndex(1)
        self.assertFalse(self.window.run_button.isEnabled())
        with patch("PySide6.QtWidgets.QMessageBox.warning") as warning:
            self.window.run_restoration()
            self.assertTrue(warning.called)
        self.window.control_pages.setCurrentIndex(0)
        QTest.mouseClick(self.window.sample_button, Qt.MouseButton.LeftButton)
        self.assertEqual(self.window.observation.pixels.shape, (256, 256))
        self.assertEqual(self.window.control_pages.currentIndex(), 0)
        self.window.control_pages.setCurrentIndex(1)
        QTest.mouseClick(self.window.run_button, Qt.MouseButton.LeftButton)
        self.assertFalse(self.window.controls.isEnabled())
        self.wait_for_worker()
        self.assertTrue(self.window.controls.isEnabled())
        received_on = []
        self.window.start_operation(
            lambda: "finished", lambda _: received_on.append(QThread.currentThread())
        )
        self.wait_for_worker()
        self.assertEqual(received_on, [self.app.thread()])
        self.assertIsNotNone(self.window.record)
        snapshot = self.window.record.output.copy()
        self.window.iterations.setValue(10)
        self.assertIn(
            "Earlier configuration",
            self.window.table.item(self.window.table.currentRow(), 8).text(),
        )
        np.testing.assert_array_equal(self.window.record.output, snapshot)
        self.window.low.setValue(-1)
        self.window.high.setValue(2)
        self.window.refresh_views()
        np.testing.assert_array_equal(self.window.record.output, snapshot)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = str(Path(directory) / "restoration.tif")
            with patch(
                "PySide6.QtWidgets.QFileDialog.getSaveFileName", return_value=(path, "")
            ):
                self.window.save_output()
            self.assertFalse(self.warning.called, str(self.warning.call_args))
            np.testing.assert_array_equal(
                read_image(path).pixels, snapshot.astype(np.float32)
            )

    def test_worker_failure_and_close_guard(self):
        def fail():
            time.sleep(0.05)
            raise RuntimeError("Expected worker failure")

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
        self.assertEqual(
            [r.config.method for r in self.window.history], ["IF", "TRIF", "RL"]
        )
        self.assertEqual(self.window.table.rowCount(), 3)
        self.assertIsNotNone(self.window.record.quality)
        self.window.select_slot(2)
        self.assertGreater(self.window.inspector.iteration.count(), 0)
        self.assertIsNotNone(self.window.other_view.array)
        self.window.input_view._auto_fit = False
        self.window.input_view.scale(2, 2)
        self.window.input_view._notify()
        for view in self.window.views:
            self.assertEqual(
                view.transform().m11(), self.window.input_view.transform().m11()
            )
            self.assertFalse(view._auto_fit)
        self.window.fit_views()
        self.assertTrue(all(v._auto_fit for v in self.window.views))
        self.window.method.setCurrentIndex(self.window.method.findData("TRIF"))
        self.window.sigma.setValue(3)
        np.testing.assert_array_equal(self.window.observation.pixels, original)
        self.window.select_history(0)
        self.assertEqual(self.window.record.config.method, "IF")
        self.window.clear_history()
        self.assertEqual(len(self.window.history), 0)
        self.assertIsNone(self.window.inspector.record)
        self.window.method.setCurrentIndex(self.window.method.findData("IF"))
        self.window.load_bars()
        self.assertEqual(self.window.method.currentData(), "RL")

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
            record = self.window.record
            with (
                patch(
                    "PySide6.QtWidgets.QFileDialog.getExistingDirectory",
                    return_value="",
                ),
                patch("PySide6.QtWidgets.QInputDialog.getText") as ask_name,
            ):
                self.window.export_experiment()
                ask_name.assert_not_called()
            with (
                patch(
                    "PySide6.QtWidgets.QFileDialog.getExistingDirectory",
                    return_value=directory,
                ),
                patch(
                    "PySide6.QtWidgets.QInputDialog.getText",
                    return_value=("saved experiment", False),
                ),
            ):
                self.window.export_experiment()
            self.assertIs(self.window.record, record)
            self.assertFalse(Path(folder).exists())
            with (
                patch(
                    "PySide6.QtWidgets.QFileDialog.getExistingDirectory",
                    return_value=directory,
                ),
                patch(
                    "PySide6.QtWidgets.QInputDialog.getText",
                    return_value=("saved experiment", True),
                ),
            ):
                self.window.export_experiment()
            self.wait_for_worker()
            self.assertIn("Experiment saved:", self.window.statusBar().currentMessage())
            with patch(
                "PySide6.QtWidgets.QFileDialog.getOpenFileName",
                return_value=(str(Path(folder) / "experiment.json"), ""),
            ):
                self.window.open_experiment()
                self.wait_for_worker()
            reopened = self.window.record
            self.assertEqual(
                reopened.saved_from, str((Path(folder) / "experiment.json").resolve())
            )
            np.testing.assert_array_equal(reopened.output, output)
            self.assertEqual(reopened.quality, record.quality)
            self.assertEqual(reopened.seconds, record.seconds)
            self.assertEqual(reopened.curve, record.curve)
            self.assertIs(self.window.inspector.record, reopened)
            self.assertEqual(self.window.inspector.iteration.count(), len(record.steps))
            self.assertEqual(self.window.tabs.currentIndex(), 0)
            self.assertIn(
                "No restoration was executed", self.window.statusBar().currentMessage()
            )
            self.window.run_restoration()
            self.wait_for_worker()
            np.testing.assert_allclose(
                self.window.record.output, output, rtol=0, atol=1e-12
            )
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
        self.assertLessEqual(mapped.width(), view.viewport().width() + 2)
        self.assertLessEqual(mapped.height(), view.viewport().height() + 2)
        self.window.inspector.metric_choice.setCurrentText("SSIM")
        self.app.processEvents()
        self.assertFalse(plot.grab().isNull())

    def test_keyboard_selection_export_and_current_inputs(self):
        self.window.load_sample()
        self.window.compare_methods()
        self.wait_for_worker()
        self.window.select_slot(0)
        self.window.table.setFocus()
        QTest.keyClick(self.window.table, Qt.Key.Key_Down)
        self.assertIs(self.window.record, self.window.history[1])
        self.assertEqual(self.window.record.config.method, "TRIF")
        np.testing.assert_array_equal(
            self.window.output_view.array, self.window.history[1].output
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = str(Path(directory) / "keyboard-selected.tif")
            with patch(
                "PySide6.QtWidgets.QFileDialog.getSaveFileName", return_value=(path, "")
            ):
                self.window.save_output()
            np.testing.assert_array_equal(
                read_image(path).pixels,
                self.window.history[1].output.astype(np.float32),
            )
        self.window.load_bars()
        self.assertIsNone(self.window.record)
        self.assertEqual(len(self.window.history), 3)
        np.testing.assert_array_equal(
            self.window.input_view.array, self.window.observation.pixels
        )
        np.testing.assert_array_equal(
            self.window.truth_view.array, self.window.ground_truth.pixels
        )
        self.assertIsNone(self.window.output_view.array)
        self.assertFalse(self.window.save_button.isEnabled())
        self.assertFalse(self.window.export_button.isEnabled())
        self.window.select_history(0)
        self.window.select_history(1)
        self.assertIn(
            "Earlier configuration",
            self.window.table.item(self.window.table.currentRow(), 8).text(),
        )
        self.window.current_button.click()
        self.assertIsNone(self.window.record)
        np.testing.assert_array_equal(
            self.window.input_view.array, self.window.observation.pixels
        )
        self.assertEqual(len(self.window.history), 3)

    def test_truth_edits_refresh_without_discarding_history(self):
        self.window.set_observation(
            read_image(sample_path("cells_s2_p200_r0_input.tif"))
        )
        truth_path = str(sample_path("cells_gt.tif"))
        self.window.tabs.setCurrentIndex(1)
        with patch(
            "PySide6.QtWidgets.QFileDialog.getOpenFileName",
            return_value=(truth_path, ""),
        ):
            self.window.open_truth()
        self.assertEqual(self.window.tabs.currentIndex(), 0)
        np.testing.assert_array_equal(
            self.window.truth_view.array, self.window.ground_truth.pixels
        )
        self.window.run_restoration()
        self.wait_for_worker()
        self.window.clear_truth()
        self.assertIsNone(self.window.record)
        self.assertIsNone(self.window.truth_view.array)
        self.assertIsNotNone(self.window.history[0].ground_truth)
        with patch(
            "PySide6.QtWidgets.QFileDialog.getOpenFileName",
            return_value=(truth_path, ""),
        ):
            self.window.open_source()
        np.testing.assert_array_equal(
            self.window.truth_view.array, self.window.source.pixels
        )

    def test_input_messages_resize_after_saved_example_and_source_loading(self):
        for points in (9, 12, 16):
            font = self.window.font()
            font.setPointSize(points)
            self.window.setFont(font)
            for width, height, sidebar in (
                (960, 640, 300),
                (1280, 820, 470),
                (960, 640, 300),
            ):
                with self.subTest(points=points, sidebar=sidebar, width=width):
                    self.window.resize(width, height)
                    self.window.centralWidget().setSizes([sidebar, width - sidebar])
                    self.window.control_pages.setCurrentIndex(0)
                    self.window.sample_button.click()
                    self.app.processEvents()
                    selector = self.window.saved_example.geometry()
                    button = self.window.sample_button.geometry()
                    self.assertLessEqual(
                        abs(selector.center().y() - button.center().y()), 1
                    )
                    self.assertLess(selector.right(), button.left())
                    self.assertGreaterEqual(
                        button.width(),
                        self.window.sample_button.minimumSizeHint().width(),
                    )
                    with patch.object(
                        self.window,
                        "choose_image",
                        return_value=str(sample_path("cells_gt.tif")),
                    ):
                        self.window.open_source()
                    for _ in range(10):
                        self.app.processEvents()
                    self.assertTrue(self.window.simulate_button.isEnabled())
                    self.assertEqual(self.window.control_pages.currentIndex(), 0)
                    self.assertIn("Source ground truth:", self.window.truth_info.text())
                    self.assertNotIn("Click", self.window.truth_info.text())
                    self.assertNotIn("Set intensity L", self.window.input_info.text())
                    for label in (
                        self.window.truth_info,
                        self.window.input_info,
                    ):
                        self.assertGreaterEqual(
                            label.height(), label.heightForWidth(label.width())
                        )
                    for preceding, following in (
                        (self.window.clear_truth_button, self.window.photons),
                        (self.window.photons, self.window.seed),
                        (self.window.simulate_button, self.window.image_information),
                        (self.window.input_info, self.window.truth_info),
                    ):
                        self.assertLess(
                            preceding.geometry().bottom(), following.geometry().top()
                        )
                    self.window.sample_button.click()
                    for _ in range(10):
                        self.app.processEvents()
                    self.assertFalse(self.window.simulate_button.isEnabled())
                    self.assertLessEqual(
                        self.window.truth_info.minimumHeight(),
                        self.window.truth_info.heightForWidth(
                            self.window.truth_info.width()
                        ),
                    )

    def test_source_waits_for_generation_without_discarding_history(self):
        self.window.load_sample()
        self.window.run_restoration()
        self.wait_for_worker()
        previous = self.window.record
        pixels = np.linspace(0.2, 0.8, 16 * 24).reshape(16, 24)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "clean source.tif"
            save_tiff(path, pixels)
            with patch.object(self.window, "choose_image", return_value=str(path)):
                self.window.open_source()
            source = self.window.source
            self.assertIsNone(self.window.observation)
            self.assertIsNone(self.window.degradation)
            self.assertIsNone(self.window.input_view.array)
            self.assertIsNone(self.window.output_view.array)
            self.assertIsNone(self.window.record)
            self.assertIsNone(self.window.inspector.record)
            self.assertTrue(self.window.observation_prompt.isVisible())
            self.app.processEvents()
            self.assertIs(
                self.window.input_display.currentWidget(),
                self.window.observation_prompt,
            )
            self.assertTrue(
                self.window.input_display.rect().contains(
                    self.window.observation_prompt.geometry()
                )
            )
            np.testing.assert_array_equal(self.window.truth_view.array, source.pixels)
            self.assertIn(
                "Source file: clean source.tif", self.window.input_info.text()
            )
            self.assertFalse(self.window.run_button.isEnabled())
            self.assertFalse(self.window.compare_button.isEnabled())
            self.assertFalse(self.window.save_button.isEnabled())
            self.assertEqual(self.window.history, [previous])

            self.window.select_history(0)
            self.assertFalse(self.window.observation_prompt.isVisible())
            np.testing.assert_array_equal(
                self.window.input_view.array, previous.observation.pixels
            )
            self.window.current_button.click()
            self.assertIsNone(self.window.input_view.array)
            np.testing.assert_array_equal(self.window.truth_view.array, source.pixels)
            self.assertTrue(self.window.observation_prompt.isVisible())

            self.window.psf_mode.setCurrentText("Gaussian")
            self.window.sigma.setValue(1)
            self.window.photons.setValue(180)
            self.window.seed.setValue(18)
            self.assertTrue(self.window.simulate_button.isEnabled())
            self.assertIsNone(self.window.observation)
            for action in (self.window.run_restoration, self.window.compare_methods):
                self.warning.reset_mock()
                action()
                self.assertIsNone(self.window.worker)
                self.assertIn("Generate degradation", str(self.warning.call_args))

            # A failed simulation must leave the source awaiting Generate.
            self.window.data_range.setValue(0.1)
            self.window.simulate_button.click()
            self.wait_for_worker()
            self.assertIsNone(self.window.observation)
            self.assertIsNone(self.window.input_view.array)
            self.assertTrue(self.window.observation_prompt.isVisible())
            self.assertFalse(self.window.run_button.isEnabled())
            self.assertFalse(self.window.compare_button.isEnabled())
            self.assertTrue(self.window.simulate_button.isEnabled())

            self.window.data_range.setValue(1)
            self.window.simulate_button.click()
            self.wait_for_worker()
            self.assertEqual(self.window.observation.pixels.shape, (16, 24))
            self.assertEqual(self.window.current_psf().pixels.shape, (16, 24))
            self.assertFalse(self.window.observation_prompt.isVisible())
            self.assertTrue(self.window.run_button.isEnabled())
            self.assertTrue(self.window.compare_button.isEnabled())
            self.assertIs(
                self.window.input_display.currentWidget(), self.window.input_view
            )
            np.testing.assert_array_equal(self.window.truth_view.array, source.pixels)
            np.testing.assert_array_equal(
                self.window.input_view.array, self.window.observation.pixels
            )
            observed = self.window.observation.pixels.copy()
            self.window.seed.setValue(19)
            self.window.photons.setValue(200)
            self.window.sigma.setValue(2)
            np.testing.assert_array_equal(self.window.observation.pixels, observed)
            self.assertIs(self.window.source, source)
            self.assertEqual(self.window.history, [previous])

    def test_shared_psf_and_intensity_change_restoration_without_regeneration(self):
        self.window.saved_example.setCurrentIndex(1)
        self.window.sample_button.click()
        observed = self.window.observation
        original_psf = self.window.current_psf()
        self.window.run_restoration()
        self.wait_for_worker()
        previous = self.window.record
        previous_psf = previous.psf.pixels.copy()
        np.testing.assert_array_equal(
            previous_psf, prepare_psf(original_psf.pixels, observed.pixels.shape)
        )

        self.window.control_pages.setCurrentIndex(1)
        self.app.processEvents()
        self.assertIs(
            self.window.psf_mode.parentWidget(), self.window.control_pages.widget(1)
        )
        self.window.psf_mode.setCurrentText("Gaussian")
        self.window.sigma.setValue(3)
        self.window.data_range.setValue(2)
        changed_psf = self.window.current_psf()
        self.assertFalse(np.array_equal(changed_psf.pixels, original_psf.pixels))
        self.assertIs(self.window.observation, observed)
        self.assertEqual(self.window.history, [previous])
        self.assertIn(
            "Earlier configuration",
            self.window.table.item(self.window.table.currentRow(), 8).text(),
        )
        config, revision = self.window.current_config(), self.window.revision
        for section in (2, 1, 3, 0, 1, 0, 1):
            self.window.control_pages.setCurrentIndex(section)
            self.app.processEvents()
            self.assertEqual(self.window.current_config(), config)
            self.assertEqual(self.window.revision, revision)
            self.assertIs(self.window.observation, observed)
            self.assertEqual(self.window.psf_mode.isVisible(), section == 1)
            self.assertEqual(self.window.data_range.isVisible(), section == 1)
            np.testing.assert_array_equal(
                self.window.current_psf().pixels, changed_psf.pixels
            )
        self.window.run_button.click()
        self.wait_for_worker()
        current = self.window.record
        np.testing.assert_array_equal(current.observation.pixels, observed.pixels)
        np.testing.assert_array_equal(current.psf.pixels, changed_psf.pixels)
        self.assertEqual(current.config.data_range, 2)
        np.testing.assert_allclose(
            current.output,
            rl(observed.pixels, otf(changed_psf.pixels), 5),
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_array_equal(previous.psf.pixels, previous_psf)
        self.assertEqual(previous.config.data_range, 1)

        self.window.control_pages.setCurrentIndex(0)
        self.app.processEvents()
        self.assertFalse(self.window.data_range.isVisible())
        self.assertEqual(self.window.sigma.value(), 3)
        self.assertEqual(self.window.data_range.value(), 2)
        self.assertIs(self.window.observation, observed)
        # Iteration exploration remains available after loading Bars reference.
        self.window.saved_example.setCurrentIndex(1)
        self.window.sample_button.click()
        self.window.iterations.setValue(200)
        self.assertEqual(self.window.iterations.value(), 200)
        self.assertEqual(len(self.window.history), 2)
        self.assertFalse(self.warning.called, str(self.warning.call_args))

    def test_pending_source_with_imported_psf_and_observation_loading(self):
        with patch.object(
            self.window, "choose_image", return_value=str(sample_path("cells_gt.tif"))
        ):
            self.window.open_source()
        self.window.psf_mode.setCurrentText("Imported PSF")
        self.assertFalse(self.window.simulate_button.isEnabled())
        with patch.object(
            self.window, "choose_image", return_value=str(sample_path("psf_s2.tif"))
        ):
            self.window.open_psf()
        self.assertTrue(self.window.simulate_button.isEnabled())
        self.assertFalse(self.window.run_button.isEnabled())
        self.window.simulate_button.click()
        self.wait_for_worker()
        self.assertTrue(self.window.run_button.isEnabled())
        self.assertIsNotNone(self.window.observation)

        with patch.object(
            self.window, "choose_image", return_value=str(sample_path("bars_gt.tif"))
        ):
            self.window.open_source()
        self.assertIsNone(self.window.input_view.array)
        with patch.object(self.window, "choose_image", return_value=""):
            self.window.open_source()
        self.assertIsNone(self.window.observation)
        self.assertIn("bars_gt.tif", self.window.source.source)
        self.window.load_sample()
        self.assertIsNone(self.window.source)
        self.assertFalse(self.window.observation_prompt.isVisible())
        self.assertTrue(self.window.run_button.isEnabled())
        with patch.object(
            self.window, "choose_image", return_value=str(sample_path("bars_gt.tif"))
        ):
            self.window.open_source()
        with patch.object(
            self.window,
            "choose_image",
            return_value=str(sample_path("cells_s2_p200_r0_input.tif")),
        ):
            self.window.open_observation()
        self.assertIsNone(self.window.source)
        self.assertFalse(self.window.observation_prompt.isVisible())
        self.assertTrue(self.window.run_button.isEnabled())
        np.testing.assert_array_equal(
            self.window.input_view.array, self.window.observation.pixels
        )

    def test_sidebar_fields_stay_inline_and_input_information_follows_actions(self):
        # Cover the startup width and the minimum supported window without
        # widening the sidebar to reveal a label or its control.
        for width, height, points in (
            (1280, 820, 9),
            (960, 640, 9),
            (960, 640, 12),
            (1280, 820, 16),
        ):
            size = (width, height)
            font = self.window.font()
            font.setPointSize(points)
            self.window.setFont(font)
            self.window.resize(*size)
            self.window.centralWidget().setSizes([300, size[0] - 300])
            for section in range(self.window.control_pages.count()):
                self.window.control_pages.setCurrentIndex(section)
                for _ in range(10):
                    self.app.processEvents()
                page_scroll = (
                    self.window.control_pages.widget(section)
                    .parentWidget()
                    .parentWidget()
                )
                self.assertEqual(page_scroll.horizontalScrollBar().maximum(), 0)
                for control, label in self.window.control_labels.items():
                    if not control.isVisible():
                        continue
                    with self.subTest(size=size, points=points, label=label.text()):
                        self.assertTrue(label.text().endswith(":"))
                        self.assertFalse(label.wordWrap())
                        self.assertLessEqual(
                            abs(
                                label.geometry().center().y()
                                - control.geometry().center().y()
                            ),
                            1,
                        )
                        self.assertLess(
                            label.geometry().right(), control.geometry().left()
                        )
                        viewport = self.window.control_pages.widget(
                            section
                        ).parentWidget()
                        position = control.mapTo(viewport, QPoint())
                        self.assertGreaterEqual(position.x(), 0)
                        self.assertLessEqual(
                            position.x() + control.width(), viewport.width()
                        )
                for control in (
                    self.window.psf_mode,
                    self.window.psf_file,
                    self.window.sigma,
                    self.window.data_range,
                ):
                    self.assertEqual(control.isVisible(), section == 1)
                    self.assertIs(
                        control.parentWidget(),
                        self.window.control_pages.widget(section)
                        if section == 1
                        else self.window.psf_parking,
                    )
                if section == 1:
                    for index in range(self.window.method.count()):
                        self.window.method.setCurrentIndex(index)
                        self.assertGreaterEqual(
                            self.window.method.width(),
                            self.window.method.fontMetrics().horizontalAdvance(
                                self.window.method.currentText()
                            )
                            + 20,
                        )
                    layout = self.window.control_pages.widget(section).layout()
                    for row, control in enumerate(
                        (
                            self.window.psf_mode,
                            self.window.psf_file,
                            self.window.sigma,
                            self.window.data_range,
                            self.window.method,
                        )
                    ):
                        self.assertEqual(layout.getWidgetPosition(control)[0], row)
            self.window.control_pages.setCurrentIndex(0)
            self.window.sample_button.click()
            for _ in range(10):
                self.app.processEvents()
            page = self.window.control_pages.widget(0)
            ordered = [
                self.window.policy,
                self.window.saved_example_row,
                self.window.load_button,
                next(
                    b
                    for b in page.findChildren(type(self.window.load_button))
                    if b.text() == "Open source for simulation…"
                ),
                next(
                    b
                    for b in page.findChildren(type(self.window.load_button))
                    if b.text() == "Open optional ground truth…"
                ),
                self.window.clear_truth_button,
                self.window.photons,
                self.window.seed,
                self.window.simulate_button,
                self.window.image_information,
            ]
            for preceding, following in pairwise(ordered):
                self.assertLess(
                    preceding.geometry().bottom(), following.geometry().top()
                )
            self.assertIn(
                "File: cells_s2_p200_r0_input.tif", self.window.input_info.text()
            )
            self.assertIn("Size: 256 × 256 px", self.window.input_info.text())
            self.assertIn("Pixel values:", self.window.input_info.text())
            self.assertNotIn("Set intensity L", self.window.input_info.text())
            self.assertIs(
                self.window.input_info.parentWidget(), self.window.image_information
            )
            self.assertIs(
                self.window.truth_info.parentWidget(), self.window.image_information
            )

    def test_inspector_quality_fit_and_reference_keyboard_selection(self):
        self.window.load_sample()
        self.window.run_restoration()
        self.wait_for_worker()
        panel = self.window.inspector
        for index in range(panel.phase.count()):
            panel.phase.setCurrentIndex(index)
            self.assertIn(
                "Quality of updated x(k+1), iteration 1", panel.range_info.text()
            )
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
        self.assertEqual(table.item(1, 0).text(), "cells")
        self.assertEqual(table.item(1, 1).text(), "TRIF")
        np.testing.assert_array_equal(
            self.window.reference_panel.python_view.array,
            self.window.reference_panel.rows[1]["record"].output,
        )
        np.testing.assert_array_equal(
            self.window.reference_panel.saved_view.array,
            self.window.reference_panel.rows[1]["reference"].pixels,
        )
        np.testing.assert_array_equal(
            self.window.reference_panel.diff_view.array,
            self.window.reference_panel.rows[1]["difference"],
        )

    def test_reference_headers_stay_readable_before_and_after_comparison(self):
        w = self.window
        w.tabs.setCurrentIndex(2)
        panel = w.reference_panel
        table = panel.table
        self.assertEqual(
            tuple(table.horizontalHeaderItem(i).text() for i in range(6)),
            (
                "Image",
                "Method",
                "Max |difference|",
                "Python PSNR (dB)",
                "Python SSIM",
                "ImageJ PSNR (saved)",
            ),
        )
        self.assertEqual(table.columnCount(), 6)
        for removed in ("details", "note", "detail"):
            self.assertFalse(hasattr(panel, removed))
        w.verify_references()
        self.wait_for_worker()
        rows = panel.rows
        for points in (9, 12, 16):
            font = w.font()
            font.setPointSize(points)
            w.setFont(font)
            for width in (1100, 1800, 2400):
                w.resize(width, 1000)
                table.setRowCount(0)
                self.app.processEvents()
                before = [table.columnWidth(i) for i in range(6)]
                metrics = table.horizontalHeader().fontMetrics()
                self.assertGreaterEqual(sum(before), table.viewport().width())
                if points == 9 and width == 2400:
                    self.assertEqual(sum(before), table.viewport().width())
                for i, size in enumerate(before):
                    self.assertGreaterEqual(
                        size, metrics.horizontalAdvance(table.HEADINGS[i]) + 24
                    )
                    self.assertFalse(table.isColumnHidden(i))
                for _ in range(2):
                    panel.accept_rows(rows)
                    self.app.processEvents()
                    self.assertEqual(before, [table.columnWidth(i) for i in range(6)])
                    for index, result in enumerate(rows):
                        table.selectRow(index)
                        self.assertFalse(table.horizontalHeader().font().bold())
                        self.assertFalse(table.horizontalHeader().highlightSections())
                        self.assertEqual(
                            table.horizontalHeader().font().pointSize(), points
                        )
                        self.assertEqual(
                            table.item(index, 2).text(), f"{result['max_abs']:.9g}"
                        )
                        self.assertEqual(
                            table.item(index, 3).text(),
                            f"{result['record'].quality['psnr']:.9g}",
                        )
                        self.assertEqual(
                            table.item(index, 4).text(),
                            f"{result['record'].quality['ssim']:.9g}",
                        )
                        self.assertEqual(
                            table.item(index, 5).text(),
                            f"{result['saved_imagej_psnr']:.9g}",
                        )
                        for column in range(6):
                            self.assertGreaterEqual(
                                table.columnWidth(column),
                                table.fontMetrics().horizontalAdvance(
                                    table.item(index, column).text()
                                )
                                + 16,
                            )
                        np.testing.assert_array_equal(
                            panel.diff_view.array, result["difference"]
                        )
                if sum(before) > table.viewport().width():
                    self.assertGreater(table.horizontalScrollBar().maximum(), 0)

    def test_contextual_controls_numeric_steps_and_saved_presets(self):
        self.assertFalse(self.window.run_button.isEnabled())
        self.assertFalse(self.window.simulate_button.isEnabled())
        self.assertFalse(self.window.clear_history_button.isEnabled())
        self.assertFalse(self.window.results_visible)
        self.assertTrue(all(run is None for run in self.window.slot_records))
        self.assertEqual(self.window.saved_example.count(), 3)
        for index, name in enumerate(["cells", "bars", "fibers"]):
            self.assertEqual(
                self.window.saved_example.itemText(index),
                f"{name.title()} — reference",
            )
            self.window.threshold.setValue(0.03)
            self.window.regularization.setValue(0.02)
            self.window.iterations.setValue(80)
            self.window.epsilon.setValue(1e-5)
            self.window.data_range.setValue(2)
            self.window.control_pages.setCurrentIndex(0)
            self.window.saved_example.setCurrentIndex(index)
            self.window.sample_button.click()
            self.assertIn(name, self.window.observation.source)
            self.assertEqual(self.window.threshold.value(), 0.1)
            self.assertEqual(self.window.regularization.value(), 0.1)
            self.assertEqual(self.window.iterations.value(), 5)
            self.assertEqual(self.window.epsilon.value(), 1e-6)
            self.assertEqual(self.window.data_range.value(), 1)
            self.assertEqual(self.window.control_pages.currentIndex(), 0)
            self.assertEqual(self.window.psf_mode.currentText(), "Imported PSF")
            self.assertIn("psf_s2.tif", self.window.imported_psf.source)
            self.assertIn(name, self.window.ground_truth.source)
            self.assertIsNone(self.window.worker)
            self.assertIsNone(self.window.record)
            self.assertEqual(self.window.history, [])
        self.assertTrue(self.window.run_button.isEnabled())
        self.assertFalse(self.window.sigma.isEnabled())
        self.assertFalse(self.window.threshold.isEnabled())
        self.assertFalse(self.window.simulate_button.isEnabled())
        self.window.method.setCurrentIndex(self.window.method.findData("IF"))
        self.assertTrue(self.window.threshold.isEnabled())
        self.assertFalse(self.window.epsilon.isEnabled())
        self.assertFalse(self.window.capture.isEnabled())
        for control in [
            self.window.epsilon,
            self.window.threshold,
            self.window.regularization,
        ]:
            before = control.value()
            control.stepUp()
            self.assertGreater(control.value(), before)
            self.assertLess(control.value(), 2 * before)
            control.stepDown()
            self.assertAlmostEqual(control.value(), before, places=12)
        self.window.psf_mode.setCurrentText("Gaussian")
        self.assertTrue(self.window.sigma.isEnabled())
        self.assertFalse(self.window.psf_file.isEnabled())
        self.window.psf_mode.setCurrentText("Imported PSF")
        self.assertTrue(self.window.psf_file.isEnabled())
        self.assertEqual(self.window.psf_file.text(), "psf_s2.tif")

    def test_psf_file_picker_modes_cancel_and_invalid_file_preserve_state(self):
        self.window.control_pages.setCurrentIndex(1)
        field = self.window.psf_file
        label = self.window.control_labels[field]
        self.assertEqual(label.text(), "Open PSF:")
        self.assertEqual(field.text(), "")
        self.assertFalse(field.isEnabled())
        self.assertFalse(label.isEnabled())
        with patch.object(self.window, "choose_image", return_value="") as choose:
            QTest.mouseClick(field, Qt.MouseButton.LeftButton)
            choose.assert_not_called()
        self.window.psf_mode.setCurrentText("Imported PSF")
        self.assertTrue(field.isEnabled())
        self.assertTrue(label.isEnabled())
        self.assertFalse(self.window.run_button.isEnabled())
        with patch.object(
            self.window, "choose_image", return_value=str(sample_path("psf_s2.tif"))
        ) as choose:
            QTest.mouseClick(field, Qt.MouseButton.LeftButton)
            choose.assert_called_once_with("Open centered PSF")
        selected = self.window.imported_psf
        self.assertEqual(field.text(), "psf_s2.tif")
        self.assertTrue(field.isReadOnly())
        self.window.sigma.setValue(3)
        with patch.object(
            self.window, "choose_image", return_value=str(sample_path("cells_gt.tif"))
        ):
            self.window.open_source()
        for section in (1, 0):
            self.window.control_pages.setCurrentIndex(section)
            self.app.processEvents()
            for key in (Qt.Key.Key_Return, Qt.Key.Key_Space):
                with patch.object(
                    self.window, "choose_image", return_value=""
                ) as choose:
                    field.setFocus()
                    QTest.keyClick(field, key)
                    choose.assert_called_once()
                self.assertIs(self.window.imported_psf, selected)
            self.window.psf_mode.setCurrentText("Gaussian")
            self.assertFalse(field.isEnabled())
            self.assertFalse(label.isEnabled())
            self.assertTrue(self.window.sigma.isEnabled())
            self.assertEqual(field.text(), "psf_s2.tif")
            self.window.psf_mode.setCurrentText("Imported PSF")
            self.assertEqual(self.window.sigma.value(), 3)
        with patch.object(self.window, "choose_image", return_value="missing.tif"):
            QTest.mouseClick(field, Qt.MouseButton.LeftButton)
        self.assertTrue(self.warning.called)
        self.assertIs(self.window.imported_psf, selected)
        self.assertEqual(field.text(), "psf_s2.tif")

    def test_method_labels_keep_ids_and_reopen_all_algorithms(self):
        self.window.load_sample()
        self.window.control_pages.setCurrentIndex(1)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            for index, (code, name) in enumerate(
                [
                    ("RL", "Richardson-Lucy"),
                    ("IF", "Thresholded inverse filtering"),
                    ("TRIF", "Zeroth-order Tikhonov"),
                ]
            ):
                self.window.method.setCurrentIndex(index)
                self.assertEqual(self.window.method.currentText(), f"{code} ({name})")
                self.assertEqual(self.window.current_config().method, code)
                self.assertEqual(self.window.threshold.isEnabled(), code == "IF")
                self.assertEqual(self.window.regularization.isEnabled(), code == "TRIF")
                self.assertEqual(self.window.iterations.isEnabled(), code == "RL")
                self.window.run_button.click()
                self.wait_for_worker()
                record = self.window.record
                self.assertEqual(record.config.method, code)
                folder = Path(directory) / code
                export_bundle(folder, record)
                with patch(
                    "PySide6.QtWidgets.QFileDialog.getOpenFileName",
                    return_value=(str(folder / "experiment.json"), ""),
                ):
                    self.window.open_experiment()
                    self.wait_for_worker()
                self.assertEqual(self.window.current_config(), record.config)
                self.assertEqual(self.window.method.currentData(), code)
                np.testing.assert_array_equal(self.window.record.output, record.output)
                self.assertEqual(
                    self.window.inspector.phase.count(), 5 if code == "RL" else 0
                )
                self.window.run_button.click()
                self.wait_for_worker()
                np.testing.assert_allclose(
                    self.window.record.output, record.output, rtol=0, atol=1e-12
                )
        self.assertFalse(self.warning.called, str(self.warning.call_args))

    def test_reopen_selected_history_no_execution_and_invalid_bundle_keeps_workspace(
        self,
    ):
        w = self.window
        w.load_sample()
        w.run_restoration()
        self.wait_for_worker()
        original = w.record
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = export_bundle(Path(directory) / "selected", original)
            w.load_bars()
            w.run_restoration()
            self.wait_for_worker()
            existing = w.record
            w.select_slot(2)
            with (
                patch(
                    "PySide6.QtWidgets.QFileDialog.getOpenFileName",
                    return_value=(str(path), ""),
                ),
                patch(
                    "image_deconvolution.ui.main_window.restore",
                    side_effect=AssertionError("Reopen must not restore"),
                ),
            ):
                w.open_experiment()
                self.wait_for_worker()
            reopened = w.record
            self.assertEqual(len(w.history), 3)
            self.assertEqual(w.active_slot, 2)
            self.assertIs(w.slot_records[2], reopened)
            self.assertIsNone(w.slot_records[0])
            self.assertTrue(any(run is existing for run in w.history))
            np.testing.assert_array_equal(reopened.output, original.output)
            for iteration, phases in original.steps.items():
                w.inspector.iteration.setCurrentIndex(
                    w.inspector.iteration.findData(iteration)
                )
                for phase, array in phases.items():
                    w.inspector.phase.setCurrentText(phase)
                    np.testing.assert_array_equal(w.inspector.view.array, array)
            w.select_history(1)
            self.assertIs(w.inspector.record, existing)
            w.select_history(2)
            self.assertIs(w.inspector.record, reopened)
            snapshot = w.observation, w.current_config(), w.record, len(w.history)
            metadata = json.loads(path.read_text())
            metadata["result"]["array"]["file_sha256"] = "tampered"
            path.write_text(json.dumps(metadata))
            with patch(
                "PySide6.QtWidgets.QFileDialog.getOpenFileName",
                return_value=(str(path), ""),
            ):
                w.open_experiment()
                self.wait_for_worker()
            self.assertTrue(self.warning.called)
            self.assertIs(w.observation, snapshot[0])
            self.assertEqual(w.current_config(), snapshot[1])
            self.assertIs(w.record, snapshot[2])
            self.assertEqual(len(w.history), snapshot[3])

    def test_reopen_missing_input_locate_cancel_and_uncaptured_inspector(self):
        w = self.window
        w.load_sample()
        w.capture.setChecked(False)
        w.clear_truth()
        w.run_restoration()
        self.wait_for_worker()
        original = w.record
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = export_bundle(Path(directory) / "selected", original)
            moved = Path(directory) / "moved.npy"
            (path.parent / "observation.npy").rename(moved)
            for pick, expected_count in [("", 1), (str(moved), 2)]:
                with patch(
                    "PySide6.QtWidgets.QFileDialog.getOpenFileName",
                    side_effect=[(str(path), ""), (pick, "")],
                ):
                    w.open_experiment()
                    self.wait_for_worker()
                self.assertEqual(len(w.history), expected_count)
            self.assertEqual(w.inspector.iteration.count(), 0)
            self.assertIn("no RL checkpoints", w.inspector.range_info.text())
            self.assertIsNone(w.record.quality)
            self.assertEqual(w.baseline.text(), "Degraded input: N/A")
        self.assertFalse(self.warning.called, str(self.warning.call_args))

    def test_psf_visibility_tracks_source_and_degraded_transitions(self):
        def check_sections(source_present):
            observation = self.window.observation
            source = self.window.source
            config = self.window.current_config()
            for section in (0, 1, 2, 3, 0):
                self.window.control_pages.setCurrentIndex(section)
                self.app.processEvents()
                visible = section == 1 or (section == 0 and source_present)
                for label, control in self.window.psf_rows:
                    self.assertEqual(control.isVisible(), visible)
                    self.assertEqual(label.isVisible(), visible)
                self.assertIs(self.window.observation, observation)
                self.assertIs(self.window.source, source)
                self.assertEqual(self.window.current_config(), config)

        check_sections(False)
        self.window.load_sample()
        check_sections(False)
        for _ in range(2):
            with patch.object(
                self.window,
                "choose_image",
                return_value=str(sample_path("cells_gt.tif")),
            ):
                self.window.open_source()
            check_sections(True)
            self.window.simulate_button.click()
            self.wait_for_worker()
            check_sections(True)
            with patch.object(
                self.window,
                "choose_image",
                return_value=str(sample_path("bars_s2_p200_r0_input.tif")),
            ):
                self.window.open_observation()
            check_sections(False)
        self.assertFalse(self.warning.called, str(self.warning.call_args))

    def test_generate_preserves_source_pixels_and_render_for_all_parameters(self):
        with patch.object(
            self.window, "choose_image", return_value=str(sample_path("cells_gt.tif"))
        ):
            self.window.open_source()
        source = self.window.source
        original = source.pixels.copy()

        def import_psf():
            with patch.object(
                self.window, "choose_image", return_value=str(sample_path("psf_s2.tif"))
            ):
                self.window.open_psf()

        for name, change in [
            ("sigma", lambda: self.window.sigma.setValue(3)),
            ("photons", lambda: self.window.photons.setValue(50)),
            ("seed", lambda: self.window.seed.setValue(17)),
            ("imported PSF", import_psf),
            ("L", lambda: self.window.data_range.setValue(2)),
            ("Gaussian mode", lambda: self.window.psf_mode.setCurrentText("Gaussian")),
            ("manual range", lambda: self.window.seed.setValue(18)),
        ]:
            with self.subTest(parameter=name):
                if name == "manual range":
                    self.window.low.setValue(0.1)
                    self.window.high.setValue(0.8)
                    self.window.refresh_views()
                limits = self.window.low.value(), self.window.high.value()
                rendered = self.window.truth_view.scene().items()[0].pixmap().toImage()
                observation = self.window.observation
                change()
                self.assertIs(self.window.observation, observation)
                self.assertEqual(
                    self.window.truth_view.scene().items()[0].pixmap().toImage(),
                    rendered,
                )
                self.window.simulate_button.click()
                self.wait_for_worker()
                self.assertEqual(
                    (self.window.low.value(), self.window.high.value()), limits
                )
                self.assertIs(self.window.source, source)
                self.assertIs(self.window.ground_truth, source)
                np.testing.assert_array_equal(self.window.source.pixels, original)
                np.testing.assert_array_equal(self.window.truth_view.array, original)
                self.assertEqual(
                    self.window.truth_view.scene().items()[0].pixmap().toImage(),
                    rendered,
                )
                if observation is not None:
                    self.assertFalse(
                        np.array_equal(
                            observation.pixels, self.window.observation.pixels
                        )
                    )
                self.assertEqual(
                    self.window.degradation["data_range"],
                    self.window.data_range.value(),
                )
                self.assertEqual(
                    self.window.degradation["seed"], self.window.seed.value()
                )
                self.assertEqual(
                    self.window.degradation["photons"], self.window.photons.value()
                )
                low, high = limits
                y, x = np.unravel_index(
                    np.argmax(self.window.observation.pixels), original.shape
                )
                expected = int(
                    np.clip(
                        (self.window.observation.pixels[y, x] - low) / (high - low),
                        0,
                        1,
                    )
                    * 255
                )
                actual = self.window.input_view.scene().items()[0].pixmap().toImage()
                self.assertEqual(actual.pixelColor(int(x), int(y)).red(), expected)
        self.assertFalse(self.warning.called, str(self.warning.call_args))

    def test_preserved_integer_display_does_not_change_pixels_or_L(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "ramp.png"
            pixels = np.arange(256, dtype=np.uint8).reshape(16, 16)
            Image.fromarray(pixels).save(path)
            self.window.set_observation(read_image(path))
            self.assertEqual(self.window.high.value(), 255)
            self.assertIn("Image scale: [0, 255]", self.window.input_info.text())
            self.assertNotIn("Set intensity L", self.window.input_info.text())
            self.assertEqual(self.window.data_range.value(), 1)
            np.testing.assert_array_equal(self.window.observation.pixels, pixels)
            displayed = self.window.input_view.scene().items()[0].pixmap().toImage()
            self.assertEqual(displayed.pixelColor(0, 8).red(), 128)
            self.window.set_observation(read_image(path, "normalize_integer"))
            self.assertEqual(self.window.high.value(), 1)
            np.testing.assert_array_equal(self.window.observation.pixels, pixels / 255)

    def test_comparison_baseline_slots_and_clear_confirmation(self):
        self.window.load_sample()
        self.window.compare_methods()
        self.wait_for_worker()
        self.assertEqual(
            [r.config.method for r in self.window.slot_records], ["IF", "TRIF", "RL"]
        )
        self.window.select_slot(2)
        record = self.window.record
        self.assertIsNotNone(self.window.other_view.array)
        self.assertIsNotNone(self.window.third_view.array)
        self.assertEqual(
            self.window.baseline.text(),
            f"Degraded input: MSE {record.input_quality['mse']:.5g}; PSNR {record.input_quality['psnr']:.3f} dB; SSIM {record.input_quality['ssim']:.5f}",
        )
        self.assertFalse(hasattr(self.window, "quality_info"))
        self.assertAlmostEqual(record.input_quality["psnr"], 31.300618881, places=7)
        self.assertLess(record.psnr_change, 0)
        with patch(
            "PySide6.QtWidgets.QMessageBox.question",
            return_value=QMessageBox.StandardButton.No,
        ):
            self.window.request_clear_history()
        self.assertEqual(len(self.window.history), 3)
        self.window.load_bars()
        self.window.run_restoration()
        self.wait_for_worker()
        self.assertIsNone(self.window.slot_records[1])
        self.assertIsNone(self.window.other_view.array)
        with patch(
            "PySide6.QtWidgets.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            self.window.request_clear_history()
        self.assertEqual(self.window.history, [])
        self.assertFalse(self.window.export_button.isEnabled())

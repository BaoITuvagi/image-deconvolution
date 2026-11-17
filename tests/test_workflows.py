"""Maintained image-I/O, run-snapshot, and desktop workflow regression tests."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gc
import tempfile
import time
import unittest
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt, QThread
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from image_deconvolution.core.algorithms import otf, rl
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
            time.sleep(0.002)  # Yield the GIL so the Python worker can make progress.
        self.assertIsNone(self.window.worker, "Worker failed to finish")

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
        self.assertIn("earlier configuration", self.window.result_info.text())
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
        self.assertGreater(self.window.inspector.iteration.count(), 0)
        self.window.other_run.setCurrentIndex(1)
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
        self.assertEqual(self.window.table.currentRow(), 2)
        self.window.table.setFocus()
        QTest.keyClick(self.window.table, Qt.Key.Key_Up)
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
        self.assertIn("earlier configuration", self.window.result_info.text())
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
                    for label in (
                        self.window.truth_info,
                        self.window.input_info,
                        self.window.simulation_note,
                    ):
                        self.assertGreaterEqual(
                            label.height(), label.heightForWidth(label.width())
                        )
                    for preceding, following in (
                        (self.window.clear_truth_button, self.window.photons),
                        (self.window.photons, self.window.seed),
                        (self.window.simulate_button, self.window.simulation_note),
                        (self.window.simulation_note, self.window.image_information),
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
                self.window.simulation_note,
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
        self.assertIn("cells / TRIF", self.window.reference_panel.detail.text())
        np.testing.assert_array_equal(
            self.window.reference_panel.python_view.array,
            self.window.reference_panel.rows[1]["record"].output,
        )
        self.assertTrue(self.window.reference_panel.table.isColumnHidden(3))
        self.window.reference_panel.details.click()
        self.assertFalse(self.window.reference_panel.table.isColumnHidden(3))
        self.assertTrue(self.window.reference_panel.note.isVisible())

    def test_contextual_controls_numeric_steps_and_saved_presets(self):
        self.assertFalse(self.window.run_button.isEnabled())
        self.assertFalse(self.window.simulate_button.isEnabled())
        self.assertFalse(self.window.clear_history_button.isEnabled())
        self.assertFalse(self.window.other_run.isEnabled())
        self.assertFalse(self.window.third_run.isEnabled())
        for index, (name, iterations) in enumerate(
            [("cells", 5), ("bars", 100), ("fibers", 5), ("bars", 5)]
        ):
            self.window.saved_example.setCurrentIndex(index)
            self.window.sample_button.click()
            self.assertIn(name, self.window.observation.source)
            self.assertEqual(self.window.iterations.value(), iterations)
            self.assertEqual(self.window.control_pages.currentIndex(), 0)
        self.assertTrue(self.window.run_button.isEnabled())
        self.assertFalse(self.window.sigma.isEnabled())
        self.assertFalse(self.window.threshold.isEnabled())
        self.assertFalse(self.window.simulate_button.isEnabled())
        self.window.method.setCurrentText("IF")
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
        self.assertIn("Gaussian sigma=", self.window.psf_info.text())
        self.window.psf_mode.setCurrentText("Imported PSF")
        self.assertIn("psf_s2.tif", self.window.psf_info.text())

    def test_preserved_integer_display_does_not_change_pixels_or_L(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "ramp.png"
            pixels = np.arange(256, dtype=np.uint8).reshape(16, 16)
            Image.fromarray(pixels).save(path)
            self.window.set_observation(read_image(path))
            self.assertEqual(self.window.high.value(), 255)
            self.assertEqual(self.window.data_range.value(), 1)
            np.testing.assert_array_equal(self.window.observation.pixels, pixels)
            displayed = self.window.input_view.scene().items()[0].pixmap().toImage()
            self.assertEqual(displayed.pixelColor(0, 8).red(), 128)
            self.window.set_observation(read_image(path, "normalize_integer"))
            self.assertEqual(self.window.high.value(), 1)
            np.testing.assert_array_equal(self.window.observation.pixels, pixels / 255)

    def test_comparison_baseline_filtered_choices_and_clear_confirmation(self):
        self.window.load_sample()
        self.window.compare_methods()
        self.wait_for_worker()
        record = self.window.record
        self.assertIsNotNone(self.window.other_view.array)
        self.assertIsNotNone(self.window.third_view.array)
        methods = {
            record.config.method,
            self.window.other_run.currentData().config.method,
            self.window.third_run.currentData().config.method,
        }
        self.assertEqual(methods, {"IF", "TRIF", "RL"})
        target = self.window.third_run.currentData()
        index = next(
            i
            for i in range(self.window.other_run.count())
            if self.window.other_run.itemData(i) is target
        )
        self.window.other_run.setCurrentIndex(index)
        self.assertIsNot(
            self.window.other_run.currentData(), self.window.third_run.currentData()
        )
        self.assertIn("Degraded input:", self.window.quality_info.text())
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
        self.assertEqual(self.window.other_run.count(), 1)
        self.assertIsNone(self.window.other_view.array)
        with patch(
            "PySide6.QtWidgets.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            self.window.request_clear_history()
        self.assertEqual(self.window.history, [])
        self.assertFalse(self.window.export_button.isEnabled())

"""Repeatable acceptance checks for source and frozen desktop distributions.

This is a developer CLI path, not an interactive restoration feature. It writes
results only into a newly created directory and exercises the actual Qt window.
"""

import json
import logging
import socket
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QHelpEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QToolTip

from image_deconvolution.core.algorithms import otf, rl
from image_deconvolution.experiments.export import environment_versions
from image_deconvolution.experiments.inspection import quality
from image_deconvolution.io.images import prepare_psf, read_image, sample_path
from image_deconvolution.ui.main_window import MainWindow

logger = logging.getLogger(__name__)


def verify_desktop(destination):
    """Exercise window actions, workers, file round trips and bundled references."""
    root = Path(destination).resolve()
    root.mkdir(parents=False, exist_ok=False)
    checks, errors, network_attempts = [], [], []
    app = QApplication.instance() or QApplication([sys.argv[0]])
    window = MainWindow()
    window.error = lambda message: errors.append(str(message))
    window.show()

    def pump(milliseconds=50):
        deadline = time.monotonic() + milliseconds / 1000
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.002)

    def wait():
        deadline = time.monotonic() + 60
        while (window.worker is not None or window.baseline.pending) and time.monotonic() < deadline:
            pump(10)
        if window.worker is not None or window.baseline.pending:
            raise AssertionError("Processing or baseline worker exceeded 60 seconds.")
        if errors:
            raise AssertionError("Unexpected UI errors: " + "; ".join(errors))
        pump()

    def click_view(view):
        viewport = view.viewport()
        point = viewport.rect().center()
        for event_type, buttons in [
            (QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton),
            (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton),
        ]:
            app.sendEvent(
                viewport,
                QMouseEvent(
                    event_type,
                    QPointF(point),
                    QPointF(viewport.mapToGlobal(point)),
                    Qt.MouseButton.LeftButton,
                    buttons,
                    Qt.KeyboardModifier.NoModifier,
                ),
            )

    def deny_network(*args, **kwargs):
        network_attempts.append("connection attempted")
        raise OSError("Network access disabled during desktop acceptance checks.")

    report = {
        "frozen": bool(getattr(sys, "frozen", False)),
        "environment": environment_versions(),
        "checks": checks,
        "working_directory": str(Path.cwd()),
        "passed": False,
    }
    try:
        with (
            patch.object(socket.socket, "connect", deny_network),
            patch.object(socket, "create_connection", deny_network),
        ):
            window.control_pages.setCurrentIndex(1)
            assert not window.run_button.isEnabled()
            window.run_restoration()
            assert errors and "Open a degraded image" in errors[-1]
            errors.clear()
            checks.append("Missing observation rejected by the UI")

            original_font = window.font()
            for points in (9, 12, 16):
                font = window.font()
                font.setPointSize(points)
                window.setFont(font)
                pump()
                header = window.table.horizontalHeader()
                assert header.font().pointSize() == points
                assert not header.font().bold() and not header.highlightSections()
                window.table.fit_columns()
                assert header.length() >= window.table.viewport().width()
                for column, text in enumerate(window.table.HEADINGS):
                    assert window.table.columnWidth(column) >= header.fontMetrics().horizontalAdvance(text) + 24
            window.setFont(original_font)
            window.resize(1280, 820)
            pump()
            checks.append("Readable full-width normal-weight history headers at 9/12/16-point text sizes")

            window.control_pages.setCurrentIndex(0)
            assert window.psf_file.text() == "" and not window.psf_file.isEnabled()
            assert window.saved_example.count() == 6
            for index, name in enumerate(("cells", "bars", "fibers") * 2):
                report_preset = index >= 3
                assert window.saved_example.itemText(index) == (
                    f"{name.title()} — Report — P=200" if report_preset else f"{name.title()} — reference"
                )
                window.saved_example.setCurrentIndex(index)
                window.sample_button.click()
                wait()
                assert name in window.observation.source
                assert window.threshold.value() == (0.3 if report_preset else 0.1)
                assert window.regularization.value() == 0.1
                assert window.iterations.value() == (2 if report_preset else 5)
                assert window.epsilon.value() == 1e-6
                assert window.data_range.value() == 1
                assert window.record is None and window.worker is None
                assert window.control_pages.currentIndex() == 0
                assert window.psf_file.text() == "psf_s2.tif"
                assert window.psf_file.isEnabled() and not window.sigma.isEnabled()
                assert not window.psf_mode.isVisible()
                assert "Set intensity L" not in window.input_info.text()
                values, _ = quality(window.ground_truth, window.observation.pixels, 1)
                assert f"MSE {values['mse']:.5g}" in window.baseline.text()
                assert f"PSNR {values['psnr']:.3f} dB" in window.baseline.text()
                assert window.history == [] and not window.results_visible
            window.data_range.setValue(2)
            wait()
            assert "L=2" in window.baseline.toolTip()
            checks.append(
                "All saved examples show baseline before Run; current L updates without history or regeneration"
            )
            window.saved_example.setCurrentIndex(3)
            window.sample_button.click()
            wait()
            assert window.grab().save(str(root / "report-preset.png"))
            checks.append("Six reference/report presets load without restoration")
            window.saved_example.setCurrentIndex(0)
            window.sample_button.click()
            wait()
            assert window.observation.pixels.shape == (256, 256)
            assert all(not box.isVisible() for box in window.image_boxes[2:])
            assert window.input_view.height() > 0.7 * window.image_layout.geometry().height()
            assert window.grab().save(str(root / "loaded.png"))
            assert window.control_pages.currentIndex() == 0
            selector = window.saved_example.geometry()
            button = window.sample_button.geometry()
            assert abs(selector.center().y() - button.center().y()) <= 1
            assert selector.right() < button.left()
            for section in range(window.control_pages.count()):
                window.control_pages.setCurrentIndex(section)
                pump()
                for control in (
                    window.psf_mode,
                    window.psf_file,
                    window.sigma,
                    window.data_range,
                ):
                    assert control.isVisible() == (section == 1)
                    assert control.parentWidget() is (
                        window.control_pages.widget(section) if section == 1 else window.psf_parking
                    )
                if section == 1:
                    layout = window.control_pages.widget(section).layout()
                    start = layout.getWidgetPosition(window.psf_mode)[0]
                    assert start == (layout.getWidgetPosition(window.seed)[0] + 1 if section == 0 else 0)
                    assert layout.getWidgetPosition(window.data_range)[0] == start + 3
                    assert layout.getWidgetPosition(window.psf_file)[0] == start + 1
                    assert layout.getWidgetPosition(window.sigma)[0] == start + 2
                for control, label in window.control_labels.items():
                    if control.isVisible():
                        assert label.text().endswith(":") and not label.wordWrap()
                        assert abs(label.geometry().center().y() - control.geometry().center().y()) <= 1
            window.control_pages.setCurrentIndex(1)
            window.run_button.click()
            assert not window.controls.isEnabled()
            wait()
            record = window.record
            assert all(box.isVisible() for box in window.image_boxes[2:])
            assert window.active_slot == 0
            assert window.slot_records[0] is record
            assert window.slot_records[1:] == [None, None]
            for box, view in zip(window.image_boxes[2:], window.views[2:]):
                assert box.toolTip() == "" and view.viewport().toolTip() == ""
                point = view.viewport().rect().center()
                QToolTip.hideText()
                app.sendEvent(
                    view.viewport(),
                    QHelpEvent(QEvent.Type.ToolTip, point, view.viewport().mapToGlobal(point)),
                )
                pump()
                assert not QToolTip.isVisible()
            assert window.baseline.heightForWidth(window.baseline.width()) <= window.baseline.fontMetrics().height()
            assert window.image_boxes[2].full_title == "1: RL: iterations=5, epsilon=1e-06"
            assert window.image_boxes[2].layout().count() == 1
            assert window.grab().save(str(root / "first-run.png"))
            H = otf(prepare_psf(record.psf.pixels, record.observation.pixels.shape))
            np.testing.assert_allclose(record.output, rl(record.observation.pixels, H, 5), rtol=0, atol=1e-12)
            checks.append("Saved input -> configured RL worker -> baseline agreement")

            original = record.output.copy()
            window.iterations.setValue(6)
            assert window.table.item(window.table.currentRow(), 8).text() == "Earlier configuration"
            window.low.setValue(-0.1)
            window.high.setValue(1.2)
            window.refresh_views()
            np.testing.assert_array_equal(record.output, original)
            output = root / "saved output.tif"
            with patch.object(QFileDialog, "getSaveFileName", return_value=(str(output), "")):
                window.save_button.click()
            np.testing.assert_array_equal(read_image(output).pixels, original.astype(np.float32))
            checks.append("Historical configuration marker, display independence and UI TIFF save")

            for _ in range(3):
                window.run_button.click()
                wait()
                assert window.active_slot == 0
                assert window.slot_records[0] is window.history[-1]
                assert window.slot_records[1:] == [None, None]
            assert len(window.history) == 4
            click_view(window.other_view)
            assert window.active_slot == 1 and window.record is None
            assert window.input_view.array is not None
            assert not window.save_button.isEnabled()
            window.select_history(0)
            assert window.slot_records[1] is window.history[0]
            window.run_button.click()
            wait()
            assert window.active_slot == 1
            assert window.slot_records[1] is window.history[-1]
            assert window.slot_records[2] is None
            click_view(window.output_view)
            assert window.table.currentRow() == 3
            assert window.record is window.slot_records[0]
            assert window.grab().save(str(root / "selected-slots.png"))
            checks.append("Four repeated runs replace one slot; mouse selection routes history and new results")

            window.control_pages.setCurrentIndex(0)
            with patch.object(
                QFileDialog,
                "getOpenFileName",
                return_value=(str(sample_path("cells_gt.tif")), ""),
            ):
                window.open_source()
            pump()
            assert window.observation is None and window.input_view.array is None
            assert window.baseline.text() == "Degraded input: N/A"
            assert window.psf_mode.isVisible() and window.data_range.isVisible()
            assert (
                window.control_pages.widget(0).layout().getWidgetPosition(window.psf_mode)[0]
                == window.control_pages.widget(0).layout().getWidgetPosition(window.seed)[0] + 1
            )
            assert "Source ground truth:" in window.truth_info.text()
            assert "Click" not in window.truth_info.text()
            assert window.observation_prompt.isVisible()
            np.testing.assert_array_equal(window.truth_view.array, window.source.pixels)
            assert not window.run_button.isEnabled() and not window.compare_button.isEnabled()
            for label in (window.truth_info, window.input_info):
                assert label.height() >= label.heightForWidth(label.width())
            assert window.input_info.geometry().bottom() < window.truth_info.geometry().top()
            assert window.simulate_button.geometry().bottom() < window.image_information.geometry().top()
            checks.append("Inline sidebar fields, retained input panel and readable image information below actions")
            window.psf_mode.setCurrentText("Gaussian")
            assert not window.psf_file.isEnabled() and window.sigma.isEnabled()
            assert window.psf_file.text() == "psf_s2.tif"
            window.sigma.setValue(2)
            window.seed.setValue(17)
            source_pixels = window.source.pixels.copy()
            source_display = window.truth_view.scene().items()[0].pixmap().toImage()
            display_range = window.low.value(), window.high.value()
            window.simulate()
            wait()
            assert (window.low.value(), window.high.value()) == display_range
            np.testing.assert_array_equal(window.ground_truth.pixels, source_pixels)
            assert window.truth_view.scene().items()[0].pixmap().toImage() == source_display
            assert window.psf_mode.isVisible()
            observed = window.observation.pixels.copy()
            assert not window.observation_prompt.isVisible()
            assert window.run_button.isEnabled() and window.compare_button.isEnabled()
            np.testing.assert_array_equal(window.input_view.array, observed)
            assert window.degradation["seed"] == 17
            window.threshold.setValue(0.02)
            window.iterations.setValue(10)
            window.sigma.setValue(3)
            window.data_range.setValue(2)
            np.testing.assert_array_equal(window.observation.pixels, observed)
            window.clear_history()
            window.control_pages.setCurrentIndex(1)
            window.compare_button.click()
            wait()
            assert [r.config.method for r in window.history] == ["IF", "TRIF", "RL"]
            assert all(r.config.data_range == 2 for r in window.history)
            assert all("sigma=3" in r.psf.source for r in window.history)
            assert window.degradation["data_range"] == 1
            assert all(r.quality is not None for r in window.history)
            assert all(r.input_quality is not None for r in window.history)
            assert [run.config.method for run in window.slot_records] == [
                "IF",
                "TRIF",
                "RL",
            ]
            window.select_slot(1)
            assert window.other_view.array is not None
            checks.append(
                "Source-only ground truth -> explicit seeded observation -> three-method fixed-input comparison"
            )

            window.clear_truth()
            window.capture.setChecked(False)
            window.run_button.click()
            wait()
            assert window.record.quality is None
            assert window.record.input_quality is None
            assert window.baseline.text() == "Degraded input: N/A"
            assert window.baseline.toolTip() == ""
            checks.append("No quality measurement without ground truth")

            window.saved_example.setCurrentIndex(1)
            window.sample_button.click()
            assert window.iterations.value() == 5
            window.iterations.setValue(200)
            window.run_button.click()
            wait()
            assert len(window.record.steps) <= 8
            curve = window.record.curve
            assert curve[-1]["psnr"] < max(row["psnr"] for row in curve)
            window.inspector.iteration.setCurrentIndex(window.inspector.iteration.count() - 1)
            window.inspector.phase.setCurrentIndex(4)
            window.inspector.metric_choice.setCurrentText("SSIM")
            window.tabs.setCurrentIndex(1)
            pump(150)
            assert window.grab().save(str(root / "inspector.png"))
            checks.append("Bounded RL phases and late-iteration quality loss on saved bars")

            window.load_research("fibers")
            window.capture.setChecked(True)
            window.run_button.click()
            wait()
            output = window.record.output.copy()
            saved = window.record
            bundle = root / "exported experiment"
            with (
                patch.object(QFileDialog, "getExistingDirectory", return_value=str(root)),
                patch.object(QInputDialog, "getText", return_value=(bundle.name, True)),
            ):
                window.export_button.click()
            wait()
            assert (bundle / "experiment.json").is_file()
            with patch.object(
                QFileDialog,
                "getOpenFileName",
                return_value=(str(bundle / "experiment.json"), ""),
            ):
                window.open_experiment()
            wait()
            reopened = window.record
            assert reopened.saved_from == str((bundle / "experiment.json").resolve())
            assert reopened is not saved
            np.testing.assert_array_equal(reopened.output, output)
            assert reopened.quality == saved.quality
            assert reopened.input_quality == saved.input_quality
            assert reopened.seconds == saved.seconds
            assert reopened.curve == saved.curve
            assert window.tabs.currentIndex() == 0
            assert window.slot_records[window.active_slot] is reopened
            window.tabs.setCurrentIndex(1)
            for iteration, phases in saved.steps.items():
                window.inspector.iteration.setCurrentIndex(window.inspector.iteration.findData(iteration))
                for phase, array in phases.items():
                    window.inspector.phase.setCurrentText(phase)
                    np.testing.assert_array_equal(window.inspector.view.array, array)
            pump()
            assert window.grab().save(str(root / "reopened-inspector.png"))
            window.tabs.setCurrentIndex(0)
            pump()
            assert window.grab().save(str(root / "reopened-compare.png"))
            checks.append(
                "Saved float64 output, recorded measurements/time and all captured RL phases reopen without rerunning"
            )

            metadata = json.loads((bundle / "experiment.json").read_text())
            metadata["schema_version"] = 1
            del metadata["result"]
            legacy = bundle / "legacy.json"
            legacy.write_text(json.dumps(metadata))
            with patch.object(QFileDialog, "getOpenFileName", return_value=(str(legacy), "")):
                window.open_experiment()
            wait()
            np.testing.assert_array_equal(window.record.output, output.astype(np.float32))
            assert window.record.quality == saved.quality
            assert window.record.curve == saved.curve
            assert window.inspector.iteration.count() == 0
            assert "legacy export" in window.inspector.range_info.text()
            checks.append("Legacy TIFF result and recorded curve reopen with an explicit missing-checkpoint message")
            window.run_button.click()
            wait()
            np.testing.assert_allclose(window.record.output, output, rtol=0, atol=1e-12)
            checks.append("UI bundle export -> JSON reopen -> identical rerun within 1e-12")

            for method, capture in [
                ("RL", False),
                ("RL", True),
                ("IF", True),
                ("TRIF", True),
            ]:
                window.method.setCurrentIndex(window.method.findData(method))
                window.capture.setChecked(capture)
                window.clear_truth()
                window.run_button.click()
                wait()
                uncaptured = window.record
                target = root / f"saved-{method}-{capture}"
                with (
                    patch.object(QFileDialog, "getExistingDirectory", return_value=str(root)),
                    patch.object(QInputDialog, "getText", return_value=(target.name, True)),
                ):
                    window.export_button.click()
                wait()
                with patch.object(
                    QFileDialog,
                    "getOpenFileName",
                    return_value=(str(target / "experiment.json"), ""),
                ):
                    window.open_experiment()
                wait()
                np.testing.assert_array_equal(window.record.output, uncaptured.output)
                assert window.record.quality is None
                assert window.inspector.iteration.count() == (
                    len(uncaptured.steps) if method == "RL" and capture else 0
                )
                assert window.inspector.chart_view.rows == []
            checks.append(
                "Captured/uncaptured RL, IF and TRIF reopen without ground truth or fabricated quality curves"
            )

            window.tabs.setCurrentIndex(2)
            reference_table = window.reference_panel.table
            assert reference_table.columnCount() == 6
            for removed in ("details", "note", "detail"):
                assert not hasattr(window.reference_panel, removed)
            window.reference_panel.verify.click()
            wait()
            rows = window.reference_panel.rows
            assert len(rows) == 6 and max(r["max_abs"] for r in rows) < 1e-6
            report["reference_max_abs"] = max(r["max_abs"] for r in rows)
            pump(150)
            assert window.grab().save(str(root / "references.png"))
            checks.append("Six saved reference comparisons and signed difference views")

            reference_font, reference_size = window.font(), window.size()
            for points in (9, 12, 16):
                font = window.font()
                font.setPointSize(points)
                window.setFont(font)
                for width in (1100, 1800):
                    window.resize(width, 1000)
                    reference_table.setRowCount(0)
                    pump()
                    columns = [reference_table.columnWidth(i) for i in range(6)]
                    header = reference_table.horizontalHeader()
                    assert not header.font().bold() and not header.highlightSections()
                    assert header.font().pointSize() == points
                    assert header.length() >= reference_table.viewport().width()
                    for column, text in enumerate(reference_table.HEADINGS):
                        assert columns[column] >= header.fontMetrics().horizontalAdvance(text) + 24
                    window.reference_panel.accept_rows(rows)
                    pump()
                    assert columns == [reference_table.columnWidth(i) for i in range(6)]
                    for index, result in enumerate(rows):
                        reference_table.selectRow(index)
                        np.testing.assert_array_equal(
                            window.reference_panel.python_view.array,
                            result["record"].output,
                        )
                        np.testing.assert_array_equal(
                            window.reference_panel.saved_view.array,
                            result["reference"].pixels,
                        )
                        np.testing.assert_array_equal(window.reference_panel.diff_view.array, result["difference"])
                    assert not header.font().bold()
            window.setFont(reference_font)
            window.resize(reference_size)
            window.reference_panel.verify.click()
            wait()
            assert reference_table.rowCount() == 6
            checks.append(
                "Six reference columns keep readable normal-weight headers before/after comparison at 9/12/16-point fonts, resizing and repeat comparison"
            )

            window.tabs.setCurrentIndex(0)
            window.control_pages.setCurrentIndex(1)
            window.clear_history()
            window.compare_button.click()
            wait()
            window.select_slot(2)
            window.fit_views()
            pump(150)
            heights = [view.viewport().height() for view in window.views[2:]]
            assert max(heights) - min(heights) <= 1
            for run, box in zip(window.slot_records, window.image_boxes[2:]):
                assert box.layout().count() == 1 and not hasattr(box, "parameters")
                assert (
                    box.full_title
                    == f"{window.run_ids[id(run)]}: {run.config.method}: {window.run_parameters(run.config)}"
                )
                assert box.toolTip() == ""
            report["comparison_layout"] = {
                "window": [window.width(), window.height()],
                "baseline_height": window.baseline.height(),
                "viewports": [[view.viewport().width(), view.viewport().height()] for view in window.views],
            }
            checks.append(
                "Single-line baseline and inline result titles reclaim height with three aligned result viewports"
            )
            assert window.grab().save(str(root / "workspace.png"))

            window.saved_example.setCurrentIndex(3)
            window.sample_button.click()
            wait()
            window.compare_button.click()
            wait()
            report_runs = window.history[-3:].copy()
            for run, method, psnr in zip(
                report_runs,
                ["IF", "TRIF", "RL"],
                [38.46252628, 32.70165832, 31.16803614],
                strict=True,
            ):
                assert run.config.method == method
                assert run.config.threshold == 0.3 and run.config.iterations == 2
                assert abs(run.quality["psnr"] - psnr) < 1e-7
            window.select_slot(2)
            pump()
            assert window.grab().save(str(root / "report-comparison.png"))
            for saved in report_runs:
                row = next(i for i, run in enumerate(window.history) if run is saved)
                window.select_history(row)
                window.threshold.setValue(0.04)
                window.iterations.setValue(90)
                target = root / f"report-{saved.config.method}"
                with (
                    patch.object(QFileDialog, "getExistingDirectory", return_value=str(root)),
                    patch.object(QInputDialog, "getText", return_value=(target.name, True)),
                ):
                    window.export_button.click()
                wait()
                with patch.object(
                    QFileDialog,
                    "getOpenFileName",
                    return_value=(str(target / "experiment.json"), ""),
                ):
                    window.open_experiment()
                wait()
                assert window.current_config() == saved.config
                np.testing.assert_array_equal(window.record.output, saved.output)
                assert window.record.quality == saved.quality
                assert window.record.curve == saved.curve
                assert bool(window.record.steps) == (saved.config.method == "RL")
                assert window.active_slot == 2 and window.tabs.currentIndex() == 0
            window.tabs.setCurrentIndex(1)
            assert window.inspector.iteration.count() == 2
            for iteration, phases in saved.steps.items():
                window.inspector.iteration.setCurrentIndex(window.inspector.iteration.findData(iteration))
                for phase, array in phases.items():
                    window.inspector.phase.setCurrentText(phase)
                    np.testing.assert_array_equal(window.inspector.view.array, array)
            pump()
            assert window.grab().save(str(root / "report-reopened-inspector.png"))
            for row in window.reference_panel.rows:
                assert row["record"].config.iterations == 5
                assert row["record"].config.regularization == 0.1
            checks.append(
                "Report three-method results, edited-control export/reopen and captured RL inspection preserve fixed reference settings"
            )

            def fail():
                time.sleep(0.1)
                raise ValueError("Deliberate worker recovery check")

            window.start_operation(fail, window.accept_record)
            assert not window.close()
            deadline = time.monotonic() + 10
            while window.worker is not None and time.monotonic() < deadline:
                pump(10)
            assert window.worker is None and window.controls.isEnabled()
            assert errors == ["Deliberate worker recovery check"]
            errors.clear()
            checks.append("Worker failure recovery and safe close guard")
            assert not network_attempts
            checks.append("Workflow completed with Python socket connections denied")
        assert window.close()
        pump()
        report["passed"] = True
        return_code = 0
    except Exception as exc:
        logger.exception("Desktop workflow verification failed")
        report["error"] = str(exc)
        report["ui_errors"] = errors
        if window.worker is not None:
            window.worker.wait(60000)
            pump()
        window.baseline.stop_pending()
        if window.baseline.worker is not None:
            window.baseline.worker.wait(60000)
            pump()
        window.close()
        return_code = 1
    (root / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return return_code

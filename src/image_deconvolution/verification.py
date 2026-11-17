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
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog

from image_deconvolution.core.algorithms import otf, rl
from image_deconvolution.experiments.export import environment_versions
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
        while window.worker is not None and time.monotonic() < deadline:
            pump(10)
        if window.worker is not None:
            raise AssertionError("Processing worker exceeded 60 seconds.")
        if errors:
            raise AssertionError("Unexpected UI errors: " + "; ".join(errors))
        pump()

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

            window.control_pages.setCurrentIndex(0)
            window.sample_button.click()
            pump()
            assert window.observation.pixels.shape == (256, 256)
            assert window.control_pages.currentIndex() == 0
            selector = window.saved_example.geometry()
            button = window.sample_button.geometry()
            assert selector.center().y() == button.center().y()
            assert selector.right() < button.left()
            window.control_pages.setCurrentIndex(1)
            window.run_button.click()
            assert not window.controls.isEnabled()
            wait()
            record = window.record
            H = otf(prepare_psf(record.psf.pixels, record.observation.pixels.shape))
            np.testing.assert_allclose(
                record.output, rl(record.observation.pixels, H, 5), rtol=0, atol=1e-12
            )
            checks.append("Saved input -> configured RL worker -> baseline agreement")

            original = record.output.copy()
            window.iterations.setValue(6)
            assert "earlier configuration" in window.result_info.text()
            window.low.setValue(-0.1)
            window.high.setValue(1.2)
            window.refresh_views()
            np.testing.assert_array_equal(record.output, original)
            output = root / "saved output.tif"
            with patch.object(
                QFileDialog, "getSaveFileName", return_value=(str(output), "")
            ):
                window.save_button.click()
            np.testing.assert_array_equal(
                read_image(output).pixels, original.astype(np.float32)
            )
            checks.append("Stale-result label, display independence and UI TIFF save")

            window.control_pages.setCurrentIndex(0)
            with patch.object(
                QFileDialog,
                "getOpenFileName",
                return_value=(str(sample_path("cells_gt.tif")), ""),
            ):
                window.open_source()
            pump()
            for label in (window.truth_info, window.input_info, window.simulation_note):
                assert label.height() >= label.heightForWidth(label.width())
            assert (
                window.truth_info.geometry().bottom()
                < window.input_info.geometry().top()
            )
            assert (
                window.input_info.geometry().bottom() < window.photons.geometry().top()
            )
            checks.append(
                "Inline saved-example Load, retained input panel and wrapped source status"
            )
            window.psf_mode.setCurrentText("Gaussian")
            window.sigma.setValue(2)
            window.seed.setValue(17)
            window.simulate()
            wait()
            observed = window.observation.pixels.copy()
            assert window.degradation["seed"] == 17
            window.threshold.setValue(0.02)
            window.iterations.setValue(10)
            np.testing.assert_array_equal(window.observation.pixels, observed)
            window.clear_history()
            window.control_pages.setCurrentIndex(1)
            window.compare_button.click()
            wait()
            assert [r.config.method for r in window.history] == ["IF", "TRIF", "RL"]
            assert all(r.quality is not None for r in window.history)
            assert all(r.input_quality is not None for r in window.history)
            assert {
                window.record.config.method,
                window.other_run.currentData().config.method,
                window.third_run.currentData().config.method,
            } == {"IF", "TRIF", "RL"}
            window.other_run.setCurrentIndex(2)
            assert window.other_view.array is not None
            checks.append(
                "Explicit seeded degradation and three-method fixed-input comparison"
            )

            window.clear_truth()
            window.capture.setChecked(False)
            window.run_button.click()
            wait()
            assert window.record.quality is None
            assert window.record.input_quality is None
            checks.append("No quality measurement without ground truth")

            window.load_bars()
            window.run_button.click()
            wait()
            assert len(window.record.steps) <= 8
            curve = window.record.curve
            assert curve[-1]["psnr"] < max(row["psnr"] for row in curve)
            window.inspector.iteration.setCurrentIndex(
                window.inspector.iteration.count() - 1
            )
            window.inspector.phase.setCurrentIndex(4)
            window.inspector.metric_choice.setCurrentText("SSIM")
            window.tabs.setCurrentIndex(1)
            pump(150)
            assert window.grab().save(str(root / "inspector.png"))
            checks.append(
                "Bounded RL phases and late-iteration quality loss on saved bars"
            )

            window.load_research("fibers")
            window.capture.setChecked(True)
            window.run_button.click()
            wait()
            output = window.record.output.copy()
            bundle = root / "exported experiment"
            with (
                patch.object(
                    QFileDialog, "getExistingDirectory", return_value=str(root)
                ),
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
            window.run_button.click()
            wait()
            np.testing.assert_allclose(window.record.output, output, rtol=0, atol=1e-12)
            checks.append(
                "UI bundle export -> JSON reopen -> identical rerun within 1e-12"
            )

            window.tabs.setCurrentIndex(2)
            window.reference_panel.verify.click()
            wait()
            rows = window.reference_panel.rows
            assert len(rows) == 6 and max(r["max_abs"] for r in rows) < 1e-6
            report["reference_max_abs"] = max(r["max_abs"] for r in rows)
            pump(150)
            assert window.grab().save(str(root / "references.png"))
            checks.append("Six saved reference comparisons and signed difference views")

            window.tabs.setCurrentIndex(0)
            window.control_pages.setCurrentIndex(1)
            window.clear_history()
            window.compare_button.click()
            wait()
            window.other_run.setCurrentIndex(window.other_run.count() - 2)
            window.fit_views()
            pump(150)
            assert window.grab().save(str(root / "workspace.png"))

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
        window.close()
        return_code = 1
    (root / "verification.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return return_code

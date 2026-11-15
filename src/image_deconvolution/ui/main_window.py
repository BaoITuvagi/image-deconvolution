"""Desktop restoration workflow with immutable run inputs and worker execution."""

from pathlib import Path
from dataclasses import replace
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QGridLayout,
    QLabel, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSpinBox,
    QSizePolicy, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from image_deconvolution.core.algorithms import psf_gaussian
from image_deconvolution.experiments.runs import RunConfig, degrade, restore
from image_deconvolution.io.images import ImageData, read_image, sample_path, save_tiff
from image_deconvolution.ui.image_view import ImageView
from image_deconvolution.ui.inspection_panel import InspectionPanel
from image_deconvolution.ui.worker import ProcessingThread


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Deconvolution")
        self.resize(1280, 820)
        self.setMinimumSize(960, 640)
        self.observation = None
        self.imported_psf = None
        self.record = None
        self.worker = None
        self.revision = 0
        self.ground_truth = None
        self.source = None
        self.degradation = None
        self.history = []
        self.record_revisions = {}
        self.controls = QWidget()
        self.form = QFormLayout(self.controls)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.policy = QComboBox()
        self.policy.addItem("Preserve values", "preserve")
        self.policy.addItem("Normalize unsigned integers to [0, 1]", "normalize_integer")
        self.form.addRow("Loading policy", self.policy)
        self.load_button = self.button("Open degraded image…", self.open_observation)
        self.form.addRow(self.load_button)
        self.sample_button = self.button("Load saved cells example", self.load_sample)
        self.form.addRow(self.sample_button)
        self.input_info = QLabel("No image loaded.")
        self.input_info.setWordWrap(True)
        self.form.addRow(self.input_info)
        self.psf_mode = QComboBox()
        self.psf_mode.addItems(["Gaussian", "Imported PSF"])
        self.form.addRow("PSF", self.psf_mode)
        self.sigma = self.number(0.1, 100, 2, 3)
        self.form.addRow("Gaussian sigma (px)", self.sigma)
        self.form.addRow(self.button("Open PSF…", self.open_psf))
        self.psf_info = QLabel("Centered PSFs are normalized to unit sum.")
        self.psf_info.setWordWrap(True)
        self.form.addRow(self.psf_info)
        self.iterations = QSpinBox()
        self.iterations.setRange(1, 1000)
        self.iterations.setValue(20)
        self.form.addRow("RL iterations", self.iterations)
        self.epsilon = self.number(1e-12, 1, 1e-6, 12)
        self.form.addRow("RL epsilon", self.epsilon)
        self.run_button = self.button("Run Richardson-Lucy", self.run_restoration)
        self.form.addRow(self.run_button)
        self.save_button = self.button("Save selected output TIFF…", self.save_output)
        self.save_button.setEnabled(False)
        self.form.addRow(self.save_button)
        self.result_info = QLabel("No restoration run. Timing includes OTF and algorithm only.")
        self.result_info.setWordWrap(True)
        self.form.addRow(self.result_info)
        self.low = self.number(-1e12, 1e12, 0, 6)
        self.high = self.number(-1e12, 1e12, 1, 6)
        self.form.addRow("Display minimum", self.low)
        self.form.addRow("Display maximum", self.high)
        self.form.addRow(self.button("Apply shared display range", self.refresh_views))
        self.form.addRow(self.button("Fit all images", self.fit_views))
        note = QLabel("Display range affects viewing only. Scroll to zoom; drag to pan. Measurements use full numerical images.")
        note.setWordWrap(True)
        self.form.addRow(note)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.controls)
        scroll.setMinimumWidth(300)
        self.tabs = QTabWidget()
        comparison = QWidget()
        images = QGridLayout(comparison)
        self.input_view = ImageView()
        self.output_view = ImageView()
        self.truth_view = ImageView()
        self.other_view = ImageView()
        self.views = [self.input_view, self.truth_view, self.output_view, self.other_view]
        for i, (title, view) in enumerate([
            ("Selected run observation", self.input_view), ("Selected run ground truth", self.truth_view),
            ("Selected restoration", self.output_view), ("Other history run (same input only)", self.other_view),
        ]):
            box = QGroupBox(title)
            layout = QVBoxLayout(box)
            layout.addWidget(view)
            images.addWidget(box, i // 2, i % 2)
        for view in self.views:
            for other in self.views:
                if view is not other:
                    view.navigated.connect(other.synchronize)
        compare_page = QSplitter(Qt.Orientation.Vertical)
        compare_page.addWidget(comparison)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Run / method", "Parameters", "Time (s)", "MSE", "PSNR (dB)", "SSIM", "Configuration"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.cellClicked.connect(self.select_history)
        compare_page.addWidget(self.table)
        compare_page.setSizes([600, 150])
        self.tabs.addTab(compare_page, "Compare & history")
        self.inspector = InspectionPanel()
        self.tabs.addTab(self.inspector, "RL inspector")
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(self.tabs)
        splitter.setSizes([320, 960])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Open an image or load the saved example.")
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        self.menuBar().addMenu("File").addAction(exit_action)
        for control in [self.psf_mode, self.sigma, self.iterations, self.epsilon]:
            signal = control.currentIndexChanged if isinstance(control, QComboBox) else control.valueChanged
            signal.connect(self.mark_changed)
        self.setup_experiments()
        for label in self.controls.findChildren(QLabel):
            label.setWordWrap(True)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        for combo in self.controls.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)

    def setup_experiments(self):
        self.form.insertRow(3, self.button("Open source for simulation…", self.open_source))
        self.form.insertRow(4, self.button("Open optional ground truth…", self.open_truth))
        self.form.insertRow(5, self.button("Clear ground truth", self.clear_truth))
        self.truth_info = QLabel("No ground truth: quality metrics unavailable.")
        self.truth_info.setWordWrap(True)
        self.form.insertRow(6, self.truth_info)
        self.form.insertRow(7, self.button("Load saved bars example", self.load_bars))
        self.photons = self.number(1, 1e8, 200, 2)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.data_range = self.number(1e-9, 1e12, 1, 9)
        self.form.addRow("Poisson photon level", self.photons)
        self.form.addRow("Noise seed", self.seed)
        self.form.addRow("Intensity range L", self.data_range)
        self.form.addRow(self.button("Generate / regenerate degradation", self.simulate))
        self.method = QComboBox()
        self.method.addItems(["RL", "IF", "TRIF"])
        self.threshold = self.number(1e-9, 1, 0.01, 9)
        self.regularization = self.number(1e-12, 1e6, 0.1, 12)
        self.capture = QCheckBox("Capture RL steps")
        self.capture.setToolTip("Store at most eight checkpoints within 64 MiB. History retains at most eight runs or 256 MiB, except a single larger run.")
        self.capture.setChecked(True)
        self.form.addRow("Method", self.method)
        self.form.addRow("IF OTF threshold", self.threshold)
        self.form.addRow("TRIF lambda (L = I)", self.regularization)
        self.form.addRow(self.capture)
        self.compare_button = self.button("Compare IF, TRIF, and RL", self.compare_methods)
        self.form.addRow(self.compare_button)
        self.other_run = QComboBox()
        self.form.addRow("Other output", self.other_run)
        self.other_run.currentIndexChanged.connect(self.refresh_views)
        self.form.addRow(self.button("Clear run history", self.clear_history))
        self.run_button.setText("Run selected method")
        for control in [self.method, self.threshold, self.regularization, self.data_range]:
            signal = control.currentIndexChanged if isinstance(control, QComboBox) else control.valueChanged
            signal.connect(self.mark_changed)
        self.capture.toggled.connect(self.mark_changed)

    def open_source(self):
        path = self.choose_image("Open source / ground truth for simulation")
        if path:
            try:
                source = read_image(path, self.policy.currentData())
                self.set_observation(source)
                self.source = source
                self.ground_truth = source
                self.truth_info.setText("Source loaded. Click Generate / regenerate degradation explicitly.")
            except Exception as exc:
                self.error(exc)

    def open_truth(self):
        path = self.choose_image("Open compatible ground truth")
        if path:
            try:
                self.ground_truth = read_image(path, self.policy.currentData())
                self.truth_info.setText(f"Ground truth: {Path(path).name}. Range L must match its intensity convention.")
                self.mark_changed()
            except Exception as exc:
                self.error(exc)

    def clear_truth(self):
        self.ground_truth = None
        self.truth_info.setText("No ground truth: quality metrics unavailable.")
        self.mark_changed()

    def simulate(self):
        try:
            if self.source is None:
                raise ValueError("Open a source image for simulation first.")
            source, psf = self.source, self.current_psf()
            photons, seed, data_range = self.photons.value(), self.seed.value(), self.data_range.value()
            def operation():
                observation, details = degrade(source, psf, photons, seed, data_range)
                return observation, details, source
            def accept(result):
                observation, details, truth = result
                self.set_observation(observation)
                self.source = self.ground_truth = truth
                self.degradation = details
                self.truth_info.setText(f"Simulated Poisson input; seed {seed}, photons {photons:g}. Ground truth retained.")
            self.start_operation(operation, accept)
        except Exception as exc:
            self.error(exc)

    def current_config(self):
        return RunConfig(self.iterations.value(), self.epsilon.value(), self.method.currentText(),
                         self.threshold.value(), self.regularization.value(), self.data_range.value(), self.capture.isChecked())

    def compare_methods(self):
        try:
            observation, psf = self.observation, self.current_psf()
            config, truth, degradation = self.current_config(), self.ground_truth, self.degradation
            self.start_operation(lambda: [restore(observation, psf, replace(config, method=method), truth, degradation)
                                          for method in ["IF", "TRIF", "RL"]], self.accept_many)
        except Exception as exc:
            self.error(exc)

    def accept_many(self, records):
        for record in records:
            self.accept_record(record)

    def clear_history(self):
        self.history.clear()
        self.record_revisions.clear()
        self.record = None
        self.table.setRowCount(0)
        self.other_run.clear()
        self.save_button.setEnabled(False)
        self.result_info.setText("Run history cleared.")
        self.inspector.set_record(None)
        self.refresh_views()

    def select_history(self, row, _column=0):
        if 0 <= row < len(self.history):
            self.record = self.history[row]
            self.show_record()

    @staticmethod
    def button(text, slot):
        button = QPushButton(text)
        button.clicked.connect(slot)
        return button

    @staticmethod
    def number(minimum, maximum, value, decimals=6):
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setKeyboardTracking(False)
        return spin

    def error(self, message):
        QMessageBox.warning(self, "Cannot complete operation", str(message))

    def choose_image(self, title):
        return QFileDialog.getOpenFileName(self, title, "", "Grayscale images (*.tif *.tiff *.png)")[0]

    def open_observation(self):
        path = self.choose_image("Open degraded image")
        if path:
            try:
                self.set_observation(read_image(path, self.policy.currentData()))
            except Exception as exc:
                self.error(exc)

    def set_observation(self, image):
        self.observation = image
        self.ground_truth = self.source = self.degradation = None
        if hasattr(self, "truth_info"):
            self.truth_info.setText("No ground truth: quality metrics unavailable.")
        self.mark_changed()
        a = image.pixels
        self.input_info.setText(f"{Path(image.source).name}\n{a.shape[1]} × {a.shape[0]}, float64 processing\nValues: {a.min():.6g} to {a.max():.6g}\nPolicy: {image.policy}")
        self.refresh_views()

    def open_psf(self):
        path = self.choose_image("Open centered PSF")
        if path:
            try:
                self.imported_psf = read_image(path)
                self.psf_mode.setCurrentText("Imported PSF")
                self.psf_info.setText(f"{Path(path).name}; center-padded and normalized when run.")
                self.mark_changed()
            except Exception as exc:
                self.error(exc)

    def load_sample(self):
        try:
            self.imported_psf = read_image(sample_path("psf_s2.tif"))
            self.psf_mode.setCurrentText("Imported PSF")
            self.iterations.setValue(5)
            self.psf_info.setText("Saved sigma-2 PSF; normalized when run.")
            self.low.setValue(0)
            self.high.setValue(1)
            self.set_observation(read_image(sample_path("cells_s2_p200_r0_input.tif")))
            self.ground_truth = read_image(sample_path("cells_gt.tif"))
            self.data_range.setValue(1)
            self.truth_info.setText("Saved cells ground truth, normalized research range L=1.")
            self.mark_changed()
            self.refresh_views()
        except Exception as exc:
            self.error(exc)

    def load_bars(self):
        try:
            self.imported_psf = read_image(sample_path("psf_s2.tif"))
            self.psf_mode.setCurrentText("Imported PSF")
            self.set_observation(read_image(sample_path("bars_s2_p200_r0_input.tif")))
            self.ground_truth = read_image(sample_path("bars_gt.tif"))
            self.iterations.setValue(100)
            self.data_range.setValue(1)
            self.capture.setChecked(True)
            self.psf_info.setText("Saved sigma-2 PSF; normalized when run.")
            self.truth_info.setText("Saved bars ground truth, L=1. Inspect early and late iteration quality.")
            self.mark_changed()
            self.refresh_views()
        except Exception as exc:
            self.error(exc)

    def mark_changed(self, *_):
        self.revision += 1
        if hasattr(self, "table"):
            for row in range(self.table.rowCount()):
                self.table.setItem(row, 6, QTableWidgetItem("Earlier configuration"))
        if self.record is not None:
            self.result_info.setText("Selected output belongs to an earlier configuration. Its original inputs and parameters are retained.")

    def current_psf(self):
        if self.observation is None:
            raise ValueError("Open a degraded image first.")
        if self.psf_mode.currentText() == "Gaussian":
            return ImageData(psf_gaussian(self.observation.pixels.shape, self.sigma.value()),
                             f"Gaussian sigma={self.sigma.value():g} px")
        if self.imported_psf is None:
            raise ValueError("Open a PSF or select Gaussian.")
        return self.imported_psf

    def run_restoration(self):
        try:
            observation, psf = self.observation, self.current_psf()
            config, truth, degradation = self.current_config(), self.ground_truth, self.degradation
            self.start_operation(lambda: restore(observation, psf, config, truth, degradation), self.accept_record)
        except Exception as exc:
            self.error(exc)

    def start_operation(self, operation, accept):
        if self.worker is not None:
            return
        self.controls.setEnabled(False)
        self.statusBar().showMessage("Processing… You can inspect the views while the worker runs.")
        self.worker = ProcessingThread(operation, self)
        self.worker.result.connect(accept)
        self.worker.failed.connect(self.error)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()

    def worker_finished(self):
        worker, self.worker = self.worker, None
        worker.deleteLater()
        self.controls.setEnabled(True)
        self.statusBar().showMessage("Ready.")

    def accept_record(self, record):
        self.history.append(record)
        self.record_revisions[id(record)] = self.revision
        def stored_bytes():
            return sum(r.output.nbytes + r.observation.pixels.nbytes + r.psf.pixels.nbytes +
                       sum(a.nbytes for step in r.steps.values() for a in step.values()) for r in self.history)
        while len(self.history) > 8 or (len(self.history) > 1 and stored_bytes() > 256*1024*1024):
            dropped = self.history.pop(0)
            self.record_revisions.pop(id(dropped), None)
        self.table.setRowCount(len(self.history))
        self.other_run.blockSignals(True)
        self.other_run.clear()
        self.other_run.addItem("None", None)
        for row, run in enumerate(self.history):
            config = run.config
            parameters = (f"n={config.iterations}, eps={config.epsilon:g}" if config.method == "RL" else
                          f"tau={config.threshold:g}" if config.method == "IF" else f"lambda={config.regularization:g}")
            values = [f"{row+1}: {config.method}", parameters, f"{run.seconds:.6f}"]
            values += ["N/A" if run.quality is None else f"{run.quality[key]:.7g}" for key in ["mse", "psnr", "ssim"]]
            values += ["Current snapshot" if self.record_revisions[id(run)] == self.revision else "Earlier configuration"]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))
            self.other_run.addItem(f"{row+1}: {config.method} ({parameters})", run)
        self.other_run.blockSignals(False)
        self.table.resizeColumnsToContents()
        self.record = record
        self.show_record()

    def show_record(self):
        record = self.record
        self.save_button.setEnabled(True)
        c = record.config
        parameters = f"iterations={c.iterations}, epsilon={c.epsilon:g}" if c.method == "RL" else f"threshold={c.threshold:g}" if c.method == "IF" else f"lambda={c.regularization:g}, L=I"
        self.result_info.setText(f"{c.method}: {parameters}\nSource: {Path(record.observation.source).name}\n{record.seconds:.6f} s (OTF + algorithm only).\n{record.metric_note} Fixed intensity L={c.data_range:g}.")
        self.inspector.set_record(record)
        self.refresh_views()

    def refresh_views(self):
        if self.high.value() <= self.low.value():
            self.error("Display maximum must exceed display minimum.")
            return
        observation = self.observation if self.record is None else self.record.observation
        truth = self.ground_truth if self.record is None else self.record.ground_truth
        self.input_view.set_image(None if observation is None else observation.pixels,
                                  self.low.value(), self.high.value())
        self.output_view.set_image(None if self.record is None else self.record.output,
                                   self.low.value(), self.high.value())
        self.truth_view.set_image(None if truth is None else truth.pixels, self.low.value(), self.high.value())
        other = self.other_run.currentData() if hasattr(self, "other_run") else None
        compatible = (other is not None and self.record is not None and
                      other.observation.pixels.shape == self.record.observation.pixels.shape and
                      np.array_equal(other.observation.pixels, self.record.observation.pixels))
        self.other_view.set_image(other.output if compatible else None, self.low.value(), self.high.value())

    def fit_views(self):
        for view in self.views:
            view.fit_image()

    def save_output(self):
        if self.record is None:
            self.error("Run a restoration first.")
            return
        path = QFileDialog.getSaveFileName(self, "Save numerical output", "restored.tif", "Float TIFF (*.tif *.tiff)")[0]
        if path:
            try:
                save_tiff(path, self.record.output, overwrite=True)
                self.statusBar().showMessage("Saved float32 TIFF without clipping or display conversion.")
            except Exception as exc:
                self.error(exc)

    def closeEvent(self, event):
        if self.worker is not None:
            self.statusBar().showMessage("Wait for processing to finish before closing.")
            event.ignore()
        else:
            event.accept()

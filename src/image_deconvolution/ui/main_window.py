"""Desktop restoration workflow with immutable run inputs and worker execution."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSpinBox,
    QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from image_deconvolution.core.algorithms import psf_gaussian
from image_deconvolution.experiments.runs import RunConfig, restore
from image_deconvolution.io.images import ImageData, read_image, sample_path, save_tiff
from image_deconvolution.ui.image_view import ImageView
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
        self.controls = QWidget()
        self.form = QFormLayout(self.controls)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
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
        images = QHBoxLayout(comparison)
        self.input_view = ImageView()
        self.output_view = ImageView()
        self.views = [self.input_view, self.output_view]
        for title, view in [("Observed image", self.input_view), ("Selected restoration", self.output_view)]:
            box = QGroupBox(title)
            layout = QVBoxLayout(box)
            layout.addWidget(view)
            images.addWidget(box)
        self.input_view.navigated.connect(self.output_view.synchronize)
        self.output_view.navigated.connect(self.input_view.synchronize)
        self.tabs.addTab(comparison, "Compare")
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
        except Exception as exc:
            self.error(exc)

    def mark_changed(self, *_):
        self.revision += 1
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
            config = RunConfig(self.iterations.value(), self.epsilon.value())
            self.start_operation(lambda: restore(observation, psf, config), self.accept_record)
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
        self.record = record
        self.save_button.setEnabled(True)
        self.result_info.setText(f"RL {record.config.iterations} iterations; epsilon {record.config.epsilon:g}\n{record.seconds:.6f} s (OTF + algorithm; excludes I/O/display).")
        self.refresh_views()

    def refresh_views(self):
        if self.high.value() <= self.low.value():
            self.error("Display maximum must exceed display minimum.")
            return
        self.input_view.set_image(None if self.observation is None else self.observation.pixels,
                                  self.low.value(), self.high.value())
        self.output_view.set_image(None if self.record is None else self.record.output,
                                   self.low.value(), self.high.value())

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

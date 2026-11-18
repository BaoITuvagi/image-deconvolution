"""Compact input quality with coalesced background measurements."""

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import QLabel, QSizePolicy

from image_deconvolution.experiments.inspection import quality
from image_deconvolution.ui.worker import ProcessingThread


class BaselineSummary(QLabel):
    def __init__(self):
        super().__init__("Degraded input: N/A")
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setWordWrap(True)
        policy = self.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.worker = None
        self.requested = None
        self.cache = None
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.start_measurement)

    @staticmethod
    def key(request):
        observation, truth, data_range = request
        return id(observation), id(truth), data_range

    @property
    def pending(self):
        return self.timer.isActive() or self.worker is not None

    def display_values(self, values, context=""):
        self.setText(
            "Degraded input: N/A"
            if values is None
            else f"Degraded input: MSE {values['mse']:.5g}; PSNR {values['psnr']:.3f} dB; SSIM {values['ssim']:.5f}"
        )
        self.setToolTip(context if values is not None else "")

    def show_historical(self, values, context):
        self.requested = None
        self.timer.stop()
        self.display_values(values, context)

    def measure_current(self, observation, truth, data_range):
        if observation is None or truth is None:
            self.requested = self.cache = None
            self.timer.stop()
            self.display_values(None)
            return
        request = observation, truth, data_range
        self.requested = request
        if self.cache is not None and self.key(self.cache[0]) == self.key(request):
            self.display_values(self.cache[1], f"Current inputs; intensity L={data_range:g}.")
            return
        self.cache = None
        self.display_values(None)
        if self.worker is None:
            self.timer.start(0)

    @Slot()
    def start_measurement(self):
        if self.requested is None or self.worker is not None:
            return
        request = self.requested
        observation, truth, data_range = request
        self.running_key = self.key(request)
        self.worker = ProcessingThread(lambda: (request, quality(truth, observation.pixels, data_range)[0]), self)
        self.worker.result.connect(self.receive_values, Qt.ConnectionType.QueuedConnection)
        self.worker.failed.connect(self.measurement_failed)
        self.worker.finished.connect(self.measurement_finished)
        self.worker.start()

    @Slot(object)
    def receive_values(self, result):
        request, values = result
        if self.requested is not None and self.key(request) == self.key(self.requested):
            # Keep input owners alive alongside the identity key to prevent ID reuse.
            self.cache = request, values
            self.display_values(values, f"Current inputs; intensity L={request[2]:g}.")

    @Slot(str)
    def measurement_failed(self, _message):
        if self.requested is not None and self.key(self.requested) == self.running_key:
            self.cache = self.requested, None
            self.display_values(None)

    @Slot()
    def measurement_finished(self):
        worker, self.worker = self.worker, None
        worker.deleteLater()
        if self.requested is not None and self.key(self.requested) != self.running_key:
            self.timer.start(0)

    def stop_pending(self):
        self.timer.stop()
        self.requested = None

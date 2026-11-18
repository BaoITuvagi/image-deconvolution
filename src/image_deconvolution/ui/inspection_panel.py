"""RL intermediate views and exploratory quality curves."""

from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QVBoxLayout, QWidget

from image_deconvolution.ui.curve_plot import CurvePlot
from image_deconvolution.ui.image_view import ImageView


class InspectionPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.record = None
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.iteration = QComboBox()
        self.phase = QComboBox()
        self.metric_choice = QComboBox()
        self.metric_choice.addItems(["PSNR", "SSIM"])
        form.addRow("Update iteration", self.iteration)
        form.addRow("Intermediate image", self.phase)
        form.addRow("Curve", self.metric_choice)
        layout.addLayout(form)
        self.range_info = QLabel("No RL capture selected.")
        layout.addWidget(self.range_info)
        self.view = ImageView()
        layout.addWidget(self.view, 2)
        self.chart_view = CurvePlot()
        self.range_info.setWordWrap(True)
        layout.addWidget(self.chart_view, 1)
        self.iteration.currentIndexChanged.connect(self.show_phase)
        self.phase.currentIndexChanged.connect(self.show_phase)
        self.metric_choice.currentIndexChanged.connect(self.show_curve)

    def set_record(self, record):
        self.record = record
        self.iteration.blockSignals(True)
        self.phase.blockSignals(True)
        self.iteration.clear()
        self.phase.clear()
        for iteration in record.steps if record is not None else {}:
            self.iteration.addItem(str(iteration), iteration)
        if record is not None and record.steps:
            self.phase.addItems(next(iter(record.steps.values())).keys())
        self.iteration.blockSignals(False)
        self.phase.blockSignals(False)
        self.show_curve()
        self.show_phase()

    def show_curve(self, *_):
        metric = self.metric_choice.currentText().lower()
        self.chart_view.set_curve(
            [] if self.record is None else self.record.curve, metric
        )

    def show_phase(self, *_):
        if (
            self.record is None
            or self.iteration.currentData() is None
            or not self.phase.currentText()
        ):
            self.view.set_image(None, 0, 1)
            self.range_info.setText(
                self.record.inspection_note
                if self.record is not None and self.record.inspection_note
                else "Select an RL run with Capture RL steps enabled. Quality curves require suitable ground truth."
            )
            return
        iteration = self.iteration.currentData()
        array = self.record.steps[iteration][self.phase.currentText()]
        low, high = float(array.min()), float(array.max())
        if high <= low:
            high = low + 1
        self.view.set_image(array, low, high)
        values = next(
            (r for r in self.record.curve if r["iteration"] == iteration), None
        )
        detail = (
            ""
            if values is None
            else f"\nQuality of updated x(k+1), iteration {iteration}: PSNR {values['psnr']:.5g} dB; SSIM {values['ssim']:.6g}"
        )
        self.range_info.setText(
            f"Update {iteration}: x(k) is iteration {iteration - 1}; updated x(k+1) is iteration {iteration}.\nDisplay-only phase range [{low:.5g}, {high:.5g}]{detail}"
        )

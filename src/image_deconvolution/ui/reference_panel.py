"""Saved reference comparisons with separate agreement and quality labels."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from image_deconvolution.ui.image_view import ImageView


class ReferencePanel(QWidget):
    verify_requested = Signal()
    load_requested = Signal(str)

    def __init__(self):
        super().__init__()
        self.rows = []
        layout = QVBoxLayout(self)
        note = QLabel(
            "Saved DeconvolutionLab2 references — this action does not run ImageJ. "
            "New Python results use saved float TIFF inputs, sigma-2 PSF normalized to 1, circular boundaries, "
            "RL 5 iterations initialized from y, epsilon 1e-6, or TRIF lambda 0.1 (L=I). "
            "The saved plugin used Academic FFT, no padding/apodization and intact float output. "
            "Agreement (max absolute difference/RMSE) differs from restoration quality against ground truth. IF is not verified against a different plugin variant."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        controls = QHBoxLayout()
        self.example = QComboBox()
        self.example.addItems(["cells", "bars", "fibers"])
        controls.addWidget(self.example)
        load = QPushButton("Load saved example into workspace")
        load.clicked.connect(lambda: self.load_requested.emit(self.example.currentText()))
        controls.addWidget(load)
        self.verify = QPushButton("Compare six saved RL / TRIF references")
        self.verify.clicked.connect(self.verify_requested.emit)
        controls.addWidget(self.verify)
        layout.addLayout(controls)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["Image", "Method", "Max |difference|", "RMSE agreement",
            "New Python PSNR", "New Python SSIM", "Saved ImageJ MSE", "Saved ImageJ PSNR", "Python SSIM of saved DL2"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self.selection_changed)
        layout.addWidget(self.table, 1)
        images = QHBoxLayout()
        self.python_view, self.saved_view, self.diff_view = ImageView(), ImageView(), ImageView()
        for title, view in [("New Python output [0, 1]", self.python_view), ("Saved DL2 output [0, 1]", self.saved_view),
                            ("Signed difference (separate range)", self.diff_view)]:
            box = QGroupBox(title)
            box_layout = QVBoxLayout(box)
            box_layout.addWidget(view)
            images.addWidget(box)
        for view in [self.python_view, self.saved_view, self.diff_view]:
            for other in [self.python_view, self.saved_view, self.diff_view]:
                if view is not other:
                    view.navigated.connect(other.synchronize)
        layout.addLayout(images, 2)
        self.detail = QLabel("No new Python comparison yet. Select a row after comparison.")
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)

    def accept_rows(self, rows):
        self.rows = rows
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for row, result in enumerate(rows):
            q = result["record"].quality
            values = [result["image"], result["method"], result["max_abs"], result["rmse"], q["psnr"], q["ssim"],
                      result["saved_imagej_mse"], result["saved_imagej_psnr"], result["reference_quality_python"]["ssim"]]
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value if isinstance(value, str) else f"{value:.9g}"))
        self.table.resizeColumnsToContents()
        self.table.setCurrentCell(0, 0)
        self.table.selectRow(0)
        self.table.blockSignals(False)
        self.select_row(0)

    def selection_changed(self):
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.select_row(rows[0].row())

    def select_row(self, row, _column=0):
        if not 0 <= row < len(self.rows):
            return
        result = self.rows[row]
        self.python_view.set_image(result["record"].output, 0, 1)
        self.saved_view.set_image(result["reference"].pixels, 0, 1)
        bound = max(result["max_abs"], 1e-12)
        self.diff_view.set_image(result["difference"], -bound, bound)
        baseline = "within" if result["max_abs"] < 1e-6 else "outside"
        self.detail.setText(f"{result['image']} / {result['method']}: max |difference|={result['max_abs']:.9g}, "
                            f"RMSE={result['rmse']:.9g}. Signed display range [{-bound:.5g}, {bound:.5g}]. "
                            f"This result is {baseline} the recorded 1e-6 configuration-specific baseline; it is not a universal tolerance. "
                            "ImageJ MSE/PSNR columns are saved measurements. All SSIM values here are calculated in Python.")

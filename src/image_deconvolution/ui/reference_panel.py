"""Saved reference comparisons with separate agreement and quality labels."""

from PySide6.QtCore import QEvent, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from image_deconvolution.ui.image_view import ImageView


class ReferenceTable(QTableWidget):
    """Readable columns with the same sizing policy before and after comparison."""

    HEADINGS = (
        "Image",
        "Method",
        "Max |difference|",
        "Python PSNR (dB)",
        "Python SSIM",
        "ImageJ PSNR (saved)",
    )

    def __init__(self):
        super().__init__(0, len(self.HEADINGS))
        self.setHorizontalHeaderLabels(self.HEADINGS)
        header = self.horizontalHeader()
        header.setHighlightSections(False)
        font = QFont()
        font.setBold(False)
        header.setFont(font)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        header.setStretchLastSection(False)
        self.verticalHeader().hide()

    def fit_columns(self):
        metrics = self.horizontalHeader().fontMetrics()
        minimum = [metrics.horizontalAdvance(text) + 24 for text in self.HEADINGS]
        samples = ("fibers", "TRIF") + ("-1.23456789e-07",) * 4
        for column, sample in enumerate(samples):
            minimum[column] = max(minimum[column], metrics.horizontalAdvance(sample) + 16)
        extra = max(0, self.viewport().width() - sum(minimum))
        for column, width in enumerate(minimum):
            share = extra // len(minimum) + (column < extra % len(minimum))
            self.setColumnWidth(column, width + share)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_columns()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            QTimer.singleShot(0, self.fit_columns)


class ReferencePanel(QWidget):
    verify_requested = Signal()

    def __init__(self):
        super().__init__()
        self.rows = []
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("New Python calculations against six saved DeconvolutionLab2 outputs. No ImageJ is executed.")
        )
        self.verify = QPushButton("Compare six saved RL / TRIF references")
        self.verify.clicked.connect(self.verify_requested.emit)
        layout.addWidget(self.verify)
        self.table = ReferenceTable()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self.selection_changed)
        layout.addWidget(self.table, 1)
        images = QHBoxLayout()
        self.python_view, self.saved_view, self.diff_view = (
            ImageView(),
            ImageView(),
            ImageView(),
        )
        for title, view in [
            ("New Python output [0, 1]", self.python_view),
            ("Saved DL2 output [0, 1]", self.saved_view),
            ("Signed difference (separate range)", self.diff_view),
        ]:
            box = QGroupBox(title)
            box_layout = QVBoxLayout(box)
            box_layout.addWidget(view)
            images.addWidget(box)
        for view in [self.python_view, self.saved_view, self.diff_view]:
            for other in [self.python_view, self.saved_view, self.diff_view]:
                if view is not other:
                    view.navigated.connect(other.synchronize)
        layout.addLayout(images, 2)

    def accept_rows(self, rows):
        self.rows = rows
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for row, result in enumerate(rows):
            q = result["record"].quality
            values = [
                result["image"],
                result["method"],
                result["max_abs"],
                q["psnr"],
                q["ssim"],
                result["saved_imagej_psnr"],
            ]
            for column, value in enumerate(values):
                self.table.setItem(
                    row,
                    column,
                    QTableWidgetItem(value if isinstance(value, str) else f"{value:.9g}"),
                )
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

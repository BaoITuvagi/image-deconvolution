"""Consistent history columns, independent of row contents and selection."""

from PySide6.QtCore import QEvent, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QHeaderView, QTableWidget


class HistoryTable(QTableWidget):
    HEADINGS = (
        "Run / method",
        "Input",
        "Parameters",
        "Time (s)",
        "MSE",
        "PSNR (dB)",
        "SSIM",
        "PSNR change (dB)",
        "Configuration",
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
        self.setWordWrap(True)

    def fit_columns(self):
        metrics = self.horizontalHeader().fontMetrics()
        minimum = [metrics.horizontalAdvance(text) + 24 for text in self.HEADINGS]
        for column, sample in [
            (3, "0.0001234"),
            (4, "0.00012345"),
            (5, "-123.45"),
            (6, "-0.12345"),
        ]:
            minimum[column] = max(
                minimum[column], metrics.horizontalAdvance(sample) + 16
            )
        extra = max(0, self.viewport().width() - sum(minimum))
        weights = [1, 3, 4, 1, 1, 1, 1, 1, 3]
        assigned = 0
        for column, width in enumerate(minimum):
            share = (
                extra - assigned
                if column == len(minimum) - 1
                else extra * weights[column] // sum(weights)
            )
            self.setColumnWidth(column, width + share)
            assigned += share
        self.resizeRowsToContents()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_columns()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            QTimer.singleShot(0, self.fit_columns)

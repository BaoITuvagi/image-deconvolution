"""A small Qt Widgets line plot without optional Qt Charts dependencies."""

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


class CurvePlot(QWidget):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.metric_key = "psnr"
        self.setMinimumHeight(230)
        self.setMinimumWidth(320)

    def set_curve(self, rows, metric):
        self.rows, self.metric_key = rows, metric
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.palette().base())
        color = self.palette().text().color()
        painter.setPen(color)
        title = "PSNR (dB), fixed L" if self.metric_key == "psnr" else "SSIM"
        painter.drawText(
            QRectF(0, 4, self.width(), 25),
            Qt.AlignmentFlag.AlignCenter,
            f"Exploratory RL quality — {title}",
        )
        plot = QRectF(76, 40, max(1, self.width() - 98), max(1, self.height() - 92))
        painter.drawRect(plot)
        points = [
            (r["iteration"], r[self.metric_key])
            for r in self.rows
            if math.isfinite(r[self.metric_key])
        ]
        if not points:
            painter.drawText(
                plot,
                Qt.AlignmentFlag.AlignCenter,
                "No finite measurements: suitable ground truth is required; perfect-match PSNR is infinite.",
            )
            painter.end()
            return
        xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
        ymin, ymax = min(y for _, y in points), max(y for _, y in points)
        if xmax == xmin:
            xmax = xmin + 1
        padding = max((ymax - ymin) * 0.08, 1e-4)
        ymin, ymax = ymin - padding, ymax + padding
        for tick in range(5):
            fraction = tick / 4
            y = plot.bottom() - fraction * plot.height()
            painter.setPen(QPen(self.palette().mid().color(), 0.5))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(color)
            painter.drawText(
                QRectF(0, y - 10, 68, 20),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"{ymin + fraction * (ymax - ymin):.5g}",
            )
            x = plot.left() + fraction * plot.width()
            painter.drawText(
                QRectF(x - 28, plot.bottom() + 6, 56, 22),
                Qt.AlignmentFlag.AlignCenter,
                f"{xmin + fraction * (xmax - xmin):.0f}",
            )
        path = QPainterPath()
        for index, (x, y) in enumerate(points):
            p = QPointF(
                plot.left() + (x - xmin) / (xmax - xmin) * plot.width(),
                plot.bottom() - (y - ymin) / (ymax - ymin) * plot.height(),
            )
            path.moveTo(p) if index == 0 else path.lineTo(p)
        painter.setPen(QPen(QColor("#1685b9"), 2))
        painter.drawPath(path)
        painter.setPen(color)
        painter.drawText(
            QRectF(plot.left(), self.height() - 28, plot.width(), 24),
            Qt.AlignmentFlag.AlignCenter,
            "Iteration count (selected measurements)",
        )
        painter.end()

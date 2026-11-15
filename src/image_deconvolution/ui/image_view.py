"""Display-only grayscale conversion and synchronized image coordinates."""

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene


class ImageView(QGraphicsView):
    navigated = Signal(object, object, bool)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.array = None
        self._syncing = False
        self._auto_fit = True
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.horizontalScrollBar().valueChanged.connect(self._notify)
        self.verticalScrollBar().valueChanged.connect(self._notify)

    def set_image(self, array, low, high):
        old_shape = None if self.array is None else self.array.shape
        self.array = array
        self.scene().clear()
        if array is None:
            return
        display = (np.clip((array - low) / (high - low), 0, 1) * 255).astype(np.uint8)
        height, width = display.shape
        image = QImage(display.data, width, height, display.strides[0],
                       QImage.Format.Format_Grayscale8).copy()
        self.scene().addPixmap(QPixmap.fromImage(image))
        self.setSceneRect(0, 0, width, height)
        if old_shape != array.shape:
            self.fit_image()

    def fit_image(self):
        if self.array is not None:
            self._auto_fit = True
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._notify()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.array is not None and self._auto_fit:
            self.fit_image()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._auto_fit = False
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        self._auto_fit = False
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        if 0.02 <= self.transform().m11() * factor <= 100:
            self.scale(factor, factor)
            self._notify()
        event.accept()

    def _notify(self):
        if not self._syncing:
            self.navigated.emit(self.transform(), self.mapToScene(self.viewport().rect().center()), self._auto_fit)

    def synchronize(self, transform, center, auto_fit=False):
        self._syncing = True
        self._auto_fit = auto_fit
        self.setTransform(transform)
        self.centerOn(center)
        self._syncing = False

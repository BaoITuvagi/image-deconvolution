"""A read-only filename field that opens a picker by mouse or keyboard."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLineEdit, QSizePolicy


class FileField(QLineEdit):
    clicked = Signal()

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(80)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.rect().contains(event.position().toPoint())
            and self.isEnabled()
        ):
            self.clicked.emit()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            if not event.isAutoRepeat() and self.isEnabled():
                self.clicked.emit()
            event.accept()
        else:
            super().keyPressEvent(event)

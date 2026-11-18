"""A fixed comparison position with mouse and keyboard selection."""

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import QGroupBox, QVBoxLayout

from image_deconvolution.ui.wrapped_label import WrappedLabel


class ResultSlot(QGroupBox):
    selected = Signal()

    def __init__(self, number, view):
        super().__init__(f"Result {number}")
        self.setObjectName("resultSlot")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(f"Result slot {number}")
        self.setToolTip("Select this slot, then run a method or choose a history row.")
        self.setStyleSheet(
            "QGroupBox#resultSlot { border: 2px solid palette(mid); "
            "border-radius: 4px; margin-top: 0.8em; }"
            "QGroupBox#resultSlot[active=true] { border-color: palette(highlight); }"
            "QGroupBox#resultSlot::title { subcontrol-origin: margin; left: 6px; }"
        )
        layout = QVBoxLayout(self)
        self.parameters = WrappedLabel()
        self.parameters.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.parameters)
        layout.addWidget(view)
        view.viewport().installEventFilter(self)

    def set_active(self, active):
        if self.property("active") == active:
            return
        self.setProperty("active", active)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def eventFilter(self, watched, event):
        if (
            event.type() == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self.selected.emit()
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected.emit()
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.selected.emit()
            event.accept()
        else:
            super().keyPressEvent(event)

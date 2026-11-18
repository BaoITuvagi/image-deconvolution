"""A fixed comparison position with mouse and keyboard selection."""

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtWidgets import QGroupBox, QVBoxLayout


class ResultSlot(QGroupBox):
    selected = Signal()

    def __init__(self, number, view):
        super().__init__(f"Result {number}")
        self.setObjectName("resultSlot")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(f"Result slot {number}")
        self.full_title = self.title()
        self.setStyleSheet(
            "QGroupBox#resultSlot { border: 2px solid palette(mid); "
            "border-radius: 4px; margin-top: 0.8em; }"
            "QGroupBox#resultSlot[active=true] { border-color: palette(highlight); }"
            "QGroupBox#resultSlot::title { subcontrol-origin: margin; left: 6px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(view)
        view.viewport().installEventFilter(self)

    def set_result_title(self, text):
        self.full_title = text
        self.setAccessibleDescription(text)
        self.fit_title()

    def fit_title(self):
        super().setTitle(
            self.fontMetrics().elidedText(
                self.full_title, Qt.TextElideMode.ElideRight, max(0, self.width() - 20)
            )
        )

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        return QSize(0, hint.height())

    def resizeEvent(self, event):
        self.fit_title()
        super().resizeEvent(event)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (
            QEvent.Type.FontChange,
            QEvent.Type.StyleChange,
        ) and hasattr(self, "full_title"):
            self.fit_title()

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

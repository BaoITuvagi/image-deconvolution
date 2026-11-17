"""Status text that keeps its full wrapped height inside scrollable forms."""

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QLabel


class WrappedLabel(QLabel):
    """Prevent form layouts from retaining a previous message's shorter height."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.update_wrapped_height()

    def update_wrapped_height(self):
        height = max(0, self.heightForWidth(max(1, self.width())))
        # QLabel restores this flag on text changes. Manage the height explicitly
        # so QFormLayout cannot reuse a stale height-for-width row calculation.
        policy = self.sizePolicy()
        if policy.hasHeightForWidth():
            policy.setHeightForWidth(False)
            self.setSizePolicy(policy)
        if self.minimumHeight() != height:
            self.setMinimumHeight(height)

    def setText(self, text):
        super().setText(text)
        self.update_wrapped_height()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_wrapped_height()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self.update_wrapped_height()

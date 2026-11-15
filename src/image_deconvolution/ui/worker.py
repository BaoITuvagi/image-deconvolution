"""Run processing in a QThread; deliver results only through signals."""

from PySide6.QtCore import QThread, Signal


class ProcessingThread(QThread):
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation

    def run(self):
        try:
            self.result.emit(self.operation())
        except Exception as exc:
            self.failed.emit(str(exc))

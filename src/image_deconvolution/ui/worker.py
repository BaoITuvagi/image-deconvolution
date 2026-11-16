"""Run processing in a QThread; deliver results only through signals."""

import logging

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


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
            logger.exception("Background processing failed")
            self.failed.emit(str(exc))

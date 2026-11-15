"""Minimal application window for the foundation milestone."""

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget


class MainWindow(QMainWindow):
    """Present the application purpose without unfinished feature controls."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Image Deconvolution")
        self.resize(760, 440)
        self.setMinimumSize(480, 300)

        content = QWidget(self)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)

        title = QLabel("Image Deconvolution")
        font = title.font()
        font.setPointSize(22)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        description = QLabel(
            "A local workspace for studying image restoration with inverse "
            "filtering, Tikhonov regularization, and Richardson-Lucy."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        availability = QLabel(
            "Image loading and restoration controls are not available in this version."
        )
        availability.setWordWrap(True)
        layout.addWidget(availability)
        layout.addStretch()

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)
        self.setCentralWidget(content)

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        self.menuBar().addMenu("File").addAction(exit_action)

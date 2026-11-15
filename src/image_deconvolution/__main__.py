"""Launch with python -m image_deconvolution."""

import sys


def main() -> int:
    """Create the desktop application and run its event loop."""
    from PySide6.QtWidgets import QApplication

    from image_deconvolution.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Image Deconvolution")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

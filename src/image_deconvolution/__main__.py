"""Launch with python -m image_deconvolution."""

import argparse
import sys


def main() -> int:
    """Create the desktop application and run its event loop."""
    parser = argparse.ArgumentParser(description="Offline desktop image deconvolution")
    parser.add_argument(
        "--verify-workflow",
        metavar="NEW_OUTPUT_DIRECTORY",
        help="Run repeatable desktop acceptance checks and write a report in a new folder",
    )
    arguments = parser.parse_args()
    if arguments.verify_workflow:
        from image_deconvolution.verification import verify_desktop

        return verify_desktop(arguments.verify_workflow)
    from PySide6.QtWidgets import QApplication

    from image_deconvolution.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Image Deconvolution")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

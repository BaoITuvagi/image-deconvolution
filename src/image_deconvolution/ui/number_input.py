"""Decimal controls with useful steps at small parameter scales."""

from PySide6.QtCore import QLocale
from PySide6.QtWidgets import QDoubleSpinBox


class NumberInput(QDoubleSpinBox):
    def __init__(self):
        super().__init__()
        self.setLocale(QLocale.c())
        self.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
        self.setKeyboardTracking(False)

    def textFromValue(self, value):
        # Keep ordinary decimal editing and configured precision without padding.
        text = f"{value:.{self.decimals()}f}"
        return (text.rstrip("0").rstrip(".") if self.decimals() else text) or "0"

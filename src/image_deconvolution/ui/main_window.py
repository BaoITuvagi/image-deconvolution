"""Desktop restoration workflow with immutable run inputs and worker execution."""

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL.Image import DecompressionBombError
from PySide6.QtCore import QEvent, QSize, Qt, QTimer, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStyle,
    QStyleOptionComboBox,
    QTableWidgetItem,
    QTabWidget,
    QToolBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from image_deconvolution.core.algorithms import psf_gaussian
from image_deconvolution.experiments.export import (
    MissingInput,
    export_bundle,
    load_bundle,
)
from image_deconvolution.experiments.reference import (
    compare_references,
    research_example,
)
from image_deconvolution.experiments.runs import RunConfig, degrade, restore
from image_deconvolution.io.images import ImageData, read_image, save_tiff
from image_deconvolution.ui.baseline_summary import BaselineSummary
from image_deconvolution.ui.file_field import FileField
from image_deconvolution.ui.history_table import HistoryTable
from image_deconvolution.ui.image_view import ImageView
from image_deconvolution.ui.inspection_panel import InspectionPanel
from image_deconvolution.ui.number_input import NumberInput
from image_deconvolution.ui.reference_panel import ReferencePanel
from image_deconvolution.ui.result_slot import ResultSlot
from image_deconvolution.ui.worker import ProcessingThread
from image_deconvolution.ui.wrapped_label import WrappedLabel

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Deconvolution")
        self.resize(1280, 820)
        self.setMinimumSize(960, 640)
        self.observation = None
        self.imported_psf = None
        self.record = None
        self.worker = None
        self.revision = 0
        self.ground_truth = None
        self.source = None
        self.degradation = None
        self.history = []
        self.slot_records = [None, None, None]
        self.active_slot = 0
        self.view_record = None
        self.results_visible = False
        self.run_ids = {}
        self.next_run_id = 1
        self.record_revisions = {}
        self.controls = QWidget()
        self.form = QFormLayout(self.controls)
        self.form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.policy = QComboBox()
        self.policy.addItem("Preserve values", "preserve")
        self.policy.addItem(
            "Normalize unsigned integers to [0, 1]", "normalize_integer"
        )
        self.form.addRow("Loading policy", self.policy)
        self.load_button = self.button("Open degraded image…", self.open_observation)
        self.form.addRow(self.load_button)
        self.saved_example = QComboBox()
        for title, preset in [
            ("Cells — reference", ("cells", 5)),
            ("Bars — reference", ("bars", 5)),
            ("Fibers — reference", ("fibers", 5)),
        ]:
            self.saved_example.addItem(title, preset)
        self.sample_button = QToolButton()
        self.sample_button.setText("Load")
        self.sample_button.clicked.connect(self.load_selected_example)
        self.sample_button.setToolTip("Load the selected saved example.")
        self.saved_example_row = QWidget()
        example_layout = QHBoxLayout(self.saved_example_row)
        example_layout.setContentsMargins(0, 0, 0, 0)
        example_layout.setSpacing(4)
        example_layout.addWidget(self.saved_example, 1)
        example_layout.addWidget(self.sample_button)
        self.form.addRow("Saved example", self.saved_example_row)
        self.input_info = WrappedLabel("No image loaded.")
        self.input_info.setWordWrap(True)
        self.form.addRow(self.input_info)
        self.psf_mode = QComboBox()
        self.psf_mode.addItems(["Gaussian", "Imported PSF"])
        self.form.addRow("PSF", self.psf_mode)
        self.sigma = self.number(0.1, 100, 2, 3)
        self.form.addRow("Gaussian sigma (px)", self.sigma)
        self.psf_file = FileField()
        self.psf_file.setAccessibleName("Open PSF")
        self.psf_file.clicked.connect(self.open_psf)
        self.form.addRow("Open PSF", self.psf_file)
        self.iterations = QSpinBox()
        self.iterations.setRange(1, 1000)
        self.iterations.setValue(20)
        self.form.addRow("RL iterations", self.iterations)
        self.epsilon = self.number(1e-12, 1, 1e-6, 12)
        self.form.addRow("RL epsilon", self.epsilon)
        self.run_button = self.button("Run Richardson-Lucy", self.run_restoration)
        self.form.addRow(self.run_button)
        self.save_button = self.button("Save selected output TIFF…", self.save_output)
        self.save_button.setEnabled(False)
        self.form.addRow(self.save_button)
        self.low = self.number(-1e12, 1e12, 0, 6)
        self.high = self.number(-1e12, 1e12, 1, 6)
        self.form.addRow("Display minimum", self.low)
        self.form.addRow("Display maximum", self.high)
        self.form.addRow(self.button("Apply shared display range", self.refresh_views))
        self.form.addRow(self.button("Fit all images", self.fit_views))
        note = WrappedLabel(
            "Display range affects viewing only. Scroll to zoom; drag to pan. Measurements use full numerical images."
        )
        note.setWordWrap(True)
        self.form.addRow(note)
        scroll = QScrollArea()
        self.sidebar_scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.controls)
        scroll.setMinimumWidth(300)
        self.tabs = QTabWidget()
        comparison = QWidget()
        images = QGridLayout(comparison)
        images.setContentsMargins(6, 4, 6, 4)
        images.setSpacing(6)
        for column in range(6):
            images.setColumnStretch(column, 1)
        images.setRowStretch(0, 1)
        images.setRowStretch(1, 0)
        self.image_layout = images
        self.input_view = ImageView()
        self.observation_prompt = QLabel(
            "Generate degradation to create an observation."
        )
        self.observation_prompt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.observation_prompt.setWordWrap(True)
        self.input_display = QStackedWidget()
        self.input_display.addWidget(self.input_view)
        self.input_display.addWidget(self.observation_prompt)
        self.output_view = ImageView()
        self.truth_view = ImageView()
        self.other_view = ImageView()
        self.third_view = ImageView()
        self.views = [
            self.input_view,
            self.truth_view,
            self.output_view,
            self.other_view,
            self.third_view,
        ]
        self.image_boxes = []
        for i, (title, view) in enumerate(
            [
                ("Current observation", self.input_view),
                ("Current ground truth", self.truth_view),
                ("Selected restoration", self.output_view),
                ("Additional restoration", self.other_view),
                ("Third restoration", self.third_view),
            ]
        ):
            if i < 2:
                box = QGroupBox(title)
                layout = QVBoxLayout(box)
                layout.setContentsMargins(6, 6, 6, 6)
                layout.addWidget(self.input_display if i == 0 else view)
            else:
                slot = i - 2
                box = ResultSlot(slot + 1, view)
                box.selected.connect(lambda index=slot: self.select_slot(index))
            self.image_boxes.append(box)
            if i < 2:
                images.addWidget(box, 0, i * 3, 1, 3)
            else:
                images.addWidget(box, 1, (i - 2) * 2, 1, 2)
        for view in self.views:
            for other in self.views:
                if view is not other:
                    view.navigated.connect(other.synchronize)
        compare_page = QSplitter(Qt.Orientation.Vertical)
        compare_page.addWidget(comparison)
        self.table = HistoryTable()
        self.table.setSelectionBehavior(HistoryTable.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(HistoryTable.SelectionMode.SingleSelection)
        self.table.setEditTriggers(HistoryTable.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.history_selection_changed)
        compare_page.addWidget(self.table)
        compare_page.setSizes([600, 150])
        workspace = QWidget()
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(4)
        context = QHBoxLayout()
        self.current_button = self.button(
            "View current inputs", self.show_current_inputs
        )
        self.current_button.setEnabled(False)
        self.baseline = BaselineSummary()
        context.addWidget(self.baseline, 1)
        context.addWidget(self.current_button)
        workspace_layout.addLayout(context)
        workspace_layout.addWidget(compare_page)
        self.tabs.addTab(workspace, "Compare && history")
        self.inspector = InspectionPanel()
        self.tabs.addTab(self.inspector, "RL inspector")
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(self.tabs)
        splitter.setSizes([380, 900])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Open an image or load the saved example.")
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        self.menuBar().addMenu("File").addAction(exit_action)
        for control in [self.psf_mode, self.sigma, self.iterations, self.epsilon]:
            signal = (
                control.currentIndexChanged
                if isinstance(control, QComboBox)
                else control.valueChanged
            )
            signal.connect(self.mark_changed)
        self.setup_experiments()
        self.reference_panel = ReferencePanel()
        self.reference_panel.verify_requested.connect(self.verify_references)
        self.tabs.addTab(self.reference_panel, "Saved references")
        self.export_button = self.button(
            "Export selected experiment…", self.export_experiment
        )
        self.export_button.setEnabled(False)
        self.form.addRow(self.export_button)
        self.form.addRow(self.button("Reopen experiment JSON…", self.open_experiment))
        field_labels = {
            item.widget()
            for row in range(self.form.rowCount())
            if (item := self.form.itemAt(row, QFormLayout.ItemRole.LabelRole))
            is not None
        }
        for label in self.controls.findChildren(QLabel):
            label.setWordWrap(label not in field_labels)
            policy = label.sizePolicy()
            policy.setHorizontalPolicy(
                QSizePolicy.Policy.Preferred
                if label in field_labels
                else QSizePolicy.Policy.Ignored
            )
            label.setSizePolicy(policy)
        for combo in self.controls.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(10)
        for combo in self.controls.findChildren(QComboBox):
            combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            combo.setMinimumWidth(80)
        for combo in (self.policy, self.saved_example):
            combo.setToolTip(combo.currentText())
            combo.currentTextChanged.connect(combo.setToolTip)
        self.organize_controls()
        self.update_controls()
        self.refresh_views()

    def organize_controls(self):
        """Keep the complete workflow accessible without a single very long form."""
        groups = [
            (
                "Inputs && simulation",
                [
                    "policy",
                    "saved_example_row",
                    "Open degraded image…",
                    "Open source for simulation…",
                    "Open optional ground truth…",
                    "Clear ground truth",
                    "photons",
                    "seed",
                    "Generate / regenerate degradation",
                    "image_information",
                ],
            ),
            (
                "Restoration",
                [
                    "method",
                    "threshold",
                    "regularization",
                    "iterations",
                    "epsilon",
                    "capture",
                    "Run selected method",
                    "Compare IF, TRIF, and RL",
                    "Save selected output TIFF…",
                ],
            ),
            (
                "Display && history",
                [
                    "low",
                    "high",
                    "Apply shared display range",
                    "Fit all images",
                    "Clear run history",
                    "note",
                ],
            ),
            (
                "Experiment files",
                ["Export selected experiment…", "Reopen experiment JSON…"],
            ),
        ]
        rows = {}
        while self.form.rowCount():
            taken = self.form.takeRow(0)
            widget = taken.fieldItem.widget()
            key = (
                widget.text()
                if isinstance(widget, QPushButton)
                else next(
                    (
                        name
                        for name in [
                            "policy",
                            "saved_example_row",
                            "truth_info",
                            "input_info",
                            "photons",
                            "seed",
                            "psf_mode",
                            "sigma",
                            "psf_file",
                            "method",
                            "threshold",
                            "regularization",
                            "iterations",
                            "epsilon",
                            "data_range",
                            "capture",
                            "low",
                            "high",
                        ]
                        if getattr(self, name) is widget
                    ),
                    "note",
                )
            )
            label = taken.labelItem.widget() if taken.labelItem is not None else None
            rows[key] = (label, widget)
        self.control_pages = QToolBox()
        self.control_labels = {}
        self.image_information = QGroupBox("Image information")
        information_layout = QVBoxLayout(self.image_information)
        for key in ("input_info", "truth_info"):
            _, widget = rows.pop(key)
            widget.setTextFormat(Qt.TextFormat.PlainText)
            information_layout.addWidget(widget)
        rows["image_information"] = (None, self.image_information)

        def add_rows(layout, keys):
            layout.setFieldGrowthPolicy(
                QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
            )
            layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
            for key in keys:
                label, widget = rows.pop(key)
                layout.addRow(label, widget) if label is not None else layout.addRow(
                    widget
                )
                if label is not None:
                    label.setText(label.text().rstrip(":") + ":")
                    self.control_labels[widget] = label

        self.psf_rows = [
            rows.pop(key) for key in ["psf_mode", "psf_file", "sigma", "data_range"]
        ]
        self.psf_parking = QWidget(self.controls)
        self.psf_parking.hide()
        for label, widget in self.psf_rows:
            widget.setParent(self.psf_parking)
            if label is not None:
                label.setParent(self.psf_parking)
                label.setText(label.text().rstrip(":") + ":")
                self.control_labels[widget] = label
        for title, keys in groups:
            page = QWidget()
            layout = QFormLayout(page)
            add_rows(layout, keys)
            self.control_pages.addItem(page, title)
        if rows:
            raise RuntimeError(f"Unassigned workflow controls: {list(rows)}")
        self.form.addRow(self.control_pages)
        self.control_pages.currentChanged.connect(self.place_psf_controls)
        self.place_psf_controls(self.control_pages.currentIndex())

    def place_psf_controls(self, section):
        """Show shared fields for source simulation or restoration as applicable."""
        for label, widget in self.psf_rows:
            layout = widget.parentWidget().layout()
            if isinstance(layout, QFormLayout):
                layout.takeRow(widget)
            widget.setParent(self.psf_parking)
            if label is not None:
                label.setParent(self.psf_parking)
        if section == 1 or (section == 0 and self.source is not None):
            layout = self.control_pages.widget(section).layout()
            row = layout.getWidgetPosition(self.seed)[0] + 1 if section == 0 else 0
            for label, widget in self.psf_rows:
                if label is None:
                    layout.insertRow(row, widget)
                else:
                    layout.insertRow(row, label, widget)
                    label.show()
                widget.show()
                row += 1
        self.update_sidebar_width()

    def update_sidebar_width(self):
        """Reserve room for inline fields and a vertical scrollbar in any section."""
        option = QStyleOptionComboBox()
        self.method.initStyleOption(option)
        metrics = self.method.fontMetrics()
        text_size = QSize(
            max(
                metrics.horizontalAdvance(self.method.itemText(i))
                for i in range(self.method.count())
            ),
            metrics.height(),
        )
        self.method.setMinimumWidth(
            self.method.style()
            .sizeFromContents(
                QStyle.ContentsType.CT_ComboBox, option, text_size, self.method
            )
            .width()
        )
        margins = self.form.contentsMargins()
        minimum = max(
            self.control_pages.widget(i).layout().minimumSize().width()
            for i in range(self.control_pages.count())
        )
        self.sidebar_scroll.setMinimumWidth(
            max(
                300,
                minimum
                + margins.left()
                + margins.right()
                + 2 * self.sidebar_scroll.frameWidth()
                + 2 * self.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent),
            )
        )

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (
            QEvent.Type.FontChange,
            QEvent.Type.StyleChange,
        ) and hasattr(self, "control_pages"):
            QTimer.singleShot(0, self.update_sidebar_width)

    def setup_experiments(self):
        self.form.insertRow(
            3, self.button("Open source for simulation…", self.open_source)
        )
        self.form.insertRow(
            4, self.button("Open optional ground truth…", self.open_truth)
        )
        self.clear_truth_button = self.button("Clear ground truth", self.clear_truth)
        self.form.insertRow(5, self.clear_truth_button)
        self.truth_info = WrappedLabel("No ground truth: quality metrics unavailable.")
        self.truth_info.setWordWrap(True)
        self.form.insertRow(6, self.truth_info)
        self.photons = self.number(1, 1e8, 200, 2)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.data_range = self.number(1e-9, 1e12, 1, 9)
        self.form.addRow("Poisson photon level", self.photons)
        self.form.addRow("Noise seed", self.seed)
        self.form.addRow("Intensity range L", self.data_range)
        self.simulate_button = self.button(
            "Generate / regenerate degradation", self.simulate
        )
        self.form.addRow(self.simulate_button)
        self.method = QComboBox()
        for code, name in [
            ("RL", "Richardson-Lucy"),
            ("IF", "Thresholded inverse filtering"),
            ("TRIF", "Zeroth-order Tikhonov"),
        ]:
            self.method.addItem(f"{code} ({name})", code)
        self.method.currentTextChanged.connect(self.method.setToolTip)
        self.method.setToolTip(self.method.currentText())
        self.threshold = self.number(1e-9, 1, 0.01, 9)
        self.regularization = self.number(1e-12, 1e6, 0.1, 12)
        self.capture = QCheckBox("Capture RL steps")
        self.capture.setToolTip(
            "Store at most eight checkpoints within 64 MiB. History retains at most eight runs or 256 MiB, except a single larger run."
        )
        self.capture.setChecked(True)
        self.form.addRow("Method", self.method)
        self.form.addRow("IF OTF threshold", self.threshold)
        self.form.addRow("TRIF lambda (L = I)", self.regularization)
        self.form.addRow(self.capture)
        self.compare_button = self.button(
            "Compare IF, TRIF, and RL", self.compare_methods
        )
        self.compare_button.setToolTip(
            "Uses each method's retained parameters. Select IF, TRIF, or RL to edit its settings before comparison."
        )
        self.form.addRow(self.compare_button)
        self.clear_history_button = self.button(
            "Clear run history", self.request_clear_history
        )
        self.form.addRow(self.clear_history_button)
        self.run_button.setText("Run selected method")
        for control in [
            self.method,
            self.threshold,
            self.regularization,
            self.data_range,
        ]:
            signal = (
                control.currentIndexChanged
                if isinstance(control, QComboBox)
                else control.valueChanged
            )
            signal.connect(self.mark_changed)
        self.capture.toggled.connect(self.mark_changed)

    def load_selected_example(self):
        name, iterations = self.saved_example.currentData()
        self.load_research(name, iterations)

    def load_research(self, name, iterations=5):
        if self.worker is not None:
            return
        try:
            observation, psf, truth = research_example(name)
            self.imported_psf = psf
            self.psf_mode.setCurrentText("Imported PSF")
            self.set_observation(observation, (0, 1))
            self.ground_truth = truth
            self.method.setCurrentIndex(self.method.findData("RL"))
            self.iterations.setValue(iterations)
            self.epsilon.setValue(1e-6)
            self.regularization.setValue(0.1)
            self.threshold.setValue(0.1)
            self.capture.setChecked(True)
            self.data_range.setValue(1)
            self.low.setValue(0)
            self.high.setValue(1)
            self.truth_info.setText(f"Saved {name} ground truth; L=1.")
            self.mark_changed()
            self.refresh_views()
            self.tabs.setCurrentIndex(0)
        except (OSError, ValueError, DecompressionBombError) as exc:
            self.error(exc)

    def verify_references(self):
        self.start_operation(compare_references, self.reference_panel.accept_rows)

    def export_experiment(self):
        if self.record is None:
            self.error("Select a restoration run first.")
            return
        parent = QFileDialog.getExistingDirectory(
            self, "Choose parent folder for the new experiment"
        )
        if not parent:
            return
        name, accepted = QInputDialog.getText(
            self,
            "New experiment",
            "New folder name (must not exist):",
            text="experiment",
        )
        if accepted:
            reserved = {"CON", "PRN", "AUX", "NUL"} | {
                f"{prefix}{i}" for prefix in ["COM", "LPT"] for i in range(1, 10)
            }
            if (
                not name.strip()
                or name != name.strip()
                or name in {".", ".."}
                or name.endswith(".")
                or name.split(".")[0].upper() in reserved
                or any(c in name for c in '<>:"/\\|?*')
                or any(ord(c) < 32 for c in name)
            ):
                self.error(
                    "Enter a single valid new folder name, without path separators or reserved characters."
                )
                return
            destination = Path(parent) / name
            record = self.record
            self.start_operation(
                lambda: export_bundle(destination, record), self.export_finished
            )

    def export_finished(self, path):
        self.completion_message = f"Experiment saved: {path}"

    def open_experiment(self):
        path = QFileDialog.getOpenFileName(
            self, "Reopen experiment configuration", "", "Experiment JSON (*.json)"
        )[0]
        if not path:
            return
        overrides = {}
        try:
            while True:
                try:
                    images, config, degradation = load_bundle(path, overrides)
                    for name, control in [
                        ("iterations", self.iterations),
                        ("epsilon", self.epsilon),
                        ("threshold", self.threshold),
                        ("regularization", self.regularization),
                        ("data_range", self.data_range),
                    ]:
                        value = getattr(config, name)
                        if not control.minimum() <= value <= control.maximum():
                            raise ValueError(
                                f"The saved {name} exceeds this version's control limits."
                            )
                        if (
                            isinstance(control, QDoubleSpinBox)
                            and round(value, control.decimals()) != value
                        ):
                            raise ValueError(
                                f"The saved {name} requires more precision than these controls support."
                            )
                    break
                except MissingInput as exc:
                    located = QFileDialog.getOpenFileName(
                        self, str(exc), "", "Exact numerical input (*.npy)"
                    )[0]
                    if not located:
                        return
                    overrides[exc.role] = located
            self.set_observation(images["observation"], (0, config.data_range))
            self.imported_psf = images["psf"]
            self.ground_truth = images["ground_truth"]
            self.degradation = degradation
            self.psf_mode.setCurrentText("Imported PSF")
            self.method.setCurrentIndex(self.method.findData(config.method))
            self.iterations.setValue(config.iterations)
            self.epsilon.setValue(config.epsilon)
            self.threshold.setValue(config.threshold)
            self.regularization.setValue(config.regularization)
            self.data_range.setValue(config.data_range)
            self.capture.setChecked(config.inspect)
            # Refuse configurations that the current controls cannot represent.
            if self.current_config() != config:
                raise ValueError(
                    "This configuration exceeds the controls supported by this version."
                )
            self.truth_info.setText(
                "Reopened exact input snapshots; "
                + (
                    "ground truth included."
                    if self.ground_truth is not None
                    else "no ground truth: quality metrics unavailable."
                )
            )
            self.mark_changed()
            self.refresh_views()
            self.tabs.setCurrentIndex(0)
            self.control_pages.setCurrentIndex(1)
        except (OSError, ValueError, TypeError, EOFError) as exc:
            self.error(exc)

    def open_source(self):
        path = self.choose_image("Open source / ground truth for simulation")
        if path:
            try:
                source = read_image(path, self.policy.currentData())
                self.observation = self.degradation = None
                self.source = self.ground_truth = source
                self.mark_changed()
                self.update_input_details(source)
                self.truth_info.setText(f"Source ground truth: {Path(path).name}.")
                self.show_current_inputs()
            except (OSError, ValueError, DecompressionBombError) as exc:
                self.error(exc)

    def open_truth(self):
        path = self.choose_image("Open compatible ground truth")
        if path:
            try:
                self.ground_truth = read_image(path, self.policy.currentData())
                self.truth_info.setText(f"Ground truth: {Path(path).name}.")
                self.mark_changed()
                self.show_current_inputs()
            except (OSError, ValueError, DecompressionBombError) as exc:
                self.error(exc)

    def clear_truth(self):
        self.ground_truth = None
        self.truth_info.setText("No ground truth: quality metrics unavailable.")
        self.mark_changed()
        self.show_current_inputs()

    def simulate(self):
        try:
            if self.source is None:
                raise ValueError("Open a source image for simulation first.")
            source, psf = self.source, self.current_psf(self.source)
            photons, seed, data_range = (
                self.photons.value(),
                self.seed.value(),
                self.data_range.value(),
            )

            def operation():
                observation, details = degrade(source, psf, photons, seed, data_range)
                return observation, details, source

            def accept(result):
                observation, details, truth = result
                self.set_observation(observation, (self.low.value(), self.high.value()))
                self.source = self.ground_truth = truth
                self.degradation = details
                self.truth_info.setText(
                    f"Simulated Poisson input; seed {seed}, photons {photons:g}. Ground truth retained."
                )
                self.update_controls()
                self.refresh_views()

            self.start_operation(operation, accept)
        except ValueError as exc:
            self.error(exc)

    def current_config(self):
        return RunConfig(
            self.iterations.value(),
            self.epsilon.value(),
            self.method.currentData(),
            self.threshold.value(),
            self.regularization.value(),
            self.data_range.value(),
            self.capture.isChecked(),
        )

    def compare_methods(self):
        try:
            observation, psf = self.observation, self.current_psf()
            config, truth, degradation = (
                self.current_config(),
                self.ground_truth,
                self.degradation,
            )
            self.tabs.setCurrentIndex(0)
            slot = self.active_slot
            self.start_operation(
                lambda: [
                    restore(
                        observation,
                        psf,
                        replace(config, method=method),
                        truth,
                        degradation,
                    )
                    for method in ["IF", "TRIF", "RL"]
                ],
                lambda records: self.accept_many(records, slot),
            )
        except ValueError as exc:
            self.error(exc)

    def accept_many(self, records, selected_slot=None):
        self.store_records(records)
        self.slot_records = [
            run if any(run is entry for entry in self.history) else None
            for run in records
        ]
        self.active_slot = self.active_slot if selected_slot is None else selected_slot
        self.view_record = next(run for run in self.slot_records if run is not None)
        self.results_visible = True
        self.select_slot(self.active_slot)

    def clear_history(self):
        self.history.clear()
        self.record_revisions.clear()
        self.run_ids.clear()
        self.table.setRowCount(0)
        self.show_current_inputs()

    def request_clear_history(self):
        if (
            self.history
            and QMessageBox.question(
                self,
                "Clear run history?",
                "Remove all in-memory runs? Export any results you need first. Saved files are kept.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            == QMessageBox.StandardButton.Yes
        ):
            self.clear_history()

    def show_current_inputs(self):
        """Return to current inputs without discarding immutable history."""
        self.record = self.view_record = None
        self.slot_records = [None, None, None]
        self.active_slot = 0
        self.results_visible = False
        self.tabs.setCurrentIndex(0)
        self.show_record()

    def select_slot(self, index):
        self.active_slot = index
        self.record = self.slot_records[index]
        if self.record is not None:
            self.view_record = self.record
        self.show_record()

    def history_selection_changed(self):
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.select_history(rows[0].row())

    def select_history(self, row, _column=0):
        if 0 <= row < len(self.history):
            self.place_record(self.history[row], self.active_slot)

    def place_record(self, record, slot):
        # Keep the observation and ground truth consistent across visible outputs.
        self.slot_records = [
            run if run is None or self.same_context(run, record) else None
            for run in self.slot_records
        ]
        self.slot_records[slot] = record
        self.active_slot = slot
        self.record = self.view_record = record
        self.results_visible = True
        self.show_record()

    @staticmethod
    def button(text, slot):
        button = QPushButton(text)
        button.clicked.connect(slot)
        return button

    @staticmethod
    def number(minimum, maximum, value, decimals=6):
        spin = NumberInput()
        spin.setDecimals(decimals)
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setKeyboardTracking(False)
        return spin

    def error(self, message):
        QMessageBox.warning(self, "Cannot complete operation", str(message))

    def choose_image(self, title):
        return QFileDialog.getOpenFileName(
            self, title, "", "Grayscale images (*.tif *.tiff *.png)"
        )[0]

    def open_observation(self):
        path = self.choose_image("Open degraded image")
        if path:
            try:
                self.set_observation(read_image(path, self.policy.currentData()))
            except (OSError, ValueError, DecompressionBombError) as exc:
                self.error(exc)

    def set_observation(self, image, display_range=None):
        self.observation = image
        self.ground_truth = self.source = self.degradation = None
        if hasattr(self, "truth_info"):
            self.truth_info.setText("No ground truth: quality metrics unavailable.")
        self.mark_changed()
        self.update_input_details(image, display_range)
        self.show_current_inputs()

    def update_input_details(self, image, display_range=None):
        """Describe source or observation pixels without changing their input role."""
        a = image.pixels
        if display_range is None:
            display_range = (
                (0, image.nominal_range)
                if image.nominal_range is not None
                else (
                    float(a.min()),
                    float(a.max()) if a.max() > a.min() else float(a.min()) + 1,
                )
            )
        self.low.setValue(display_range[0])
        self.high.setValue(display_range[1])
        if self.high.value() <= self.low.value():
            self.high.setValue(
                self.low.value() + max(1e-6, abs(self.low.value()) * 1e-6)
            )
        scale_note = (
            f"\nImage scale: [0, {image.nominal_range:g}]"
            if image.nominal_range is not None
            else f"\nDisplay range on load: [{self.low.value():g}, {self.high.value():g}]"
        )
        role = (
            "Source file"
            if self.source is image and self.observation is None
            else "File"
        )
        self.input_info.setText(
            f"{role}: {Path(image.source).name}\nSize: {a.shape[1]} × {a.shape[0]} px\nPixel values: {a.min():.5g} to {a.max():.5g}"
            + scale_note
        )
        self.input_info.setToolTip(
            f"Loading policy: {image.policy}. Processing uses float64 numerical pixels."
        )

    def open_psf(self):
        path = self.choose_image("Open centered PSF")
        if path:
            try:
                self.imported_psf = read_image(path)
                self.psf_mode.setCurrentText("Imported PSF")
                self.mark_changed()
            except (OSError, ValueError, DecompressionBombError) as exc:
                self.error(exc)

    def load_sample(self):
        self.load_research("cells")

    def load_bars(self):
        self.load_research("bars")

    def mark_changed(self, *_):
        self.revision += 1
        if hasattr(self, "table"):
            for row in range(self.table.rowCount()):
                self.table.setItem(
                    row,
                    self.table.columnCount() - 1,
                    QTableWidgetItem("Earlier configuration"),
                )
        if hasattr(self, "baseline"):
            self.update_baseline()
        if self.record is not None:
            self.refresh_views()
        if hasattr(self, "clear_history_button"):
            self.update_controls()

    def update_controls(self):
        if hasattr(self, "psf_rows"):
            section = self.control_pages.currentIndex()
            target = (
                self.control_pages.widget(section)
                if section == 1 or (section == 0 and self.source is not None)
                else self.psf_parking
            )
            if self.psf_mode.parentWidget() is not target:
                self.place_psf_controls(section)
        gaussian = self.psf_mode.currentText() == "Gaussian"
        psf_ready = gaussian or self.imported_psf is not None
        ready = self.observation is not None and psf_ready
        psf_source = self.imported_psf.source if self.imported_psf is not None else ""
        self.psf_file.setText(Path(psf_source).name if psf_source else "")
        self.psf_file.setCursorPosition(0)
        self.psf_file.setToolTip(psf_source)
        method = self.method.currentData()
        for control, enabled in [
            (self.psf_file, not gaussian),
            (self.sigma, gaussian),
            (self.threshold, method == "IF"),
            (self.regularization, method == "TRIF"),
            (self.iterations, method == "RL"),
            (self.epsilon, method == "RL"),
            (self.capture, method == "RL"),
        ]:
            control.setEnabled(enabled)
            label = getattr(self, "control_labels", {}).get(control)
            if label is not None:
                label.setEnabled(enabled)
        for button in [self.run_button, self.compare_button]:
            button.setEnabled(ready)
        self.run_button.setToolTip(
            "Run the selected method on the current inputs."
            if ready
            else "Generate degradation before running restoration."
            if self.source is not None and self.observation is None
            else "Open an observation and choose a Gaussian or imported PSF first."
        )
        self.simulate_button.setEnabled(psf_ready and self.source is not None)
        self.simulate_button.setToolTip(
            "Use the PSF and intensity L above Generate; shared with Restoration."
            if self.source is not None
            else "Open source for simulation first; saved examples are already degraded."
        )
        self.photons.setEnabled(self.source is not None)
        self.seed.setEnabled(self.source is not None)
        self.clear_truth_button.setEnabled(self.ground_truth is not None)
        self.clear_history_button.setEnabled(bool(self.history))
        self.data_range.setToolTip(
            "Declared intensity scale for simulation, PSNR and SSIM: 1 for normalized pixels, 255 for preserved 8-bit pixels. Viewing does not change this scale."
        )
        self.save_button.setEnabled(self.record is not None)
        if hasattr(self, "export_button"):
            self.export_button.setEnabled(self.record is not None)

    def current_psf(self, image=None):
        image = self.observation if image is None else image
        if image is None:
            if self.source is not None:
                raise ValueError("Generate degradation before running restoration.")
            raise ValueError("Open a degraded image first.")
        if self.psf_mode.currentText() == "Gaussian":
            return ImageData(
                psf_gaussian(image.pixels.shape, self.sigma.value()),
                f"Gaussian sigma={self.sigma.value():g} px",
            )
        if self.imported_psf is None:
            raise ValueError("Open a PSF or select Gaussian.")
        return self.imported_psf

    def run_restoration(self):
        try:
            observation, psf = self.observation, self.current_psf()
            config, truth, degradation = (
                self.current_config(),
                self.ground_truth,
                self.degradation,
            )
            self.tabs.setCurrentIndex(0)
            slot = self.active_slot
            self.start_operation(
                lambda: restore(observation, psf, config, truth, degradation),
                lambda record: self.accept_record(record, slot),
            )
        except ValueError as exc:
            self.error(exc)

    def start_operation(self, operation, accept):
        if self.worker is not None:
            return
        self.controls.setEnabled(False)
        self.completion_message = "Ready."
        if hasattr(self, "reference_panel"):
            self.reference_panel.setEnabled(False)
        self.statusBar().showMessage(
            "Processing… You can inspect the views while the worker runs."
        )
        self.worker = ProcessingThread(operation, self)
        self.pending_accept = accept
        self.worker.result.connect(
            self.receive_result, Qt.ConnectionType.QueuedConnection
        )
        self.worker.failed.connect(self.operation_failed)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()

    @Slot(object)
    def receive_result(self, result):
        try:
            self.pending_accept(result)
        except Exception as exc:
            logger.exception("Failed to display processing result")
            self.operation_failed(str(exc))

    @Slot()
    def worker_finished(self):
        worker, self.worker = self.worker, None
        worker.deleteLater()
        self.controls.setEnabled(True)
        self.reference_panel.setEnabled(True)
        self.update_controls()
        self.statusBar().showMessage(self.completion_message)

    @Slot(str)
    def operation_failed(self, message):
        self.completion_message = f"Operation failed: {message}"
        self.error(message)

    def accept_record(self, record, slot=None):
        self.store_records([record])
        self.place_record(record, self.active_slot if slot is None else slot)

    def store_records(self, records):
        for record in records:
            self.history.append(record)
            self.record_revisions[id(record)] = self.revision
            self.run_ids[id(record)] = self.next_run_id
            self.next_run_id += 1

        def stored_bytes():
            return sum(
                r.output.nbytes
                + r.observation.pixels.nbytes
                + r.psf.pixels.nbytes
                + (0 if r.ground_truth is None else r.ground_truth.pixels.nbytes)
                + sum(a.nbytes for step in r.steps.values() for a in step.values())
                for r in self.history
            )

        while len(self.history) > 8 or (
            len(self.history) > 1 and stored_bytes() > 256 * 1024 * 1024
        ):
            dropped = self.history.pop(0)
            self.record_revisions.pop(id(dropped), None)
            self.run_ids.pop(id(dropped), None)
            self.slot_records = [
                None if run is dropped else run for run in self.slot_records
            ]
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.history))
        for row, run in enumerate(self.history):
            config = run.config
            parameters = (
                f"n={config.iterations}, eps={config.epsilon:g}"
                if config.method == "RL"
                else f"tau={config.threshold:g}"
                if config.method == "IF"
                else f"lambda={config.regularization:g}, regularizer=I"
            ) + f", L={config.data_range:g}"
            values = [
                f"{self.run_ids[id(run)]}: {config.method}",
                Path(run.observation.source).name,
                parameters,
                f"{run.seconds:.4g}",
            ]
            values += [
                "N/A" if run.quality is None else f"{run.quality[key]:.5g}"
                for key in ["mse", "psnr", "ssim"]
            ]
            values += ["N/A" if run.psnr_change is None else f"{run.psnr_change:+.3f}"]
            values += [
                "Current snapshot"
                if self.record_revisions[id(run)] == self.revision
                else "Earlier configuration"
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.table.setItem(row, col, item)
        self.table.blockSignals(False)
        self.table.fit_columns()

    def show_record(self):
        record = self.record
        self.table.blockSignals(True)
        self.table.clearSelection()
        self.table.setCurrentCell(-1, -1)
        if record is not None:
            row = next(i for i, run in enumerate(self.history) if run is record)
            self.table.setCurrentCell(row, 0)
            self.table.selectRow(row)
        self.table.blockSignals(False)
        self.save_button.setEnabled(record is not None)
        self.export_button.setEnabled(record is not None)
        self.current_button.setEnabled(self.view_record is not None)
        self.inspector.set_record(record)
        self.update_baseline()
        self.update_controls()
        self.refresh_views()

    @staticmethod
    def same_observation(first, second):
        return np.array_equal(first.observation.pixels, second.observation.pixels)

    @classmethod
    def same_context(cls, first, second):
        a, b = first.ground_truth, second.ground_truth
        return cls.same_observation(first, second) and (
            a is b
            or (a is not None and b is not None and np.array_equal(a.pixels, b.pixels))
        )

    @staticmethod
    def run_parameters(config):
        if config.method == "RL":
            return f"iterations={config.iterations}, epsilon={config.epsilon:g}"
        if config.method == "IF":
            return f"threshold={config.threshold:g}"
        return f"lambda={config.regularization:g}, L=I"

    def update_baseline(self):
        record = self.view_record
        if record is not None:
            self.baseline.show_historical(
                record.input_quality,
                f"Input quality for run {self.run_ids[id(record)]}; intensity L={record.config.data_range:g}.",
            )
        else:
            self.baseline.measure_current(
                self.observation, self.ground_truth, self.data_range.value()
            )

    def refresh_views(self):
        self.update_baseline()
        self.image_layout.setRowStretch(1, 1 if self.results_visible else 0)
        if self.high.value() <= self.low.value():
            self.error("Display maximum must exceed display minimum.")
            return
        context = self.view_record
        observation = self.observation if context is None else context.observation
        truth = self.ground_truth if context is None else context.ground_truth
        self.input_view.set_image(
            None if observation is None else observation.pixels,
            self.low.value(),
            self.high.value(),
        )
        self.input_display.setCurrentWidget(
            self.observation_prompt
            if context is None and self.source is not None and observation is None
            else self.input_view
        )
        self.truth_view.set_image(
            None if truth is None else truth.pixels, self.low.value(), self.high.value()
        )
        self.image_boxes[0].setTitle(
            "Current observation"
            if context is None
            else f"Observation: {Path(observation.source).name}"
        )
        self.image_boxes[1].setTitle(
            "Current ground truth"
            if context is None
            else "Ground truth for displayed results"
        )
        for index, (box, view, run) in enumerate(
            zip(self.image_boxes[2:], self.views[2:], self.slot_records)
        ):
            box.setVisible(self.results_visible)
            box.set_active(index == self.active_slot)
            view.set_image(
                None if run is None else run.output, self.low.value(), self.high.value()
            )
            box.set_result_title(
                f"Result {index + 1}"
                if run is None
                else f"{self.run_ids[id(run)]}: {run.config.method}: {self.run_parameters(run.config)}"
            )

    def fit_views(self):
        for view in self.views + [
            self.inspector.view,
            self.reference_panel.python_view,
            self.reference_panel.saved_view,
            self.reference_panel.diff_view,
        ]:
            view.fit_image()

    def save_output(self):
        if self.record is None:
            self.error("Run a restoration first.")
            return
        path = QFileDialog.getSaveFileName(
            self, "Save numerical output", "restored.tif", "Float TIFF (*.tif *.tiff)"
        )[0]
        if path:
            try:
                save_tiff(path, self.record.output, overwrite=True)
                self.statusBar().showMessage(
                    "Saved float32 TIFF without clipping or display conversion."
                )
            except (OSError, ValueError) as exc:
                self.error(exc)

    def closeEvent(self, event):
        if self.worker is not None or self.baseline.worker is not None:
            self.statusBar().showMessage(
                "Wait for processing to finish before closing."
            )
            event.ignore()
        else:
            self.baseline.stop_pending()
            event.accept()

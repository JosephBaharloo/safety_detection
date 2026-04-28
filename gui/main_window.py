from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PyQt6.QtCore import pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QCloseEvent, QIcon
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from gui.alarm_panel import AlarmPanel
from gui.stream_cell import StreamCell


class MainWindow(QMainWindow):
    """Main application view. Contains no detection or alarm business logic."""

    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    close_requested = pyqtSignal()

    def __init__(self, alarm_sound_path: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Safety Equipment Detection")
        self.resize(1400, 820)

        self._stream_cells: dict[str, StreamCell] = {}
        self._grid_columns: int = 2

        # Menu Bar
        self._create_menu_bar()

        # Status Bar
        self._status_bar = self.statusBar()
        self._status_label = QLabel("Ready")
        self._status_bar.addWidget(self._status_label)

        central: QWidget = QWidget(self)
        root_layout: QVBoxLayout = QVBoxLayout(central)

        controls_layout: QHBoxLayout = QHBoxLayout()
        self._start_button: QPushButton = QPushButton("Start Monitoring")
        self._start_button.setIcon(QIcon())  # Placeholder for icon
        self._stop_button: QPushButton = QPushButton("Stop Monitoring")
        self._stop_button.setIcon(QIcon())  # Placeholder for icon
        controls_layout.addWidget(self._start_button)
        controls_layout.addWidget(self._stop_button)
        controls_layout.addStretch(1)

        self._streams_container: QWidget = QWidget()
        self._streams_grid: QGridLayout = QGridLayout(self._streams_container)

        self._alarm_panel: AlarmPanel = AlarmPanel()
        if alarm_sound_path is not None:
            self._alarm_panel.load_alarm_sound(alarm_sound_path)

        splitter: QSplitter = QSplitter()
        splitter.addWidget(self._streams_container)
        splitter.addWidget(self._alarm_panel)
        splitter.setSizes([980, 360])

        root_layout.addLayout(controls_layout)
        root_layout.addWidget(splitter, stretch=1)
        self.setCentralWidget(central)

        self._start_button.clicked.connect(self.start_requested)
        self._stop_button.clicked.connect(self.stop_requested)

    def _create_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")
        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menu_bar.addMenu("&View")
        settings_action = QAction("&Settings", self)
        settings_action.triggered.connect(self._show_settings)
        view_menu.addAction(settings_action)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def register_stream(self, stream_id: str, display_name: str) -> None:
        if stream_id in self._stream_cells:
            return

        index: int = len(self._stream_cells)
        row: int = index // self._grid_columns
        column: int = index % self._grid_columns

        cell: StreamCell = StreamCell(stream_id=stream_id, display_name=display_name)
        self._stream_cells[stream_id] = cell
        self._streams_grid.addWidget(cell, row, column)

    @pyqtSlot(str, object, object)
    def update_stream_frame(self, stream_id: str, frame: object, detections: object) -> None:
        cell: StreamCell | None = self._stream_cells.get(stream_id)
        if cell is None:
            return
        cell.update_frame(frame, detections)

    @pyqtSlot(str, str)
    def update_stream_status(self, stream_id: str, status: str) -> None:
        cell: StreamCell | None = self._stream_cells.get(stream_id)
        if cell is None:
            return
        cell.set_status(status)

    @pyqtSlot(str, object)
    def show_anomaly(self, stream_id: str, missing_items: Sequence[str]) -> None:
        cell: StreamCell | None = self._stream_cells.get(stream_id)
        if cell is None:
            return
        cell.show_anomaly(missing_items)

    @pyqtSlot(str)
    def clear_anomaly(self, stream_id: str) -> None:
        cell: StreamCell | None = self._stream_cells.get(stream_id)
        if cell is None:
            return
        cell.clear_anomaly()

    @pyqtSlot(bool, str)
    def set_alarm_state(self, active: bool, message: str) -> None:
        self._alarm_panel.set_alarm_state(active, message)

    def set_alarm_sound_path(self, sound_path: Path | str) -> None:
        self._alarm_panel.load_alarm_sound(sound_path)

    @pyqtSlot(str)
    def append_alarm_log(self, message: str) -> None:
        self._alarm_panel.append_event(message)

    def update_status(self, status: str) -> None:
        self._status_label.setText(status)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.close_requested.emit()
        super().closeEvent(event)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About Safety Equipment Detection",
            "Safety Equipment Detection Application\n\n"
            "Version 1.0\n"
            "Developed with PyQt6 and OpenCV for real-time safety monitoring."
        )

    def _show_settings(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Settings")
        dialog.setModal(True)

        layout = QVBoxLayout(dialog)

        # Simple settings: grid columns
        columns_label = QLabel("Grid Columns:")
        self._columns_edit = QLineEdit(str(self._grid_columns))
        layout.addWidget(columns_label)
        layout.addWidget(self._columns_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(lambda: self._apply_settings(dialog))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.exec()

    def _apply_settings(self, dialog: QDialog) -> None:
        try:
            new_columns = int(self._columns_edit.text())
            if new_columns > 0:
                self._grid_columns = new_columns
                # Re-layout streams if needed, but for simplicity, just update
                self._rearrange_streams()
            dialog.accept()
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for columns.")

    def _rearrange_streams(self) -> None:
        # Clear the grid
        for i in reversed(range(self._streams_grid.count())):
            self._streams_grid.itemAt(i).widget().setParent(None)

        # Re-add cells
        for index, cell in enumerate(self._stream_cells.values()):
            row = index // self._grid_columns
            column = index % self._grid_columns
            self._streams_grid.addWidget(cell, row, column)

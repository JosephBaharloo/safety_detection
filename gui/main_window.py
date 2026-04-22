from __future__ import annotations

from collections.abc import Sequence

from PyQt6.QtCore import pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QSplitter,
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

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Safety Equipment Detection")
        self.resize(1400, 820)

        self._stream_cells: dict[str, StreamCell] = {}
        self._grid_columns: int = 2

        central: QWidget = QWidget(self)
        root_layout: QVBoxLayout = QVBoxLayout(central)

        controls_layout: QHBoxLayout = QHBoxLayout()
        self._start_button: QPushButton = QPushButton("Start Monitoring")
        self._stop_button: QPushButton = QPushButton("Stop Monitoring")
        controls_layout.addWidget(self._start_button)
        controls_layout.addWidget(self._stop_button)
        controls_layout.addStretch(1)

        self._streams_container: QWidget = QWidget()
        self._streams_grid: QGridLayout = QGridLayout(self._streams_container)

        self._alarm_panel: AlarmPanel = AlarmPanel()

        splitter: QSplitter = QSplitter()
        splitter.addWidget(self._streams_container)
        splitter.addWidget(self._alarm_panel)
        splitter.setSizes([980, 360])

        root_layout.addLayout(controls_layout)
        root_layout.addWidget(splitter, stretch=1)
        self.setCentralWidget(central)

        self._start_button.clicked.connect(self.start_requested)
        self._stop_button.clicked.connect(self.stop_requested)

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

    @pyqtSlot(str)
    def append_alarm_log(self, message: str) -> None:
        self._alarm_panel.append_event(message)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.close_requested.emit()
        super().closeEvent(event)

from __future__ import annotations

from collections import defaultdict
from threading import RLock
from typing import Any, Callable, DefaultDict

from PyQt6.QtCore import QObject, Qt, pyqtSignal, pyqtSlot

EventHandler = Callable[[Any], None]


class EventBus(QObject):
    """Thread-safe pub/sub event bus with queued main-thread dispatch."""

    _event_signal = pyqtSignal(str, object)

    def __init__(self) -> None:
        super().__init__()
        self._subscribers: DefaultDict[str, list[EventHandler]] = defaultdict(list)
        self._lock: RLock = RLock()
        self._event_signal.connect(self._dispatch, Qt.ConnectionType.QueuedConnection)

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        with self._lock:
            if handler not in self._subscribers[topic]:
                self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: EventHandler) -> None:
        with self._lock:
            handlers: list[EventHandler] = self._subscribers.get(topic, [])
            if handler in handlers:
                handlers.remove(handler)
            if not handlers and topic in self._subscribers:
                del self._subscribers[topic]

    def publish(self, topic: str, payload: Any) -> None:
        self._event_signal.emit(topic, payload)

    @pyqtSlot(str, object)
    def _dispatch(self, topic: str, payload: Any) -> None:
        with self._lock:
            handlers: list[EventHandler] = list(self._subscribers.get(topic, []))

        for handler in handlers:
            handler(payload)

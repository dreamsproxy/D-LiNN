"""Application entry point for the LiSNN U1 planning graph."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .window import PlanningGraphWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("LiSNN Planning Graph")
    window = PlanningGraphWindow()
    window.show()
    return app.exec()

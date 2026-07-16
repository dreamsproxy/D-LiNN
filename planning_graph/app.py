"""Application entry point for the LiSNN U1 planning graph."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .theme import apply_dark_theme
from .window import PlanningGraphWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("LiSNN Planning Graph")
    apply_dark_theme(app)
    window = PlanningGraphWindow()
    window.show()
    return app.exec()

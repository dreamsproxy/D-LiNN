"""Public runtime facade for the planning graph rendering layer.

Qt platform plugins differ in how long borrowed native-menu wrappers remain
valid.  This module installs the renderer while giving the performance toggle
its own Python-owned menu, avoiding transient QMenu wrappers on offscreen and
native-menu platforms.
"""

from __future__ import annotations

from . import performance as _performance

RenderProfile = _performance.RenderProfile
BALANCED_PROFILE = _performance.BALANCED_PROFILE
PERFORMANCE_PROFILE = _performance.PERFORMANCE_PROFILE
active_profile = _performance.active_profile
configure_runtime = _performance.configure_runtime
detail_tier = _performance.detail_tier
grid_step_for_lod = _performance.grid_step_for_lod
performance_mode_enabled = _performance.performance_mode_enabled
set_performance_mode = _performance.set_performance_mode

_INSTALLED = False


def install_runtime_patches() -> None:
    """Install renderer patches and attach a persistent Rendering menu."""

    global _INSTALLED
    if _INSTALLED:
        return

    # The original installer attempted to borrow the existing View menu through
    # QMenuBar.actions(). Some platform plugins invalidate that transient QMenu
    # wrapper. Returning None lets the core installer create the action and
    # status indicator without touching a borrowed menu.
    _performance._find_menu = lambda window, title: None
    _performance.install_runtime_patches()

    from .window import PlanningGraphWindow

    original_window_init = PlanningGraphWindow.__init__

    def window_init(self, *args, **kwargs):
        original_window_init(self, *args, **kwargs)
        self.rendering_menu = self.menuBar().addMenu("Rendering")
        self.rendering_menu.addAction(self.performance_action)

    PlanningGraphWindow.__init__ = window_init
    _INSTALLED = True

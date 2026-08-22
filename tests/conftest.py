# -*- coding: utf-8 -*-
"""
Shared test hygiene.

Several test modules build real Tk widget trees (withdrawn roots) and destroy
them at module teardown. ttkbootstrap keeps process-level state bound to the
first root it themes — a ``Style`` singleton and a ``Publisher`` registry of
every themed widget — so the *next* module's widgets explode with
"application has been destroyed" unless that state is cleared between
modules. This autouse fixture resets it after every test module.
"""

import pytest


@pytest.fixture(autouse=True, scope="module")
def _reset_tk_between_modules():
    yield
    import tkinter as tk
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass
    root = getattr(tk, "_default_root", None)
    if root is not None:
        try:
            root.destroy()
        except Exception:
            pass
    tk._default_root = None
    try:
        from ttkbootstrap.style import Style as _BsStyle
        from ttkbootstrap.publisher import Publisher as _BsPub
        _BsStyle.instance = None
        _BsPub.clear_subscribers()
    except Exception:
        pass

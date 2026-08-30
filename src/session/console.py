"""Live console view of the trading game."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.live import Live
from rich.table import Table

_QUICK_EDIT = 0x0040
_EXTENDED_FLAGS = 0x0080


def _strip(norms) -> str:
    """Render the chronotopic momentum strip, short window first.

    A dot marks a window that has no reading yet (not enough history), which
    is what the culture receives too: silence, not a weak signal.
    """
    if not norms:
        return "-"
    return " ".join("  .  " if v == 0.0 else f"{v:+.2f}" for v in norms)


def _set_quickedit(enabled: bool) -> int | None:
    """Windows only: toggle console QuickEdit mode. Returns the prior mode.

    With QuickEdit on (the default), a stray click into the console window
    starts a text selection that BLOCKS all writes to stdout -- which would
    freeze the rich Live view, and with it the 20 Hz tick loop that renders
    it, until someone presses Escape. On real hardware that is a killed
    session from one misplaced click.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return None  # no real console (redirected/CI)
        new_mode = (mode.value | _QUICK_EDIT) if enabled else (mode.value & ~_QUICK_EDIT)
        if kernel32.SetConsoleMode(handle, new_mode | _EXTENDED_FLAGS):
            return mode.value
    except Exception:
        pass
    return None


class ConsoleView:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._live: Live | None = None
        self._console = Console()
        self._prev_console_mode: int | None = None

    def __enter__(self) -> "ConsoleView":
        if self.enabled:
            self._prev_console_mode = _set_quickedit(False)
            self._live = Live(self._render({}), console=self._console, refresh_per_second=4)
            self._live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._live:
            self._live.__exit__(*exc)
        if self._prev_console_mode is not None:
            try:
                import ctypes
                ctypes.windll.kernel32.SetConsoleMode(
                    ctypes.windll.kernel32.GetStdHandle(-10), self._prev_console_mode)
            except Exception:
                pass

    def _render(self, s: dict) -> Table:
        table = Table(title=s.get("title", "cortical trading game"), show_header=False, min_width=64)
        table.add_column("k", style="dim", width=22)
        table.add_column("v")
        rows = [
            ("phase", f"{s.get('phase', '-')}  ep {s.get('episode', '-')} step {s.get('step', '-')}"),
            ("market", f"{s.get('mid', float('nan')):.2f}"),
            ("momentum strip", _strip(s.get("momentum_norms"))),
            ("position", f"{s.get('position', 0):+d}  unrealized {s.get('unrealized_points', 0.0):+.2f} pts"),
            ("realized", f"${s.get('realized_dollars', 0.0):+.2f}  ({s.get('n_trades', 0)} trades)"),
            ("last action", f"{s.get('action', '-')}   diff {s.get('diff', 0.0):+.3f}"),
            ("spikes buy/sell", f"{s.get('buy_count', 0):.0f} / {s.get('sell_count', 0):.0f}"),
            ("feedback", f"pred {s.get('n_predictable', 0)}  unpred {s.get('n_unpredictable', 0)}"),
        ]
        for key, value in rows:
            table.add_row(key, value)
        return table

    def update(self, state: dict) -> None:
        if self._live:
            self._live.update(self._render(state))
        elif state.get("_print"):
            self._console.print(state.get("_print"))

    def log(self, message: str) -> None:
        if self._live:
            self._console.log(message)
        else:
            # flush: headless runs are usually piped to a file/log collector,
            # where block buffering would hold messages back for minutes.
            print(message, flush=True)

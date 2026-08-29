"""Live console view of the trading game."""

from __future__ import annotations

from rich.console import Console
from rich.live import Live
from rich.table import Table


class ConsoleView:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._live: Live | None = None
        self._console = Console()

    def __enter__(self) -> "ConsoleView":
        if self.enabled:
            self._live = Live(self._render({}), console=self._console, refresh_per_second=4)
            self._live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._live:
            self._live.__exit__(*exc)

    def _render(self, s: dict) -> Table:
        table = Table(title=s.get("title", "cortical trading game"), show_header=False, min_width=64)
        table.add_column("k", style="dim", width=22)
        table.add_column("v")
        rows = [
            ("phase", f"{s.get('phase', '-')}  ep {s.get('episode', '-')} step {s.get('step', '-')}"),
            ("market", f"{s.get('mid', float('nan')):.2f}  (mom {s.get('momentum_norm', 0.0):+.2f})"),
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
            print(message)

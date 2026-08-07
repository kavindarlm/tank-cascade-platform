"""
logging_utils.py
================
Terminal presentation helpers for Module 4.

PURELY OBSERVATIONAL. Nothing in this module touches the mass balance, the
objectives, the optimiser or the exported files - it only formats text so a long
MPC run can be watched live instead of sitting silent for minutes at a time.

Visual hierarchy (three levels, so the eye can find its place in a long scroll):

    ╔══════════════════════════════════════════╗
    ║  DAY 3 / 90                              ║   <- banner()      day boundary
    ╚══════════════════════════════════════════╝
    ── OPTIMISE ────────────────────────────────    <- stage()       one of 7 stages
      ▶ running NSGA-II ...                         <- substep()     work starting
        pareto front      : 118 solutions           <- kv()          aligned detail
      ✓ NSGA-II  (42.7s)                            <- done()        work finished

Everything is gated by cfg.VERBOSE_LOGGING through the thin `Log` wrapper at the
bottom; when logging is off every call is a no-op and no strings are built.
"""

import os
import sys
import time
from contextlib import contextmanager


# The tables and banners below use box-drawing characters. On Windows the
# default console/pipe encoding is cp1252, which cannot encode them and would
# abort the run with UnicodeEncodeError - so switch the streams to UTF-8 (and
# fall back to replacement characters if even that is refused).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


# ======================================================================
# COLOUR - used only when the terminal can actually render it
# ======================================================================

def _supports_colour() -> bool:
    """
    True when ANSI colour is safe to emit.

    Honours the NO_COLOR convention, refuses to colour a redirected/piped
    stream (so `python main.py > run.log` stays clean), and on Windows tries to
    switch the console into virtual-terminal mode.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # 7 = STD_OUTPUT_HANDLE, 0x4 = ENABLE_VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            return False
    return True


_COLOUR = _supports_colour()

_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}


def paint(text, *styles) -> str:
    """Wrap `text` in the given styles, or return it untouched on a plain terminal."""
    if not _COLOUR or not styles:
        return str(text)
    prefix = "".join(_CODES.get(s, "") for s in styles)
    return f"{prefix}{text}{_CODES['reset']}"


# ======================================================================
# NUMBER FORMATTING - big m3 values need separators to stay readable
# ======================================================================

def fmt_m3(value, decimals=0) -> str:
    """Format a cubic-metre volume with thousands separators and a unit."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v:                                  # NaN
        return "     n/a"
    return f"{v:,.{decimals}f} m³"


def fmt_num(value, decimals=2) -> str:
    """Format a plain number with thousands separators."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v:
        return "n/a"
    return f"{v:,.{decimals}f}"


def fmt_pct(value, decimals=1) -> str:
    """Format a fraction-or-percentage that is ALREADY in percent units."""
    try:
        return f"{float(value):.{decimals}f}%"
    except (TypeError, ValueError):
        return str(value)


def fmt_dur(seconds) -> str:
    """Human-readable duration: 0.4s / 42.7s / 3m 12s."""
    s = float(seconds)
    if s < 60:
        return f"{s:.1f}s"
    return f"{int(s // 60)}m {s % 60:04.1f}s"


def fmt_compact(value) -> str:
    """
    Compact fixed-width form for objective values inside the NSGA-II progress
    line, where four numbers of wildly different magnitude must share one row.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v:
        return "     nan"
    a = abs(v)
    if a >= 1e6:
        return f"{v/1e6:>7.2f}M"
    if a >= 1e3:
        return f"{v/1e3:>7.2f}k"
    return f"{v:>8.3f}"


# ======================================================================
# PRIMITIVES  (unconditional - the Log wrapper below does the gating)
# ======================================================================

WIDTH = 74
INDENT = "  "


def banner(title, subtitle=None):
    """Top-level boxed banner. Reserved for MPC day boundaries and the season."""
    inner = WIDTH - 2
    print()
    print(paint("╔" + "═" * inner + "╗", "cyan", "bold"))
    print(paint("║", "cyan", "bold")
          + paint(f" {title}".ljust(inner), "cyan", "bold")
          + paint("║", "cyan", "bold"))
    if subtitle:
        print(paint("║", "cyan", "bold")
              + paint(f" {subtitle}".ljust(inner), "dim")
              + paint("║", "cyan", "bold"))
    print(paint("╚" + "═" * inner + "╝", "cyan", "bold"))


def stage(name, note=None):
    """Second-level rule introducing one of the seven MPC stages."""
    label = f"── {name} "
    tail = "─" * max(0, WIDTH - len(label) - (len(note) + 3 if note else 0))
    line = label + tail
    if note:
        line += f" {note} "
    print(paint(line, "blue", "bold"))


def substep(text):
    """A unit of work that is STARTING. Always printed before the work runs."""
    print(f"{INDENT}{paint('▶', 'cyan')} {text}")


def done(text, seconds=None):
    """A unit of work that FINISHED, with its wall-clock cost."""
    timing = f"  {paint('(' + fmt_dur(seconds) + ')', 'dim')}" if seconds is not None else ""
    print(f"{INDENT}{paint('✓', 'green')} {paint(text, 'green')}{timing}")


def kv(key, value, indent=2, key_width=22):
    """Aligned key/value detail line. The column width keeps a block scannable."""
    pad = INDENT * indent
    print(f"{pad}{paint(str(key).ljust(key_width), 'dim')}: {value}")


def note(text, indent=2):
    """Dim free-text detail with no key column."""
    print(f"{INDENT * indent}{paint(text, 'dim')}")


def warn(text, indent=1):
    """Something the operator should look at, but which is not fatal."""
    print(f"{INDENT * indent}{paint('!', 'yellow', 'bold')} {paint(text, 'yellow')}")


def fail(text, indent=1):
    """Something went wrong for this step (e.g. no feasible plan)."""
    print(f"{INDENT * indent}{paint('✗', 'red', 'bold')} {paint(text, 'red')}")


def table(headers, rows, indent=2, aligns=None):
    """
    Render a small fixed-width table with per-column alignment.

    headers : list[str]
    rows    : list[list]      values are str()-ed as given (pre-format them)
    aligns  : list of '<' or '>' per column; defaults to left, then all right.
    """
    cols = len(headers)
    aligns = aligns or (["<"] + [">"] * (cols - 1))
    body = [[str(c) for c in r] for r in rows]
    widths = [max(len(headers[c]), *(len(r[c]) for r in body)) if body
              else len(headers[c]) for c in range(cols)]

    pad = INDENT * indent
    head = "  ".join(f"{headers[c]:{aligns[c]}{widths[c]}}" for c in range(cols))
    print(pad + paint(head, "bold"))
    print(pad + paint("─" * len(head), "dim"))
    for r in body:
        print(pad + "  ".join(f"{r[c]:{aligns[c]}{widths[c]}}" for c in range(cols)))


# ======================================================================
# IN-PLACE PROGRESS  (for the NSGA-II generation counter)
# ======================================================================

IS_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

_progress_width = 0


def progress(text, indent=2):
    """
    Draw a progress line.

    On a real terminal the line is rewritten in place with a carriage return, so
    the generation counter appears to tick upward rather than filling the
    scrollback. When output is piped to a file each update is a normal line, so
    the log stays readable.
    """
    global _progress_width
    line = INDENT * indent + text
    if IS_TTY:
        pad = " " * max(0, _progress_width - len(line))
        _progress_width = len(line)
        sys.stdout.write("\r" + line + pad)
        sys.stdout.flush()
    else:
        print(line)


def progress_end():
    """Close an in-place progress line so the next print starts on a fresh row."""
    global _progress_width
    if IS_TTY and _progress_width:
        sys.stdout.write("\n")
        sys.stdout.flush()
    _progress_width = 0


def bar(fraction, width=20):
    """A simple block progress bar for a fraction in [0, 1]."""
    f = max(0.0, min(1.0, float(fraction)))
    filled = int(round(f * width))
    return paint("█" * filled, "cyan") + paint("░" * (width - filled), "dim")


# ======================================================================
# TIMING
# ======================================================================

class Timer:
    """Plain stopwatch. `with Timer() as t: ...` then read `t.elapsed`."""

    def __enter__(self):
        self.start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start
        return False


@contextmanager
def timed(label, enabled=True):
    """
    Announce a block BEFORE it runs and report its duration after.

        with timed("running NSGA-II") as t:
            ...
        # -> "▶ running NSGA-II ..."   then   "✓ running NSGA-II  (42.7s)"

    The yielded Timer stays valid after the block, so callers can reuse
    `t.elapsed` in their own summary lines.
    """
    if not enabled:
        with Timer() as t:
            yield t
        return
    substep(f"{label} ...")
    t = Timer()
    t.__enter__()
    try:
        yield t
    finally:
        t.__exit__()
        done(label, t.elapsed)


# ======================================================================
# THE GATED FACADE
# ======================================================================

class Log:
    """
    Config-aware wrapper around the primitives above.

        log = Log(cfg)          # or Log(cfg.VERBOSE_LOGGING)
        log.stage("OBSERVE")
        log.kv("tanks", 14)

    When verbosity is off every method returns immediately, so callers never
    need to guard their logging with `if cfg.VERBOSE_LOGGING:`.
    """

    def __init__(self, cfg_or_flag=True):
        flag = getattr(cfg_or_flag, "VERBOSE_LOGGING", cfg_or_flag)
        self.enabled = bool(flag)

    # -- hierarchy ------------------------------------------------------
    def banner(self, title, subtitle=None):
        if self.enabled:
            banner(title, subtitle)

    def stage(self, name, note_text=None):
        if self.enabled:
            stage(name, note_text)

    def substep(self, text):
        if self.enabled:
            substep(text)

    def done(self, text, seconds=None):
        if self.enabled:
            done(text, seconds)

    def kv(self, key, value, indent=2, key_width=22):
        if self.enabled:
            kv(key, value, indent, key_width)

    def note(self, text, indent=2):
        if self.enabled:
            note(text, indent)

    def warn(self, text, indent=1):
        if self.enabled:
            warn(text, indent)

    def fail(self, text, indent=1):
        if self.enabled:
            fail(text, indent)

    def table(self, headers, rows, indent=2, aligns=None):
        if self.enabled:
            table(headers, rows, indent, aligns)

    def blank(self):
        if self.enabled:
            print()

    def raw(self, text):
        if self.enabled:
            print(text)

    # -- in-place progress ----------------------------------------------
    def progress(self, text, indent=2):
        if self.enabled:
            progress(text, indent)

    def progress_end(self):
        if self.enabled:
            progress_end()

    # -- timing ---------------------------------------------------------
    def timed(self, label):
        return timed(label, enabled=self.enabled)

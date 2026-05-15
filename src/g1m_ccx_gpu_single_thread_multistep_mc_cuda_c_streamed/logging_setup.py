"""logging_setup — shared template; deployed verbatim into every app's
own logging_setup.py file (each app is fully self-contained, so the
content is copied per-app rather than imported from a shared module).

Usage from main.py:

    from .logging_setup import configure_logging
    log = configure_logging(args.log_level, args.log_file, "<APP_TAG>")
    log.info("starting")

Sub-modules can use:

    import logging
    logger = logging.getLogger(__name__)
    logger.debug("per-batch info: ...")
    logger.info("phase boundary: ...")
    logger.warning("...")
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def configure_logging(level: str = "INFO",
                      log_file: Optional[str] = None,
                      app_tag: str = "INDEP",
                      reset: bool = True) -> logging.Logger:
    """Configure root + app loggers.

    Args:
      level:   "DEBUG" | "INFO" | "WARNING" | "ERROR".
      log_file: optional path for a file handler; the parent dir is created.
      app_tag: short tag prefixed to every record (e.g. "INDEP-cpu-single-multi-numba").
      reset:   if True, clear any pre-existing handlers from the root logger so
               re-running main() in the same process doesn't duplicate output.

    Returns the configured logger named `app_tag` (callers may use this directly
    or just rely on per-module `logging.getLogger(__name__)` calls).
    """
    lvl = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    if reset:
        for h in list(root.handlers):
            root.removeHandler(h)
    root.setLevel(lvl)

    fmt = logging.Formatter(
        fmt=f"[{app_tag}] %(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(lvl)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(lvl)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    return logging.getLogger(app_tag)


def checklist_log(n: int, msg: str, level: int = logging.INFO) -> None:
    """Emit a `[CHECKLIST-Cn] msg` line at the given level on the root logger.

    Used by the verification protocol in `context_rouse_verification.txt` to
    parse simulation progress.
    """
    logging.getLogger("checklist").log(level, "[CHECKLIST-C%d] %s", n, msg)

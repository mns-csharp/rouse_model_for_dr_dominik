"""BenchLogging — configures Python logging for both [BENCH-*]/[VALIDATION-*]
capture (into benchmark.log) and [CHECKLIST-*] capture (into checklist.log).
"""

import logging
import os


class _ChecklistFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().startswith("[CHECKLIST-")


class BenchLogging:
    _HANDLER_MARKER = "_rouse_bench_handler"
    _CHECKLIST_MARKER = "_rouse_checklist_handler"
    FORMAT = "%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s"

    @classmethod
    def configure(cls, bench_dir: str) -> str:
        os.makedirs(bench_dir, exist_ok=True)
        log_path = os.path.join(bench_dir, "benchmark.log")
        checklist_path = os.path.join(bench_dir, "checklist.log")
        root = logging.getLogger()
        root.setLevel(logging.INFO)

        have_bench = any(getattr(h, cls._HANDLER_MARKER, False) for h in root.handlers)
        have_checklist = any(getattr(h, cls._CHECKLIST_MARKER, False) for h in root.handlers)

        if not have_bench:
            fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter(cls.FORMAT))
            setattr(fh, cls._HANDLER_MARKER, True)
            root.addHandler(fh)

        if not have_checklist:
            ch = logging.FileHandler(checklist_path, mode="a", encoding="utf-8")
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter(cls.FORMAT))
            ch.addFilter(_ChecklistFilter())
            setattr(ch, cls._CHECKLIST_MARKER, True)
            root.addHandler(ch)

        has_stream = any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            for h in root.handlers
        )
        if not has_stream:
            sh = logging.StreamHandler()
            sh.setLevel(logging.INFO)
            sh.setFormatter(logging.Formatter("%(levelname)s %(name)s - %(message)s"))
            root.addHandler(sh)
        return log_path
